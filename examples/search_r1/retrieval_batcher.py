# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Async request batching for the Search-R1 retrieval service.

This module deliberately has no FAISS, PyTorch, or Transformers imports so its
queueing and failure semantics can be tested in a lightweight CPU environment.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Protocol


SearchOutput = tuple[list[list[dict[str, Any]]], list[list[float]]]


class Retriever(Protocol):
    def search(self, queries: list[str], topk: int) -> SearchOutput: ...


@dataclass
class _PendingRequest:
    queries: list[str]
    topk: int
    future: asyncio.Future[SearchOutput]


_STOP = object()


class RetrieverOverloadedError(RuntimeError):
    """The bounded request queue could not accept work before its deadline."""


class RetrieverBatcher:
    """Serialize GPU retrieval while coalescing concurrent HTTP requests."""

    def __init__(
        self,
        create_retriever: Callable[[], Retriever],
        *,
        batch_wait_ms: float,
        max_batch_queries: int,
        max_pending_requests: int,
        queue_timeout_s: float,
    ) -> None:
        self.create_retriever = create_retriever
        self.retriever: Retriever | None = None
        self.batch_wait_s = batch_wait_ms / 1000.0
        self.max_batch_queries = max_batch_queries
        self.max_pending_requests = max_pending_requests
        self.queue_timeout_s = queue_timeout_s
        self.queue: asyncio.Queue[_PendingRequest | object] = asyncio.Queue(maxsize=max_pending_requests)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="search-r1-retriever")
        self.worker: asyncio.Task[None] | None = None
        self.accepting = False
        self.deferred: _PendingRequest | object | None = None
        self.state_lock = asyncio.Lock()
        self.metrics = {
            "request_count": 0,
            "completed_request_count": 0,
            "failed_request_count": 0,
            "cancelled_request_count": 0,
            "rejected_request_count": 0,
            "batch_count": 0,
            "query_count": 0,
            "max_batch_query_count": 0,
        }

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            self.retriever = await loop.run_in_executor(self.executor, self.create_retriever)
            self.worker = asyncio.create_task(self._run(), name="search-r1-retriever-batcher")
            self.accepting = True
        except BaseException:
            self.executor.shutdown(wait=True, cancel_futures=True)
            raise

    async def close(self) -> None:
        async with self.state_lock:
            self.accepting = False
            await self.queue.put(_STOP)
        if self.worker is not None:
            await self.worker
            await self.queue.join()
            self.worker = None
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.executor, self._release_retriever)
        self.executor.shutdown(wait=True, cancel_futures=True)

    def _release_retriever(self) -> None:
        self.retriever = None

    async def search(self, queries: list[str], topk: int) -> SearchOutput:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[SearchOutput] = loop.create_future()
        self.metrics["request_count"] += 1
        async with self.state_lock:
            if not self.accepting:
                raise RuntimeError("Search-R1 retriever batcher is not running.")
            try:
                item = _PendingRequest(queries=queries, topk=topk, future=future)
                try:
                    self.queue.put_nowait(item)
                except asyncio.QueueFull:
                    await asyncio.wait_for(self.queue.put(item), timeout=self.queue_timeout_s)
            except asyncio.CancelledError:
                future.cancel()
                self.metrics["cancelled_request_count"] += 1
                raise
            except asyncio.TimeoutError as exc:
                self.metrics["rejected_request_count"] += 1
                future.cancel()
                raise RetrieverOverloadedError("Search-R1 retriever request queue is full") from exc
        try:
            return await future
        except asyncio.CancelledError:
            if not future.done():
                future.cancel()
            self.metrics["cancelled_request_count"] += 1
            raise

    async def _collect_batch(self) -> tuple[list[_PendingRequest], bool]:
        first = self.deferred
        if first is None:
            first = await self.queue.get()
        else:
            self.deferred = None
        if first is _STOP:
            return [], True
        pending = [first]
        query_count = len(first.queries)
        deadline = asyncio.get_running_loop().time() + self.batch_wait_s
        while query_count < self.max_batch_queries:
            remaining_s = deadline - asyncio.get_running_loop().time()
            if remaining_s <= 0:
                break
            try:
                item = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=remaining_s)
                except asyncio.TimeoutError:
                    break
            if item is _STOP:
                return pending, True
            if query_count > 0 and query_count + len(item.queries) > self.max_batch_queries:
                self.deferred = item
                break
            pending.append(item)
            query_count += len(item.queries)
        return pending, False

    def _execute_batch(self, pending: list[_PendingRequest]) -> list[SearchOutput]:
        if self.retriever is None:
            raise RuntimeError("Search-R1 retriever is not initialized.")
        all_queries = [query for item in pending for query in item.queries]
        max_topk = max(item.topk for item in pending)
        all_results, all_scores = self.retriever.search(all_queries, max_topk)
        split_results = []
        offset = 0
        for item in pending:
            end = offset + len(item.queries)
            results = [documents[: item.topk] for documents in all_results[offset:end]]
            scores = [document_scores[: item.topk] for document_scores in all_scores[offset:end]]
            split_results.append((results, scores))
            offset = end
        return split_results

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            pending, stop = await self._collect_batch()
            active = [item for item in pending if not item.future.cancelled()]
            try:
                if active:
                    batch_query_count = sum(len(item.queries) for item in active)
                    self.metrics["batch_count"] += 1
                    self.metrics["query_count"] += batch_query_count
                    self.metrics["max_batch_query_count"] = max(
                        self.metrics["max_batch_query_count"], batch_query_count
                    )
                    outputs = await loop.run_in_executor(self.executor, self._execute_batch, active)
                    for item, output in zip(active, outputs):
                        if not item.future.done():
                            item.future.set_result(output)
                            self.metrics["completed_request_count"] += 1
            except Exception as exc:
                for item in active:
                    if not item.future.done():
                        item.future.set_exception(exc)
                        self.metrics["failed_request_count"] += 1
            finally:
                for _ in pending:
                    self.queue.task_done()
                if stop:
                    self.queue.task_done()
            if stop:
                return

    def status(self) -> dict[str, Any]:
        worker_ready = self.worker is not None and not self.worker.done()
        return {
            "ready": self.accepting and self.retriever is not None and worker_ready,
            "accepting": self.accepting,
            "queue_depth": self.queue.qsize(),
            "queue_capacity": self.max_pending_requests,
            **self.metrics,
        }
