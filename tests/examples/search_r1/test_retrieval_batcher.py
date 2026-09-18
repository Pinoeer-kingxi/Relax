# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


SEARCH_R1_DIR = Path(__file__).resolve().parents[3] / "examples" / "search_r1"
sys.path.insert(0, str(SEARCH_R1_DIR))

from retrieval_batcher import RetrieverBatcher, RetrieverOverloadedError  # noqa: E402


class _FakeRetriever:
    def __init__(self, *, fail: bool = False) -> None:
        self.topk = 3
        self.fail = fail
        self.calls: list[tuple[list[str], int]] = []

    def search(self, queries, topk):
        self.calls.append((list(queries), topk))
        if self.fail:
            raise RuntimeError("inference failed")
        results = [[{"title": f"{query}-{index}"} for index in range(topk)] for query in queries]
        scores = [[float(topk - index) for index in range(topk)] for _query in queries]
        return results, scores


def _batcher(retriever: _FakeRetriever, **overrides) -> RetrieverBatcher:
    values = {
        "batch_wait_ms": 20,
        "max_batch_queries": 8,
        "max_pending_requests": 8,
        "queue_timeout_s": 0.05,
    }
    values.update(overrides)
    return RetrieverBatcher(lambda: retriever, **values)


def test_batcher_combines_requests_and_splits_mixed_topk():
    async def scenario():
        retriever = _FakeRetriever()
        batcher = _batcher(retriever)
        await batcher.start()
        try:
            first, second = await asyncio.gather(
                batcher.search(["a"], 1),
                batcher.search(["b", "c"], 2),
            )
            assert len(retriever.calls) == 1
            assert retriever.calls[0] == (["a", "b", "c"], 2)
            assert len(first[0][0]) == 1
            assert [len(row) for row in second[0]] == [2, 2]
            status = batcher.status()
            assert status["completed_request_count"] == 2
            assert status["batch_count"] == 1
            assert status["query_count"] == 3
            assert status["max_batch_query_count"] == 3
            assert status["ready"] is True
        finally:
            await batcher.close()
        assert batcher.status()["ready"] is False

    asyncio.run(scenario())


def test_batcher_propagates_inference_failure_to_every_request():
    async def scenario():
        retriever = _FakeRetriever(fail=True)
        batcher = _batcher(retriever)
        await batcher.start()
        try:
            results = await asyncio.gather(
                batcher.search(["a"], 1),
                batcher.search(["b"], 1),
                return_exceptions=True,
            )
            assert all(isinstance(result, RuntimeError) for result in results)
            assert batcher.status()["failed_request_count"] == 2
        finally:
            await batcher.close()

    asyncio.run(scenario())


def test_batcher_rejects_when_bounded_queue_stays_full():
    async def scenario():
        retriever = _FakeRetriever()
        batcher = _batcher(retriever, max_pending_requests=1, queue_timeout_s=0.001)
        batcher.retriever = retriever
        batcher.accepting = True
        await batcher.queue.put(object())
        try:
            with pytest.raises(RetrieverOverloadedError, match="queue is full"):
                await batcher.search(["blocked"], 1)
            assert batcher.status()["rejected_request_count"] == 1
        finally:
            batcher.accepting = False
            batcher.executor.shutdown(wait=True, cancel_futures=True)

    asyncio.run(scenario())


def test_batcher_tracks_cancelled_waiters_without_running_them():
    async def scenario():
        retriever = _FakeRetriever()
        batcher = _batcher(retriever, batch_wait_ms=100)
        await batcher.start()
        try:
            task = asyncio.create_task(batcher.search(["cancelled"], 1))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0.15)
            assert retriever.calls == []
            assert batcher.status()["cancelled_request_count"] == 1
        finally:
            await batcher.close()

    asyncio.run(scenario())


def test_batcher_tracks_cancellation_while_waiting_for_queue_capacity():
    async def scenario():
        retriever = _FakeRetriever()
        batcher = _batcher(retriever, max_pending_requests=1, queue_timeout_s=10)
        batcher.retriever = retriever
        batcher.accepting = True
        await batcher.queue.put(object())
        try:
            task = asyncio.create_task(batcher.search(["blocked"], 1))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert batcher.status()["cancelled_request_count"] == 1
            assert batcher.queue.qsize() == 1
        finally:
            batcher.accepting = False
            batcher.executor.shutdown(wait=True, cancel_futures=True)

    asyncio.run(scenario())
