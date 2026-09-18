# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""E5 and GPU FAISS retrieval service.

Adapted from ``search_r1/search/retrieval_server.py`` in
https://github.com/PeterGriffinJin/Search-R1 at commit
``598e61bd1d36895726d28a8d06b3a15bed19f5d3`` (Apache-2.0).
"""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from functools import partial
from typing import Any

import datasets
import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, StrictInt, StrictStr
from transformers import AutoModel, AutoTokenizer


try:
    from .retrieval_batcher import RetrieverBatcher, RetrieverOverloadedError
    from .retrieval_validation import RetrievalValidationError, validate_retrieval_request
except ImportError:  # Direct ``python examples/search_r1/retrieval_server.py`` execution.
    from retrieval_batcher import RetrieverBatcher, RetrieverOverloadedError
    from retrieval_validation import RetrievalValidationError, validate_retrieval_request


class E5Encoder:
    def __init__(self, model_path: str) -> None:
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).cuda()
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

    @torch.no_grad()
    def encode(self, queries: list[str]) -> np.ndarray:
        inputs = self.tokenizer(
            [f"query: {query}" for query in queries],
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.cuda() for key, value in inputs.items()}
        output = self.model(**inputs, return_dict=True)
        hidden = output.last_hidden_state.masked_fill(~inputs["attention_mask"][..., None].bool(), 0.0)
        embeddings = hidden.sum(dim=1) / inputs["attention_mask"].sum(dim=1)[..., None]
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings.float().cpu().numpy().astype(np.float32, order="C")


class E5FlatRetriever:
    def __init__(self, *, index_path: str, corpus_path: str, model_path: str, topk: int) -> None:
        index = faiss.read_index(index_path)
        clone_options = faiss.GpuMultipleClonerOptions()
        clone_options.useFloat16 = True
        clone_options.shard = True
        self.index = faiss.index_cpu_to_all_gpus(index, co=clone_options)
        self.corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
        if self.index.ntotal != len(self.corpus):
            raise ValueError(
                f"retrieval index/corpus size mismatch: index={self.index.ntotal}, corpus={len(self.corpus)}"
            )
        self.encoder = E5Encoder(model_path)
        self.topk = topk

    def search(
        self, queries: list[str], topk: int | None = None
    ) -> tuple[list[list[dict[str, Any]]], list[list[float]]]:
        topk = self.topk if topk is None else topk
        if not queries:
            return [], []
        if topk < 1 or topk > self.index.ntotal:
            raise ValueError(f"topk must be in [1, {self.index.ntotal}]")
        results: list[list[dict[str, Any]]] = []
        scores: list[list[float]] = []
        for start in range(0, len(queries), 512):
            embeddings = self.encoder.encode(queries[start : start + 512])
            batch_scores, batch_indices = self.index.search(embeddings, k=topk)
            index_rows = batch_indices.tolist()
            invalid_indices = [index for row in index_rows for index in row if index < 0 or index >= len(self.corpus)]
            if invalid_indices:
                raise RuntimeError("FAISS returned an out-of-range document index")
            documents = [self.corpus[int(index)] for row in index_rows for index in row]
            results.extend(documents[offset : offset + topk] for offset in range(0, len(documents), topk))
            scores.extend(batch_scores.tolist())
        return results, scores


class QueryRequest(BaseModel):
    queries: list[StrictStr]
    topk: StrictInt | None = None
    return_scores: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    batcher = RetrieverBatcher(
        app.state.create_retriever,
        batch_wait_ms=app.state.batch_wait_ms,
        max_batch_queries=app.state.max_batch_queries,
        max_pending_requests=app.state.max_pending_requests,
        queue_timeout_s=app.state.queue_timeout_s,
    )
    await batcher.start()
    app.state.batcher = batcher
    try:
        yield
    finally:
        await batcher.close()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(raw_request: Request) -> dict[str, Any]:
    batcher: RetrieverBatcher | None = getattr(raw_request.app.state, "batcher", None)
    if batcher is None or not batcher.status()["ready"]:
        raise HTTPException(status_code=503, detail="retriever is not ready")
    return batcher.status()


@app.get("/metrics")
async def metrics(raw_request: Request) -> dict[str, Any]:
    batcher: RetrieverBatcher | None = getattr(raw_request.app.state, "batcher", None)
    if batcher is None:
        raise HTTPException(status_code=503, detail="retriever is not ready")
    return batcher.status()


@app.post("/retrieve")
async def retrieve(request: QueryRequest, raw_request: Request) -> dict[str, Any]:
    batcher: RetrieverBatcher = raw_request.app.state.batcher
    try:
        queries, topk = validate_retrieval_request(
            request.queries,
            request.topk,
            default_topk=batcher.retriever.topk,
            max_topk=raw_request.app.state.max_topk,
            max_batch_queries=batcher.max_batch_queries,
            index_size=int(batcher.retriever.index.ntotal),
        )
    except RetrievalValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        results, scores = await batcher.search(queries, topk)
    except RetrieverOverloadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="retriever inference failed") from exc
    if request.return_scores:
        rows = [
            [{"document": document, "score": score} for document, score in zip(documents, document_scores)]
            for documents, document_scores in zip(results, scores)
        ]
    else:
        rows = results
    return {"result": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Search-R1 E5 Flat retriever.")
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--retriever_model", required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--port", type=int, default=17389)
    parser.add_argument("--batch_wait_ms", type=float, default=5.0)
    parser.add_argument("--max_batch_queries", type=int, default=512)
    parser.add_argument("--max_pending_requests", type=int, default=4096)
    parser.add_argument("--queue_timeout_s", type=float, default=1.0)
    parser.add_argument("--max_topk", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.topk < 1 or args.max_topk < 1 or args.topk > args.max_topk:
        raise SystemExit("--topk must be positive and no greater than --max_topk")
    if (
        args.batch_wait_ms < 0
        or args.max_batch_queries < 1
        or args.max_pending_requests < 1
        or args.queue_timeout_s <= 0
    ):
        raise SystemExit("batch wait must be non-negative and batch/queue limits must be positive")
    app.state.create_retriever = partial(
        E5FlatRetriever,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        model_path=args.retriever_model,
        topk=args.topk,
    )
    app.state.batch_wait_ms = args.batch_wait_ms
    app.state.max_batch_queries = args.max_batch_queries
    app.state.max_pending_requests = args.max_pending_requests
    app.state.queue_timeout_s = args.queue_timeout_s
    app.state.max_topk = args.max_topk
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
