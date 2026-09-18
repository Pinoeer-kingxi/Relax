#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build a small, reproducible E5/FAISS corpus for native search smoke tests.

The output is intentionally not a replacement for the official Search-R1 Wiki18
assets.  It is a bounded fixture that exercises the same E5 prefixes, pooling,
normalization, corpus schema, and Flat inner-product index used by the
repository's retrieval server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


DEFAULT_DATASET = "Salesforce/wikitext"
DEFAULT_DATASET_CONFIG = "wikitext-2-raw-v1"
DEFAULT_DATASET_SPLIT = "train"

GOLDEN_DOCUMENTS = (
    {
        "id": "task10-golden-hamlet",
        "title": "Hamlet",
        "text": "Hamlet is a tragedy written by William Shakespeare, probably between 1599 and 1601.",
        "url": "https://en.wikipedia.org/wiki/Hamlet",
    },
    {
        "id": "task10-golden-france",
        "title": "Paris",
        "text": "Paris is the capital and largest city of France.",
        "url": "https://en.wikipedia.org/wiki/Paris",
    },
    {
        "id": "task10-golden-moon",
        "title": "Apollo 11",
        "text": "Apollo 11 was the spaceflight that first landed humans on the Moon in 1969.",
        "url": "https://en.wikipedia.org/wiki/Apollo_11",
    },
    {
        "id": "task10-golden-relax",
        "title": "Relax Task 10 search fixture",
        "text": (
            "The DeepEyes-V2 Task 10 native search fixture uses an E5 encoder and a GPU FAISS index. "
            "Its marker phrase is cobalt telescope meadow."
        ),
        "url": "https://example.invalid/task10/native-search-fixture",
    },
)

GOLDEN_QUERIES = (
    {"query": "Who wrote Hamlet?", "expected_id": "task10-golden-hamlet"},
    {"query": "What is the capital of France?", "expected_id": "task10-golden-france"},
    {"query": "Which mission first landed humans on the Moon?", "expected_id": "task10-golden-moon"},
    {"query": "cobalt telescope meadow", "expected_id": "task10-golden-relax"},
)


class FixtureBuildError(ValueError):
    """Raised when fixture inputs cannot produce a valid matching index."""


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if len(text) < 80 or text.startswith("="):
        return None
    return text


def _dataset_documents(rows: Iterable[dict[str, Any]], limit: int) -> Iterator[dict[str, str]]:
    """Yield deterministic corpus rows from a dataset containing a text
    field."""
    emitted = 0
    seen: set[str] = set()
    for row in rows:
        text = _clean_text(row.get("text"))
        if text is None:
            continue
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        if digest in seen:
            continue
        seen.add(digest)
        yield {
            "id": f"dataset-{digest}",
            "title": f"Wikitext excerpt {emitted + 1}",
            "text": text,
            "url": "",
        }
        emitted += 1
        if emitted >= limit:
            return


def _masked_mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def _encode_documents(documents: list[dict[str, str]], *, model_path: str, batch_size: int, device: str) -> Any:
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True).to(device)
    model.eval()
    batches: list[Any] = []
    with torch.no_grad():
        for start in range(0, len(documents), batch_size):
            passages = [f"passage: {row['title']} {row['text']}" for row in documents[start : start + batch_size]]
            inputs = tokenizer(
                passages,
                max_length=256,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            output = model(**inputs, return_dict=True)
            embeddings = _masked_mean_pool(output.last_hidden_state, inputs["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            batches.append(embeddings.float().cpu().numpy())
    return np.ascontiguousarray(np.concatenate(batches, axis=0), dtype=np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_file_hashes(model_path: Path) -> dict[str, str]:
    if not model_path.is_dir():
        raise FixtureBuildError(f"model path is not a directory: {model_path}")
    files = sorted(path for path in model_path.iterdir() if path.is_file())
    if not files:
        raise FixtureBuildError(f"model path contains no files: {model_path}")
    return {path.name: _sha256(path) for path in files}


def _write_jsonl(path: Path, documents: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n")


def _load_dataset(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    kwargs: dict[str, Any] = {"split": args.dataset_split}
    if args.dataset_cache_dir is not None:
        kwargs["cache_dir"] = str(args.dataset_cache_dir)
    if args.dataset_file is not None:
        dataset_file = args.dataset_file.expanduser().resolve()
        if not dataset_file.is_file():
            raise FixtureBuildError(f"dataset file does not exist: {dataset_file}")
        from datasets import load_dataset

        return load_dataset("parquet", data_files={args.dataset_split: str(dataset_file)}, **kwargs)
    from datasets import load_dataset

    if args.dataset_revision is not None:
        kwargs["revision"] = args.dataset_revision
    return load_dataset(args.dataset, args.dataset_config, **kwargs)


def build_fixture(args: argparse.Namespace) -> dict[str, Any]:
    # Load FAISS before torch/transformers. GPU FAISS wheels may bundle a newer
    # cuBLAS than the host torch build, and the repository retrieval server uses
    # this same import order to keep the loaded CUDA symbols consistent.
    import faiss

    if args.max_documents < len(GOLDEN_DOCUMENTS):
        raise FixtureBuildError(f"--max-documents must be at least {len(GOLDEN_DOCUMENTS)}")
    if args.batch_size < 1:
        raise FixtureBuildError("--batch-size must be positive")
    output_dir = args.output_dir.resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    model_file_hashes = _model_file_hashes(model_path)
    output_files = (output_dir / "corpus.jsonl", output_dir / "index.faiss", output_dir / "manifest.json")
    existing = [path for path in output_files if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(path.name for path in existing)
        raise FixtureBuildError(f"output files already exist ({names}); pass --overwrite to replace them")

    documents = [dict(document) for document in GOLDEN_DOCUMENTS]
    remaining = args.max_documents - len(documents)
    if remaining:
        documents.extend(_dataset_documents(_load_dataset(args), remaining))
    if len(documents) != args.max_documents:
        raise FixtureBuildError(f"dataset produced only {len(documents)} usable documents")

    embeddings = _encode_documents(
        documents,
        model_path=str(model_path),
        batch_size=args.batch_size,
        device=args.device,
    )
    if embeddings.shape[0] != len(documents):
        raise FixtureBuildError("embedding count does not match corpus size")

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="task10-fixture-", dir=output_dir) as temporary:
        temporary_dir = Path(temporary)
        corpus_path = temporary_dir / "corpus.jsonl"
        index_path = temporary_dir / "index.faiss"
        manifest_path = temporary_dir / "manifest.json"
        _write_jsonl(corpus_path, documents)
        index = faiss.IndexFlatIP(int(embeddings.shape[1]))
        index.add(embeddings)
        faiss.write_index(index, str(index_path))
        manifest = {
            "schema_version": 1,
            "corpus": {
                "dataset": args.dataset,
                "dataset_config": args.dataset_config,
                "dataset_file": str(args.dataset_file.expanduser().resolve()) if args.dataset_file else None,
                "dataset_file_sha256": _sha256(args.dataset_file.expanduser().resolve())
                if args.dataset_file
                else None,
                "dataset_revision": args.dataset_revision,
                "dataset_split": args.dataset_split,
                "documents": len(documents),
                "path": "corpus.jsonl",
                "sha256": _sha256(corpus_path),
            },
            "index": {
                "dimension": int(embeddings.shape[1]),
                "metric": "inner_product",
                "normalized": True,
                "path": "index.faiss",
                "sha256": _sha256(index_path),
            },
            "encoder": {
                "model_file_sha256": model_file_hashes,
                "model_path": str(model_path),
                "model_revision": args.model_revision,
                "document_prefix": "passage: ",
                "query_prefix": "query: ",
                "pooling": "attention-mask mean",
                "max_length": 256,
            },
            "golden_queries": list(GOLDEN_QUERIES),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for temporary_path, final_path in zip((corpus_path, index_path, manifest_path), output_files, strict=True):
            os.replace(temporary_path, final_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", help="Immutable source revision recorded in the manifest.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--dataset-revision")
    parser.add_argument(
        "--dataset-file",
        type=Path,
        help="Use a pinned local Parquet file instead of downloading the named dataset.",
    )
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--dataset-cache-dir", type=Path)
    parser.add_argument("--max-documents", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        manifest = build_fixture(args)
    except FixtureBuildError as exc:
        raise SystemExit(f"fixture build failed: {exc}") from exc
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
