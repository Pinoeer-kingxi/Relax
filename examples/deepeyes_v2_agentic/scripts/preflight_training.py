#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Preflight the committed eight-GPU DeepEyes V2 training recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


TRAIN_PARQUETS = (
    "perception_all_1.parquet",
    "perception_all_2.parquet",
    "perception_all_3.parquet",
    "perception_all_4.parquet",
    "perception_all_5.parquet",
    "reason.parquet",
    "search.parquet",
    "vstar_test.parquet",
)

PRIMARY_MODEL_NAME = "Qwen3.6-35B-A3B"
JUDGE_MODEL_NAME = "Qwen2.5-1.5B-Instruct"
DATASET_REVISION = "53b38b4b3bc3feb31706b0793745161588e42cba"
TRAINING_IMAGE_OCI_DIGEST = "sha256:cd431e1094646c347aa97bc9cb8c5ac8315d4e72b0b6321f1cafd1ae12e680c6"
COMMAND_TIMEOUT_S = 30.0
TRAINING_PROBE_TIMEOUT_S = 120.0

PRIMARY_MODEL_MANIFEST = {
    "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
    "config": {
        "model_type": "qwen3_5_moe",
        "architectures.0": "Qwen3_5MoeForConditionalGeneration",
        "text_config.model_type": "qwen3_5_moe_text",
        "text_config.num_hidden_layers": 40,
        "text_config.num_experts": 256,
        "text_config.num_experts_per_tok": 8,
        "vision_config.depth": 27,
        "vision_config.hidden_size": 1152,
        "image_token_id": 248056,
    },
    "shard_sizes": {
        "model-00001-of-00026.safetensors": 3996199712,
        "model-00002-of-00026.safetensors": 1284907696,
        "model-00003-of-00026.safetensors": 3357898360,
        "model-00004-of-00026.safetensors": 3370808712,
        "model-00005-of-00026.safetensors": 3357898360,
        "model-00006-of-00026.safetensors": 3959424904,
        "model-00007-of-00026.safetensors": 1096788232,
        "model-00008-of-00026.safetensors": 3946842008,
        "model-00009-of-00026.safetensors": 1096460848,
        "model-00010-of-00026.safetensors": 3946841992,
        "model-00011-of-00026.safetensors": 1096460752,
        "model-00012-of-00026.safetensors": 3409971080,
        "model-00013-of-00026.safetensors": 1633331664,
        "model-00014-of-00026.safetensors": 3422553872,
        "model-00015-of-00026.safetensors": 1633659224,
        "model-00016-of-00026.safetensors": 3946842136,
        "model-00017-of-00026.safetensors": 1096460608,
        "model-00018-of-00026.safetensors": 3946841992,
        "model-00019-of-00026.safetensors": 1096460808,
        "model-00020-of-00026.safetensors": 3409971072,
        "model-00021-of-00026.safetensors": 1633331744,
        "model-00022-of-00026.safetensors": 3370808752,
        "model-00023-of-00026.safetensors": 3357898392,
        "model-00024-of-00026.safetensors": 3370808752,
        "model-00025-of-00026.safetensors": 3832888256,
        "model-00026-of-00026.safetensors": 2231416848,
    },
    "shard_sha256": {
        "model-00001-of-00026.safetensors": "adee7bcb930aed22e0677e58d4873b48dadb1ed8001cb5c6a0487286eadb3478",
        "model-00002-of-00026.safetensors": "88f2dfd2b9e73e4b70be533dbf61bcfa3c9a0003758900fcbc9d9b96f5751d4b",
        "model-00003-of-00026.safetensors": "8f7d72178d3f4431864978e5bcfa4c6cb1c204bc00590644d90bb19d6d522eeb",
        "model-00004-of-00026.safetensors": "12d7db38689ba3c8af74b23ef8523eca41e0cd95db870583d0663a3ee8a6bd60",
        "model-00005-of-00026.safetensors": "a836047305d0f7a7b50f0815d09d5c03ec03d59ec2c763fcdc4bf7e9936bf902",
        "model-00006-of-00026.safetensors": "c9080d718e9c5f9e337443225aa417d4c24d00ae7995d76ee3f1cc296b557d15",
        "model-00007-of-00026.safetensors": "e8c05e23131b1dd45a455ec38cfac7db14667358268623c3938d00cf3e959a68",
        "model-00008-of-00026.safetensors": "4b6a6d495053089f4a80e7cbc82e848fba44e2c0c60122233d8fdff79fa7b296",
        "model-00009-of-00026.safetensors": "a31a954bb72d1c714e751bf0aabf2ff533f5a509693ebf7dd22ad6e90be46f67",
        "model-00010-of-00026.safetensors": "246560e66570fe746653b8443e245dc334c9b8b831ea43d2d9f1b7d98623994e",
        "model-00011-of-00026.safetensors": "7180392817fe3ecb3a27a1da43b7ff22c1a94806bac49975f9f122c3126df675",
        "model-00012-of-00026.safetensors": "043fb525f6625c2f2acb75e65a9959ee3fa7b6e3fdd2034b5cfe1859b01d3cfb",
        "model-00013-of-00026.safetensors": "33a20fb20a21379bf43c84a43105f9c0cc35bd50d740b1c302dcbe4b700f5425",
        "model-00014-of-00026.safetensors": "be823e33c5cb6120ad3769d081f34a2449dc2358041fca7c29d636c1ba19130d",
        "model-00015-of-00026.safetensors": "a89d547c6f9d0b535ee5ea2f2478f163089539f3f0dd330cb23d278a19d76123",
        "model-00016-of-00026.safetensors": "69fc3ae0316482288afdcdd0b9eb7d626703ae26f7567e89aa3fc8d1ffd4ff5b",
        "model-00017-of-00026.safetensors": "e356e3943cf3852b76bb8992e674f3256013e27d54b78e8250514151cdc29637",
        "model-00018-of-00026.safetensors": "9e5e63fd1cc7d6848330c1fa363dfcb661bbc2ac87e672d0e28b71c9cb7f3c7f",
        "model-00019-of-00026.safetensors": "708644ad34f1de727bf484f396944d8ec628645d52c183e9a992e65671685e21",
        "model-00020-of-00026.safetensors": "ca083a1d1aa64f8e8a785998f543a43374f13436dc85d396eee4e72c7a84e1ae",
        "model-00021-of-00026.safetensors": "ada4ae48f3d48fe01b4c53f2f82bce25e798a9631fd33959c881156fef2ccbce",
        "model-00022-of-00026.safetensors": "def207fb42d7db31efb512755557763c23233c6e4d4c433027cb5102a7bce2f7",
        "model-00023-of-00026.safetensors": "864d52ca7768a36f514069222e8de8626264ae124097ba8fcce5b5da2c6e2ed7",
        "model-00024-of-00026.safetensors": "391acd27420cdce5935ff18152423c70620d19dac3c39a5ef1a81d369f82d737",
        "model-00025-of-00026.safetensors": "778e7f76602f05042b69ba7f3ec91f1fdffef390540b16074041c258fb81d154",
        "model-00026-of-00026.safetensors": "1a97404220077ed3d4182e10385b152004cab608377f50cec9f54a6b8d28b613",
    },
}

JUDGE_MODEL_MANIFEST = {
    "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "config": {
        "model_type": "qwen2",
        "architectures.0": "Qwen2ForCausalLM",
        "hidden_size": 1536,
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "vocab_size": 151936,
        "torch_dtype": "bfloat16",
    },
    "weight_size": 3087467144,
    "weight_sha256": "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
}

EXPECTED_DATA_SOURCE = {
    "perception_all_1.parquet": "perception",
    "perception_all_2.parquet": "perception",
    "perception_all_3.parquet": "perception",
    "perception_all_4.parquet": "perception",
    "perception_all_5.parquet": "perception",
    "reason.parquet": "reason",
    "search.parquet": "search",
    "vstar_test.parquet": "vstar-test",
}
EXPECTED_PARQUET_ROWS = {
    "perception_all_1.parquet": 10000,
    "perception_all_2.parquet": 10000,
    "perception_all_3.parquet": 10000,
    "perception_all_4.parquet": 8468,
    "perception_all_5.parquet": 8128,
    "reason.parquet": 30937,
    "search.parquet": 4856,
    "vstar_test.parquet": 191,
}

REVISION_MARKER_FILES = (".hf_revision", "revision.txt", ".revision")
SIF_REQUIRED_MODULES = ("ipykernel", "PIL", "matplotlib", "autopep8", "numpy")
APP_REQUIRED_MODULES = ("jupyter_client", "openai", "PIL", "pyarrow", "yaml", "httpx")
TRAINING_REQUIRED_MODULES = (
    "ray",
    "torch",
    "sglang",
    "megatron",
    "transformer_engine",
    "flash_attn",
    "cuda.bindings.driver",
    "cuda.bindings.runtime",
)


def _gpu_rows(output: str) -> list[dict[str, Any]]:
    rows = []
    for raw_line in output.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 5:
            raise ValueError("unexpected nvidia-smi CSV output")
        index, name, total, free, utilization = parts
        rows.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(total),
                "memory_free_mib": int(free),
                "utilization_percent": int(utilization),
            }
        )
    return rows


def _check_path(path: Path, *, executable: bool = False) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "executable": bool(exists and os.access(path, os.X_OK)) if executable else None,
        "passed": bool(exists and (not executable or os.access(path, os.X_OK))),
    }


def _is_lfs_pointer(path: Path, size: int | None) -> bool:
    if size is None or not 0 < size <= 1024:
        return False
    try:
        with path.open("rb") as handle:
            return handle.read(200).startswith(b"version https://git-lfs.github.com/spec/v1\n")
    except OSError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_file(
    path: Path,
    *,
    executable: bool = False,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    check = _check_path(path, executable=executable)
    size = path.stat().st_size if path.is_file() else None
    is_lfs_pointer = _is_lfs_pointer(path, size)
    check["size_bytes"] = size
    check["expected_size_bytes"] = expected_size
    check["git_lfs_pointer"] = is_lfs_pointer
    size_ok = size == expected_size if expected_size is not None else bool(size and size > 0)
    actual_sha256 = None
    if expected_sha256 is not None and check["passed"] and size_ok and not is_lfs_pointer:
        actual_sha256 = _sha256(path)
    check["sha256"] = actual_sha256
    check["expected_sha256"] = expected_sha256
    hash_ok = actual_sha256 == expected_sha256 if expected_sha256 is not None else True
    check["passed"] = bool(check["passed"] and size_ok and not is_lfs_pointer and hash_ok)
    return check


def _run_command(command: list[str], *, timeout_s: float = COMMAND_TIMEOUT_S) -> dict[str, Any]:
    result: dict[str, Any] = {"command": command, "timeout_s": timeout_s, "passed": False}
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except FileNotFoundError as exc:
        result.update({"error_category": "command_not_found", "error": str(exc)})
        return result
    except subprocess.TimeoutExpired as exc:
        result.update(
            {
                "error_category": "timeout",
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
            }
        )
        return result
    except OSError as exc:
        result.update({"error_category": "os_error", "error": str(exc)})
        return result
    result.update(
        {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "passed": completed.returncode == 0,
        }
    )
    if completed.returncode != 0:
        result["error_category"] = "nonzero_exit"
    return result


def _json_config_check(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    file_check = _check_file(path)
    result: dict[str, Any] = {"file": file_check, "fields": {}, "passed": False}
    if not file_check["passed"]:
        result["error"] = "missing_or_invalid_config"
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result["error"] = f"invalid_config_json:{exc.msg}"
        return result
    for dotted_key, expected_value in expected.items():
        current: Any = payload
        for part in dotted_key.split("."):
            if isinstance(current, list) and part.isdigit():
                index = int(part)
                current = current[index] if index < len(current) else None
            elif isinstance(current, dict):
                current = current.get(part)
            else:
                current = None
        result["fields"][dotted_key] = {
            "expected": expected_value,
            "actual": current,
            "passed": current == expected_value,
        }
    result["passed"] = all(item["passed"] for item in result["fields"].values())
    return result


def _revision_marker(path: Path) -> dict[str, Any]:
    for marker_name in REVISION_MARKER_FILES:
        marker = path / marker_name
        if marker.is_file():
            revision = marker.read_text(encoding="utf-8").strip().splitlines()[0]
            return {"source": str(marker), "revision": revision, "passed": True}
    return {"source": None, "revision": None, "passed": False, "error": "missing_revision_marker"}


def _check_revision(path: Path, expected_revision: str) -> dict[str, Any]:
    if (path / ".git").is_dir():
        command = _run_command(["git", "-C", str(path), "rev-parse", "HEAD"])
        revision = command.get("stdout", "").strip() if command["passed"] else None
        return {
            "source": "git",
            "expected": expected_revision,
            "revision": revision,
            "command": command,
            "passed": revision == expected_revision,
        }
    marker = _revision_marker(path)
    marker.update({"expected": expected_revision, "passed": marker.get("revision") == expected_revision})
    return marker


def _check_indexed_safetensors(
    model_dir: Path,
    expected_shard_sizes: dict[str, int],
    expected_shard_sha256: dict[str, str],
) -> dict[str, Any]:
    index_path = model_dir / "model.safetensors.index.json"
    checks: dict[str, Any] = {
        "index": _check_file(index_path),
        "expected_shard_count": len(expected_shard_sizes),
        "shards": [],
        "missing_shards": [],
        "unexpected_shards": [],
        "passed": False,
    }
    if index_path.is_file():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            checks["error"] = f"invalid_weight_index_json:{exc.msg}"
            return checks
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            checks["error"] = "weight_index_missing_weight_map"
            return checks
        shard_names = sorted({name for name in weight_map.values() if isinstance(name, str)})
        if not shard_names:
            checks["error"] = "weight_index_has_no_shards"
            return checks
        expected_names = set(expected_shard_sizes)
        actual_names = set(shard_names)
        checks["unexpected_shards"] = sorted(actual_names - expected_names)
        checks["missing_from_index"] = sorted(expected_names - actual_names)
        shard_checks = [
            _check_file(
                model_dir / shard_name,
                expected_size=expected_shard_sizes[shard_name],
                expected_sha256=expected_shard_sha256[shard_name],
            )
            for shard_name in sorted(expected_names)
        ]
        checks["shards"] = shard_checks
        checks["missing_shards"] = [item["path"] for item in shard_checks if not item["passed"]]
        checks["passed"] = (
            checks["index"]["passed"]
            and not checks["unexpected_shards"]
            and not checks["missing_from_index"]
            and len(actual_names) == len(expected_shard_sizes)
            and all(item["passed"] for item in shard_checks)
        )
        return checks
    checks["error"] = "missing_weight_index"
    return checks


def _check_primary_model(model_dir: Path) -> dict[str, Any]:
    manifest = PRIMARY_MODEL_MANIFEST
    required_metadata = {
        "config": _json_config_check(model_dir / "config.json", manifest["config"]),
        "tokenizer_config": _check_file(model_dir / "tokenizer_config.json"),
    }
    revision = _check_revision(model_dir, manifest["revision"])
    weights = _check_indexed_safetensors(model_dir, manifest["shard_sizes"], manifest["shard_sha256"])
    passed = (
        model_dir.is_dir()
        and revision["passed"]
        and all(item["passed"] for item in required_metadata.values())
        and weights["passed"]
    )
    return {
        "path": str(model_dir),
        "exists": model_dir.is_dir(),
        "revision": revision,
        "metadata": required_metadata,
        "weights": weights,
        "passed": passed,
    }


def _check_judge_model(model_dir: Path) -> dict[str, Any]:
    manifest = JUDGE_MODEL_MANIFEST
    required_metadata = {
        "config": _json_config_check(model_dir / "config.json", manifest["config"]),
        "tokenizer_config": _check_file(model_dir / "tokenizer_config.json"),
    }
    revision = _check_revision(model_dir, manifest["revision"])
    weights = _check_file(
        model_dir / "model.safetensors",
        expected_size=manifest["weight_size"],
        expected_sha256=manifest["weight_sha256"],
    )
    passed = (
        model_dir.is_dir()
        and revision["passed"]
        and all(item["passed"] for item in required_metadata.values())
        and weights["passed"]
    )
    return {
        "path": str(model_dir),
        "exists": model_dir.is_dir(),
        "revision": revision,
        "metadata": required_metadata,
        "weights": {"single_weight": weights, "passed": weights["passed"]},
        "passed": passed,
    }


def _expected_data_source(path: Path) -> str:
    return EXPECTED_DATA_SOURCE[path.name]


def _check_parquet(path: Path) -> dict[str, Any]:
    file_check = _check_file(path)
    result: dict[str, Any] = {
        "path": str(path),
        "file": file_check,
        "expected_data_source": _expected_data_source(path),
        "expected_num_rows": EXPECTED_PARQUET_ROWS[path.name],
        "passed": False,
    }
    if not file_check["passed"]:
        result["error"] = "missing_or_empty_parquet"
        return result
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as exc:
        result.update({"error": "pyarrow_unavailable", "detail": str(exc)})
        return result
    try:
        parquet_file = pq.ParquetFile(path)
    except Exception as exc:  # noqa: BLE001
        result.update({"error": "invalid_parquet", "detail": type(exc).__name__})
        return result
    schema_names = parquet_file.schema_arrow.names
    required_columns = ("prompt", "images", "reward_model", "extra_info")
    missing_columns = [name for name in required_columns if name not in schema_names]
    result.update(
        {
            "num_rows": parquet_file.metadata.num_rows,
            "num_row_groups": parquet_file.num_row_groups,
            "columns": schema_names,
            "missing_columns": missing_columns,
        }
    )
    if parquet_file.metadata.num_rows <= 0:
        result["error"] = "empty_parquet"
        return result
    if parquet_file.metadata.num_rows != result["expected_num_rows"]:
        result["error"] = "unexpected_row_count"
        return result
    if missing_columns:
        result["error"] = "missing_required_columns"
        return result
    try:
        extra_info_type = parquet_file.schema_arrow.field("extra_info").type
        extra_info_fields = [extra_info_type.field(index).name for index in range(extra_info_type.num_fields)]
    except Exception as exc:  # noqa: BLE001
        result.update({"error": "invalid_extra_info_schema", "detail": type(exc).__name__})
        return result
    result["extra_info_fields"] = extra_info_fields
    if "data_source" not in extra_info_fields:
        result["error"] = "missing_extra_info_data_source"
        return result
    try:
        table = parquet_file.read_row_group(0, columns=["extra_info"])
        sample = table.column("extra_info").chunks[0][0].as_py()
    except Exception as exc:  # noqa: BLE001
        result.update({"error": "data_source_sample_failed", "detail": type(exc).__name__})
        return result
    actual_data_source = sample.get("data_source") if isinstance(sample, dict) else None
    result["sample_data_source"] = actual_data_source
    result["passed"] = actual_data_source == result["expected_data_source"]
    if not result["passed"]:
        result["error"] = "unexpected_data_source"
    return result


def _check_dataset(data_dir: Path) -> dict[str, Any]:
    converted_checks = [_check_parquet(data_dir / "data" / name) for name in TRAIN_PARQUETS]
    provenance = _check_revision(data_dir / "data" / "raw", DATASET_REVISION)
    return {
        "raw_provenance": provenance,
        "converted": converted_checks,
        "passed": provenance["passed"] and all(item["passed"] for item in converted_checks),
    }


def _python_probe(
    command_prefix: list[str],
    modules: tuple[str, ...],
    *,
    include_cuda: bool = False,
    timeout_s: float = COMMAND_TIMEOUT_S,
) -> dict[str, Any]:
    module_list = ", ".join(repr(module) for module in modules)
    script = (
        "import importlib, json\n"
        f"mods=[{module_list}]\n"
        "out={}\n"
        "for name in mods:\n"
        "    module=importlib.import_module(name)\n"
        "    out[name]=getattr(module, '__version__', 'unknown')\n"
        "try:\n"
        "    import torch\n"
        "    out['torch.cuda.is_available']=torch.cuda.is_available()\n"
        "except Exception:\n"
        "    pass\n"
        "print(json.dumps(out, sort_keys=True))\n"
    )
    command = _run_command([*command_prefix, "-c", script], timeout_s=timeout_s)
    versions: dict[str, Any] = {}
    if command["passed"]:
        try:
            versions = json.loads(command.get("stdout", "{}"))
        except json.JSONDecodeError:
            command["passed"] = False
            command["error_category"] = "invalid_json"
    module_status = {module: module in versions for module in modules}
    cuda_available = versions.get("torch.cuda.is_available") is True
    if not include_cuda:
        versions.pop("torch.cuda.is_available", None)
    return {
        "modules": module_status,
        "versions": versions,
        "command": command,
        "passed": bool(command["passed"] and all(module_status.values()) and (not include_cuda or cuda_available)),
    }


def _probe_python(python_path: Path, modules: tuple[str, ...], *, include_cuda: bool = False) -> dict[str, Any]:
    result = _python_probe([str(python_path)], modules, include_cuda=include_cuda)
    result["python"] = _check_file(python_path, executable=True)
    result["passed"] = bool(result["passed"] and result["python"]["passed"])
    return result


def _check_training_image(image_path: Path | None, apptainer: str | None) -> dict[str, Any]:
    if image_path is None:
        return {"path": None, "passed": False, "error": "training_image_not_configured"}
    file_check = _check_file(image_path)
    digest_marker_path = Path(f"{image_path}.oci-digest")
    digest_marker = _check_file(digest_marker_path)
    actual_digest = None
    if digest_marker["passed"]:
        actual_digest = digest_marker_path.read_text(encoding="utf-8").strip()
    digest_check = {
        "path": str(digest_marker_path),
        "expected": TRAINING_IMAGE_OCI_DIGEST,
        "actual": actual_digest,
        "passed": digest_marker["passed"] and actual_digest == TRAINING_IMAGE_OCI_DIGEST,
    }
    result: dict[str, Any] = {
        "path": str(image_path),
        "file": file_check,
        "oci_digest": digest_check,
        "passed": False,
    }
    if not file_check["passed"]:
        result["error"] = "missing_or_empty_training_image"
        return result
    if apptainer is None:
        result["error"] = "apptainer_not_found"
        return result
    inspect = _run_command([apptainer, "inspect", "--json", str(image_path)])
    python_probe = _python_probe(
        [apptainer, "exec", "--nv", str(image_path), "python"],
        TRAINING_REQUIRED_MODULES,
        include_cuda=True,
        timeout_s=TRAINING_PROBE_TIMEOUT_S,
    )
    result.update({"inspect": inspect, "python_probe": python_probe})
    if inspect["passed"]:
        try:
            result["inspect_json"] = json.loads(inspect.get("stdout", "{}"))
        except json.JSONDecodeError:
            inspect["passed"] = False
            inspect["error_category"] = "invalid_json"
    result["passed"] = bool(
        file_check["passed"] and digest_check["passed"] and inspect["passed"] and python_probe["passed"]
    )
    return result


def _check_sif(sif_path: Path, apptainer: str | None) -> dict[str, Any]:
    file_check = _check_file(sif_path)
    result: dict[str, Any] = {"path": str(sif_path), "file": file_check, "passed": False}
    if not file_check["passed"]:
        result["error"] = "missing_or_empty_sif"
        return result
    if apptainer is None:
        result["error"] = "apptainer_not_found"
        return result
    inspect = _run_command([apptainer, "inspect", "--json", str(sif_path)])
    exec_probe = _run_command(
        [
            apptainer,
            "exec",
            str(sif_path),
            "python",
            "-c",
            "import " + ", ".join(SIF_REQUIRED_MODULES) + "; print('ok')",
        ]
    )
    result.update({"inspect": inspect, "exec_probe": exec_probe, "required_modules": SIF_REQUIRED_MODULES})
    if inspect["passed"]:
        try:
            result["inspect_json"] = json.loads(inspect.get("stdout", "{}"))
        except json.JSONDecodeError:
            inspect["passed"] = False
            inspect["error_category"] = "invalid_json"
    result["passed"] = bool(inspect["passed"] and exec_probe["passed"])
    return result


def evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    gpu_command = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu_error = None
    rows: list[dict[str, Any]] = []
    if gpu_command["passed"]:
        try:
            rows = _gpu_rows(gpu_command.get("stdout", ""))
        except ValueError as exc:
            gpu_error = str(exc)
    else:
        gpu_error = gpu_command.get("error_category", "nvidia-smi_failed")
    idle = [
        row
        for row in rows
        if row["memory_free_mib"] >= args.min_free_mib and row["utilization_percent"] <= args.max_utilization
    ]

    model_dir = args.model_dir / PRIMARY_MODEL_NAME
    judge_model_dir = args.model_dir / JUDGE_MODEL_NAME
    sif_path = args.sif_path or Path(
        os.environ.get("APPTAINER_IMAGE_PATH", str(args.data_dir / "sif" / "deepeyes_v2_kernel.sif"))
    )
    app_python = (
        Path(os.environ.get("DEEPEYES_V2_APP_ENV_ROOT", "/tmp/deepeyes-v2-app-env")) / ".venv" / "bin" / "python"
    )
    command_checks = {name: shutil.which(name) for name in ("apptainer", "nvidia-smi")}
    training_image = _check_training_image(args.training_image, command_checks["apptainer"])
    python_probes = {
        "app_python": _probe_python(app_python, APP_REQUIRED_MODULES),
        "training_python": training_image.get(
            "python_probe",
            {
                "modules": dict.fromkeys(TRAINING_REQUIRED_MODULES, False),
                "versions": {},
                "passed": False,
            },
        ),
    }
    path_checks = {
        "model": _check_primary_model(model_dir),
        "judge_model": _check_judge_model(judge_model_dir),
        "app_python": python_probes["app_python"],
        "data": _check_dataset(args.data_dir),
        "sif": _check_sif(sif_path, command_checks["apptainer"]),
        "training_image": training_image,
        "save_parent": {
            "path": str(args.save_dir.parent),
            "passed": args.save_dir.parent.is_dir() and os.access(args.save_dir.parent, os.W_OK),
        },
    }
    checks = {
        "gpu_inventory_readable": gpu_error is None,
        "enough_idle_gpus": len(idle) >= args.required_gpus,
        "model_present": path_checks["model"]["passed"],
        "judge_model_present": path_checks["judge_model"]["passed"],
        "data_present": path_checks["data"]["passed"],
        "sandbox_present": path_checks["sif"]["passed"],
        "training_image_present": path_checks["training_image"]["passed"],
        "app_python_present": path_checks["app_python"]["passed"],
        "training_python_modules_present": python_probes["training_python"]["passed"],
        "save_parent_writable": path_checks["save_parent"]["passed"],
        "commands_present": all(command_checks.values()),
    }
    report = {
        "schema_version": 1,
        "recipe": "Qwen3.6-35B-A3B DeepEyes-V2 eight-GPU colocated GRPO",
        "required_gpu_count": args.required_gpus,
        "gpu_idle_thresholds": {
            "minimum_free_memory_mib": args.min_free_mib,
            "maximum_utilization_percent": args.max_utilization,
        },
        "gpus": rows,
        "selected_gpu_indices": [row["index"] for row in idle[: args.required_gpus]],
        "gpu_error": gpu_error,
        "gpu_command": gpu_command,
        "paths": path_checks,
        "commands": {name: path is not None for name, path in command_checks.items()},
        "command_paths": command_checks,
        "python_probes": python_probes,
        "checks": checks,
        "status": "ready" if all(checks.values()) else "blocked",
    }
    return report, 0 if report["status"] == "ready" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument(
        "--sif-path",
        type=Path,
        help="Optional exact Apptainer image path to validate instead of DATA_DIR/sif/deepeyes_v2_kernel.sif.",
    )
    parser.add_argument(
        "--training-image",
        type=Path,
        default=Path(os.environ["RELAX_TRAINING_IMAGE_PATH"]) if os.environ.get("RELAX_TRAINING_IMAGE_PATH") else None,
        help=(
            "Pinned Relax training SIF. The adjacent <path>.oci-digest file must contain the committed OCI digest. "
            "Defaults to RELAX_TRAINING_IMAGE_PATH."
        ),
    )
    parser.add_argument("--required-gpus", type=int, default=8)
    parser.add_argument("--min-free-mib", type=int, default=45000)
    parser.add_argument("--max-utilization", type=int, default=5)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.required_gpus < 1 or args.min_free_mib < 1 or not 0 <= args.max_utilization <= 100:
        parser.error("GPU count/free-memory must be positive and utilization must be in [0, 100]")
    return args


def main() -> int:
    args = parse_args()
    try:
        report, exit_code = evaluate(args)
    except Exception as exc:  # noqa: BLE001
        report = {"schema_version": 1, "status": "blocked", "error_category": type(exc).__name__}
        exit_code = 2
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
