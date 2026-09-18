# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "examples" / "deepeyes_v2_agentic" / "scripts" / "preflight_training.py"
SPEC = importlib.util.spec_from_file_location("task10_preflight_training", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

PRIMARY_TEST_MANIFEST = {
    "revision": "primary-test-revision",
    "config": {
        "model_type": "qwen3_5_moe",
        "architectures.0": "Qwen3_5MoeForConditionalGeneration",
        "text_config.num_hidden_layers": 1,
    },
    "shard_sizes": {
        "model-00001-of-00002.safetensors": 7,
        "model-00002-of-00002.safetensors": 9,
    },
}
PRIMARY_TEST_MANIFEST["shard_sha256"] = {
    name: hashlib.sha256(b"x" * size).hexdigest() for name, size in PRIMARY_TEST_MANIFEST["shard_sizes"].items()
}
JUDGE_TEST_MANIFEST = {
    "revision": "judge-test-revision",
    "config": {
        "model_type": "qwen2",
        "architectures.0": "Qwen2ForCausalLM",
        "hidden_size": 16,
    },
    "weight_size": 11,
    "weight_sha256": hashlib.sha256(b"x" * 11).hexdigest(),
}


def test_gpu_rows_parses_stable_nvidia_smi_csv():
    rows = MODULE._gpu_rows("0, NVIDIA RTX A6000, 49140, 48500, 0\n7, NVIDIA RTX A6000, 49140, 20000, 80\n")
    assert rows[0] == {
        "index": 0,
        "name": "NVIDIA RTX A6000",
        "memory_total_mib": 49140,
        "memory_free_mib": 48500,
        "utilization_percent": 0,
    }
    assert rows[1]["index"] == 7


def test_gpu_rows_rejects_changed_output_shape():
    with pytest.raises(ValueError, match="unexpected"):
        MODULE._gpu_rows("0, incomplete")


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_bytes(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def _make_executable(path: Path) -> None:
    _write(path, "#!/bin/sh\n")
    path.chmod(0o755)


def _make_primary_model(path: Path) -> None:
    _write(path / ".hf_revision", PRIMARY_TEST_MANIFEST["revision"])
    _write(
        path / "config.json",
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "text_config": {"num_hidden_layers": 1},
            }
        ),
    )
    _write(path / "tokenizer_config.json", "{}")
    weight_map = {
        "layer_0": "model-00001-of-00002.safetensors",
        "layer_1": "model-00002-of-00002.safetensors",
    }
    _write(path / "model.safetensors.index.json", json.dumps({"weight_map": weight_map}))
    for shard_name, size in PRIMARY_TEST_MANIFEST["shard_sizes"].items():
        _write_bytes(path / shard_name, size)


def _make_judge_model(path: Path) -> None:
    _write(path / ".hf_revision", JUDGE_TEST_MANIFEST["revision"])
    _write(
        path / "config.json",
        json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"], "hidden_size": 16}),
    )
    _write(path / "tokenizer_config.json", "{}")
    _write_bytes(path / "model.safetensors", JUDGE_TEST_MANIFEST["weight_size"])


def _write_parquet(path: Path, data_source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
            ("images", pa.list_(pa.binary())),
            ("reward_model", pa.struct([("ground_truth", pa.string())])),
            ("extra_info", pa.struct([("index", pa.int64()), ("data_source", pa.string())])),
        ]
    )
    table = pa.Table.from_pydict(
        {
            "prompt": [[{"role": "user", "content": "question"}]],
            "images": [[b"image"]],
            "reward_model": [{"ground_truth": "answer"}],
            "extra_info": [{"index": 0, "data_source": data_source}],
        },
        schema=schema,
    )
    pq.write_table(table, path)


def _make_preflight_tree(tmp_path: Path) -> SimpleNamespace:
    model_root = tmp_path / "models"
    data_root = tmp_path / "assets"
    save_dir = tmp_path / "save" / "run"
    _make_primary_model(model_root / MODULE.PRIMARY_MODEL_NAME)
    _make_judge_model(model_root / MODULE.JUDGE_MODEL_NAME)
    _write(data_root / "data" / "raw" / ".hf_revision", MODULE.DATASET_REVISION)
    for parquet_name in MODULE.TRAIN_PARQUETS:
        _write_parquet(data_root / "data" / parquet_name, MODULE.EXPECTED_DATA_SOURCE[parquet_name])
    _write_bytes(data_root / "sif" / "deepeyes_v2_kernel.sif", 13)
    training_image = tmp_path / "images" / "relaxrl.sif"
    _write_bytes(training_image, 17)
    _write(Path(f"{training_image}.oci-digest"), MODULE.TRAINING_IMAGE_OCI_DIGEST)
    _make_executable(tmp_path / "app-env" / ".venv" / "bin" / "python")
    save_dir.parent.mkdir(parents=True)
    return SimpleNamespace(
        model_root=model_root,
        data_root=data_root,
        save_dir=save_dir,
        training_image=training_image,
    )


def _completed(
    command: list[str], stdout: str = "", returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode=returncode, stdout=stdout, stderr=stderr)


def _fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    assert kwargs["capture_output"] is True
    assert kwargs["check"] is False
    assert kwargs["text"] is True
    assert kwargs["timeout"] in {MODULE.COMMAND_TIMEOUT_S, MODULE.TRAINING_PROBE_TIMEOUT_S}
    if command[0] == "nvidia-smi":
        return _completed(command, "\n".join(f"{idx}, Test GPU, 49140, 48000, 0" for idx in range(8)))
    if command[0] == "/usr/bin/apptainer" and command[1:3] == ["inspect", "--json"]:
        return _completed(command, '{"data": {"attributes": {"inspect": "ok"}}}')
    if command[0] == "/usr/bin/apptainer" and command[1:3] == ["exec", "--nv"]:
        return _completed(
            command,
            '{"cuda.bindings.driver": "unknown", "cuda.bindings.runtime": "unknown", '
            '"flash_attn": "2.7.4", "megatron": "unknown", "ray": "2.9.0", "sglang": "0.5.17", '
            '"torch": "2.9.0", "torch.cuda.is_available": true, "transformer_engine": "2.14.1"}\n',
        )
    if command[0] == "/usr/bin/apptainer" and command[1] == "exec":
        assert "pandas" not in command[-1]
        return _completed(command, "ok\n")
    if command[0].endswith("python") and "jupyter_client" in command[-1]:
        return _completed(
            command,
            '{"PIL": "11.0.0", "httpx": "0.28.0", "jupyter_client": "8.0.0", '
            '"openai": "1.0.0", "pyarrow": "14.0.2", "yaml": "6.0.0"}\n',
        )
    if command[0] == "git":
        return _completed(command, PRIMARY_TEST_MANIFEST["revision"] + "\n")
    raise AssertionError(f"unexpected command: {command}")


@pytest.fixture()
def passing_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    tree = _make_preflight_tree(tmp_path)
    monkeypatch.setattr(MODULE, "PRIMARY_MODEL_MANIFEST", PRIMARY_TEST_MANIFEST)
    monkeypatch.setattr(MODULE, "JUDGE_MODEL_MANIFEST", JUDGE_TEST_MANIFEST)
    monkeypatch.setattr(MODULE, "EXPECTED_PARQUET_ROWS", dict.fromkeys(MODULE.TRAIN_PARQUETS, 1))
    monkeypatch.setenv("DEEPEYES_V2_APP_ENV_ROOT", str(tmp_path / "app-env"))
    monkeypatch.delenv("APPTAINER_IMAGE_PATH", raising=False)
    monkeypatch.setattr(MODULE.subprocess, "run", _fake_run)
    monkeypatch.setattr(MODULE.shutil, "which", lambda name: f"/usr/bin/{name}")
    return tree


def _args(tree: SimpleNamespace, **overrides: object) -> SimpleNamespace:
    values = {
        "model_dir": tree.model_root,
        "data_dir": tree.data_root,
        "save_dir": tree.save_dir,
        "sif_path": None,
        "training_image": tree.training_image,
        "required_gpus": 8,
        "min_free_mib": 45000,
        "max_utilization": 5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_evaluate_accepts_complete_exact_training_resources(passing_runtime: SimpleNamespace):
    report, exit_code = MODULE.evaluate(_args(passing_runtime))

    assert exit_code == 0
    assert report["status"] == "ready"
    assert report["paths"]["model"]["revision"]["revision"] == PRIMARY_TEST_MANIFEST["revision"]
    assert report["paths"]["model"]["weights"]["expected_shard_count"] == 2
    assert report["paths"]["judge_model"]["weights"]["single_weight"]["expected_size_bytes"] == 11
    assert len(report["paths"]["data"]["converted"]) == 8
    assert report["paths"]["sif"]["exec_probe"]["passed"] is True
    assert report["python_probes"]["training_python"]["versions"]["torch.cuda.is_available"] is True
    assert report["python_probes"]["training_python"]["modules"]["cuda.bindings.driver"] is True
    assert report["python_probes"]["training_python"]["modules"]["cuda.bindings.runtime"] is True


def test_evaluate_rejects_missing_revision_marker(passing_runtime: SimpleNamespace):
    (passing_runtime.model_root / MODULE.PRIMARY_MODEL_NAME / ".hf_revision").unlink()

    report, exit_code = MODULE.evaluate(_args(passing_runtime))

    assert exit_code == 1
    assert report["checks"]["model_present"] is False
    assert report["paths"]["model"]["revision"]["error"] == "missing_revision_marker"


def test_evaluate_accepts_git_head_revision(monkeypatch: pytest.MonkeyPatch, passing_runtime: SimpleNamespace):
    model_dir = passing_runtime.model_root / MODULE.PRIMARY_MODEL_NAME
    (model_dir / ".hf_revision").unlink()
    (model_dir / ".git").mkdir()

    report, exit_code = MODULE.evaluate(_args(passing_runtime))

    assert exit_code == 0
    assert report["paths"]["model"]["revision"]["source"] == "git"


def test_evaluate_rejects_wrong_qwen36_shard_size(passing_runtime: SimpleNamespace):
    shard = passing_runtime.model_root / MODULE.PRIMARY_MODEL_NAME / "model-00001-of-00002.safetensors"
    _write_bytes(shard, 1)

    report, exit_code = MODULE.evaluate(_args(passing_runtime))

    assert exit_code == 1
    failed_shard = report["paths"]["model"]["weights"]["shards"][0]
    assert failed_shard["size_bytes"] == 1
    assert failed_shard["expected_size_bytes"] == 7


def test_evaluate_rejects_same_size_corrupt_qwen36_shard(passing_runtime: SimpleNamespace):
    shard = passing_runtime.model_root / MODULE.PRIMARY_MODEL_NAME / "model-00001-of-00002.safetensors"
    shard.write_bytes(b"y" * 7)

    report, exit_code = MODULE.evaluate(_args(passing_runtime))

    assert exit_code == 1
    failed_shard = report["paths"]["model"]["weights"]["shards"][0]
    assert failed_shard["size_bytes"] == failed_shard["expected_size_bytes"] == 7
    assert failed_shard["sha256"] != failed_shard["expected_sha256"]


def test_evaluate_rejects_unmaterialized_git_lfs_pointer(passing_runtime: SimpleNamespace):
    pointer = passing_runtime.model_root / MODULE.PRIMARY_MODEL_NAME / "model-00001-of-00002.safetensors"
    _write(
        pointer,
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:adee7bcb930aed22e0677e58d4873b48dadb1ed8001cb5c6a0487286eadb3478\n"
        "size 7\n",
    )

    report, exit_code = MODULE.evaluate(_args(passing_runtime))

    assert exit_code == 1
    shard = report["paths"]["model"]["weights"]["shards"][0]
    assert shard["git_lfs_pointer"] is True


def test_evaluate_rejects_judge_config_mismatch(passing_runtime: SimpleNamespace):
    _write(
        passing_runtime.model_root / MODULE.JUDGE_MODEL_NAME / "config.json",
        json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"], "hidden_size": 32}),
    )

    report, exit_code = MODULE.evaluate(_args(passing_runtime))

    assert exit_code == 1
    field = report["paths"]["judge_model"]["metadata"]["config"]["fields"]["hidden_size"]
    assert field == {"expected": 16, "actual": 32, "passed": False}


def test_evaluate_rejects_text_file_posing_as_parquet(passing_runtime: SimpleNamespace):
    _write(passing_runtime.data_root / "data" / "search.parquet", "not parquet")

    report, exit_code = MODULE.evaluate(_args(passing_runtime))

    assert exit_code == 1
    failed = [item for item in report["paths"]["data"]["converted"] if item["path"].endswith("search.parquet")][0]
    assert failed["error"] == "invalid_parquet"


def test_evaluate_rejects_wrong_converted_data_source(passing_runtime: SimpleNamespace):
    _write_parquet(passing_runtime.data_root / "data" / "vstar_test.parquet", "vstar_test")

    report, exit_code = MODULE.evaluate(_args(passing_runtime))

    assert exit_code == 1
    failed = [item for item in report["paths"]["data"]["converted"] if item["path"].endswith("vstar_test.parquet")][0]
    assert failed["error"] == "unexpected_data_source"
    assert failed["expected_data_source"] == "vstar-test"


def test_evaluate_rejects_truncated_but_valid_parquet(
    monkeypatch: pytest.MonkeyPatch, passing_runtime: SimpleNamespace
):
    expected_rows = dict(MODULE.EXPECTED_PARQUET_ROWS)
    expected_rows["search.parquet"] = 2
    monkeypatch.setattr(MODULE, "EXPECTED_PARQUET_ROWS", expected_rows)

    report, exit_code = MODULE.evaluate(_args(passing_runtime))

    assert exit_code == 1
    failed = [item for item in report["paths"]["data"]["converted"] if item["path"].endswith("search.parquet")][0]
    assert failed["error"] == "unexpected_row_count"
    assert failed["num_rows"] == 1
    assert failed["expected_num_rows"] == 2


def test_evaluate_rejects_missing_raw_dataset_provenance(passing_runtime: SimpleNamespace):
    (passing_runtime.data_root / "data" / "raw" / ".hf_revision").unlink()

    report, exit_code = MODULE.evaluate(_args(passing_runtime))

    assert exit_code == 1
    assert report["paths"]["data"]["raw_provenance"]["error"] == "missing_revision_marker"


def test_evaluate_accepts_raw_dataset_git_revision(monkeypatch: pytest.MonkeyPatch, passing_runtime: SimpleNamespace):
    raw_dir = passing_runtime.data_root / "data" / "raw"
    (raw_dir / ".hf_revision").unlink()
    (raw_dir / ".git").mkdir()

    original_run = MODULE.subprocess.run

    def dataset_git_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["git", "-C", str(raw_dir)]:
            return _completed(command, MODULE.DATASET_REVISION + "\n")
        return original_run(command, **kwargs)

    monkeypatch.setattr(MODULE.subprocess, "run", dataset_git_run)

    report, exit_code = MODULE.evaluate(_args(passing_runtime))

    assert exit_code == 0
    assert report["paths"]["data"]["raw_provenance"]["source"] == "git"


def test_evaluate_checks_explicit_sif_path(passing_runtime: SimpleNamespace, tmp_path: Path):
    external_sif = tmp_path / "shared" / "fixed.sif"
    _write_bytes(external_sif, 3)

    report, exit_code = MODULE.evaluate(_args(passing_runtime, sif_path=external_sif))

    assert exit_code == 0
    assert report["paths"]["sif"]["path"] == str(external_sif)


def test_evaluate_rejects_sif_probe_timeout(monkeypatch: pytest.MonkeyPatch, passing_runtime: SimpleNamespace):
    def timeout_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[0] == "/usr/bin/apptainer" and command[1:3] == ["inspect", "--json"]:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return _fake_run(command, **kwargs)

    monkeypatch.setattr(MODULE.subprocess, "run", timeout_run)

    report, exit_code = MODULE.evaluate(_args(passing_runtime))

    assert exit_code == 1
    assert report["paths"]["sif"]["inspect"]["error_category"] == "timeout"


def test_evaluate_rejects_app_python_import_failure(monkeypatch: pytest.MonkeyPatch, passing_runtime: SimpleNamespace):
    def failed_app_python(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[0].endswith("python") and "jupyter_client" in command[-1]:
            return _completed(command, stderr="missing module", returncode=1)
        return _fake_run(command, **kwargs)

    monkeypatch.setattr(MODULE.subprocess, "run", failed_app_python)

    report, exit_code = MODULE.evaluate(_args(passing_runtime))

    assert exit_code == 1
    assert report["checks"]["app_python_present"] is False
    assert report["paths"]["app_python"]["command"]["error_category"] == "nonzero_exit"


def test_evaluate_rejects_training_python_import_failure(
    monkeypatch: pytest.MonkeyPatch, passing_runtime: SimpleNamespace
):
    def failed_training_python(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[0] == "/usr/bin/apptainer" and command[1:3] == ["exec", "--nv"]:
            return _completed(command, '{"ray": "2.9.0"}\n')
        return _fake_run(command, **kwargs)

    monkeypatch.setattr(MODULE.subprocess, "run", failed_training_python)

    report, exit_code = MODULE.evaluate(_args(passing_runtime))

    assert exit_code == 1
    assert report["checks"]["training_python_modules_present"] is False
    assert report["python_probes"]["training_python"]["modules"]["torch"] is False


def test_evaluate_rejects_training_python_without_cuda(
    monkeypatch: pytest.MonkeyPatch, passing_runtime: SimpleNamespace
):
    def cpu_only_training_python(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[0] == "/usr/bin/apptainer" and command[1:3] == ["exec", "--nv"]:
            return _completed(
                command,
                '{"flash_attn": "2.7.4", "megatron": "unknown", "ray": "2.9.0", "sglang": "0.5.17", '
                '"torch": "2.9.0", "torch.cuda.is_available": false, "transformer_engine": "2.14.1"}\n',
            )
        return _fake_run(command, **kwargs)

    monkeypatch.setattr(MODULE.subprocess, "run", cpu_only_training_python)

    report, exit_code = MODULE.evaluate(_args(passing_runtime))

    assert exit_code == 1
    assert report["checks"]["training_python_modules_present"] is False
    assert report["python_probes"]["training_python"]["versions"]["torch.cuda.is_available"] is False


def test_evaluate_rejects_missing_training_image(passing_runtime: SimpleNamespace):
    passing_runtime.training_image.unlink()

    report, exit_code = MODULE.evaluate(_args(passing_runtime))

    assert exit_code == 1
    assert report["checks"]["training_image_present"] is False
    assert report["paths"]["training_image"]["error"] == "missing_or_empty_training_image"


def test_evaluate_rejects_wrong_training_image_digest(passing_runtime: SimpleNamespace):
    _write(Path(f"{passing_runtime.training_image}.oci-digest"), "sha256:wrong")

    report, exit_code = MODULE.evaluate(_args(passing_runtime))

    assert exit_code == 1
    assert report["checks"]["training_image_present"] is False
    assert report["paths"]["training_image"]["oci_digest"]["actual"] == "sha256:wrong"
