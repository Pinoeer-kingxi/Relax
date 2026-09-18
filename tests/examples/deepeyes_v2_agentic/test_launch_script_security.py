# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCH_SCRIPT = REPO_ROOT / "examples/deepeyes_v2_agentic/run_deepeyes_v2_agentic.sh"
JUDGE_SCRIPT = REPO_ROOT / "examples/deepeyes_v2_agentic/sglang_judge_service.sh"


def test_training_launcher_keeps_runtime_secrets_out_of_process_arguments() -> None:
    script = LAUNCH_SCRIPT.read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert 'chmod 600 "${RUNTIME_ENV_FILE}"' in script
    assert '--runtime-env "${RUNTIME_ENV_FILE}"' in script
    assert '--runtime-env-json "${RUNTIME_ENV_JSON}"' not in script
    assert "declare -F cleanup_sglang_judge" in script
    assert "cleanup_sglang_judge || true" in script


def test_judge_service_does_not_inject_inline_runtime_environment() -> None:
    script = JUDGE_SCRIPT.read_text(encoding="utf-8")

    assert "--runtime-env-json" not in script
    assert "RUNTIME_ENV_JSON" not in script


def test_external_judge_secret_is_not_traced(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_curl = fake_bin / "curl"
    fake_curl.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_curl.chmod(0o755)

    data_dir = tmp_path / "data"
    save_dir = tmp_path / "save"
    model_dir = tmp_path / "models"
    app_python = tmp_path / "app-env/.venv/bin/python"
    sandbox_image = tmp_path / "sandbox.sif"
    app_python.parent.mkdir(parents=True)
    app_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    app_python.chmod(0o755)
    data_dir.mkdir()
    model_dir.mkdir()
    sandbox_image.touch()

    secret = "judge-secret-must-not-appear"
    env = os.environ.copy()
    env.update(
        {
            "APPTAINER_IMAGE_PATH": str(sandbox_image),
            "DATA_DIR": str(data_dir),
            "DEEPEYES_JUDGE_API_KEY": secret,
            "DEEPEYES_JUDGE_BASE_URL": "http://judge.invalid/v1",
            "DEEPEYES_JUDGE_MODELS": "judge-model",
            "DEEPEYES_TRACE": "1",
            "DEEPEYES_V2_APP_ENV_ROOT": str(tmp_path / "app-env"),
            "MODEL_CONFIG_DIR": str(REPO_ROOT / "scripts/models"),
            "MODEL_DIR": str(model_dir),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "RELAX_ENTRYPOINT_MODE": "test",
            "SAVE_DIR": str(save_dir),
            "START_LOCAL_JUDGE": "0",
        }
    )
    completed = subprocess.run(
        ["bash", str(LAUNCH_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert secret not in completed.stdout
    assert secret not in completed.stderr
    assert "Using configured external judge" in completed.stdout
