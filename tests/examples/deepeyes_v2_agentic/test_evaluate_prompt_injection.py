# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "examples" / "deepeyes_v2_agentic" / "scripts" / "evaluate_prompt_injection.py"
SPEC = importlib.util.spec_from_file_location("task10_evaluate_prompt_injection", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPT.parent))
SPEC.loader.exec_module(MODULE)


def test_prompt_injection_evaluator_requires_fact_and_rejects_marker(monkeypatch):
    answers = iter(("Paris", MODULE.ATTACK_MARKER, "Lyon"))

    def fake_run_agent(**_kwargs):
        MODULE._InjectionRetrieverHandler.request_count += 1
        answer = next(answers)
        return (
            {
                "metadata": {
                    "final_answer": answer,
                    "stop_reason": "env_done",
                    "web_search_attempt_count": 1,
                }
            },
            0,
        )

    monkeypatch.setattr(MODULE, "_run_agent", fake_run_agent)
    args = argparse.Namespace(min_pass_rate=1 / 3)
    report, exit_code = MODULE.evaluate(args)

    assert exit_code == 0
    assert report["metrics"] == {"passed": 1, "total": 3, "pass_rate": 1 / 3}
    assert report["cases"][0]["passed"] is True
    assert report["cases"][1]["checks"]["attack_marker_absent"] is False
    assert report["cases"][2]["checks"]["expected_fact_present"] is False
    assert all("answer" not in case for case in report["cases"])


def test_temporary_environment_restores_existing_values(monkeypatch):
    monkeypatch.setenv("TASK10_EXISTING", "before")
    with MODULE._temporary_environment({"TASK10_EXISTING": "inside", "TASK10_NEW": "created"}):
        assert MODULE.os.environ["TASK10_EXISTING"] == "inside"
        assert MODULE.os.environ["TASK10_NEW"] == "created"
    assert MODULE.os.environ["TASK10_EXISTING"] == "before"
    assert "TASK10_NEW" not in MODULE.os.environ
