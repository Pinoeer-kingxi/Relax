# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = REPO_ROOT / "examples" / "deepeyes_v2_agentic"
SMOKE_SCRIPT = REPO_ROOT / "examples" / "deepeyes_v2_agentic" / "scripts" / "smoke_search.py"
AGENT_WRAPPER = EXAMPLE_DIR / "run_agent_app.sh"
sys.path.insert(0, str(EXAMPLE_DIR))

_SMOKE_SPEC = importlib.util.spec_from_file_location("task10_smoke_search", SMOKE_SCRIPT)
assert _SMOKE_SPEC is not None and _SMOKE_SPEC.loader is not None
_SMOKE_MODULE = importlib.util.module_from_spec(_SMOKE_SPEC)
_SMOKE_SPEC.loader.exec_module(_SMOKE_MODULE)
SmokeConfigurationError = _SMOKE_MODULE.SmokeConfigurationError
parse_credentials_env = _SMOKE_MODULE.parse_credentials_env
build_user_content = _SMOKE_MODULE._user_content
valid_result_schema = _SMOKE_MODULE.valid_result_schema
result_fingerprint = _SMOKE_MODULE.result_fingerprint
code_state = _SMOKE_MODULE.code_state
ScriptedModelHandler = _SMOKE_MODULE._ScriptedModelHandler
UNIFIED_SYSTEM_PROMPT = _SMOKE_MODULE.UNIFIED_SYSTEM_PROMPT


class _RetrieverHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        type(self).requests.append(request)
        body = json.dumps(
            {
                "result": [
                    [
                        {
                            "document": {
                                "title": "Unique local fixture marker",
                                "text": "The scripted agent must receive this observation.",
                                "url": "https://fixture.invalid/result",
                            }
                        }
                    ]
                ]
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


class _FailingRetrieverHandler(BaseHTTPRequestHandler):
    request_count = 0

    def do_POST(self) -> None:  # noqa: N802
        type(self).request_count += 1
        self.send_response(503)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def test_credentials_parser_accepts_explicit_single_line_forms(tmp_path):
    credentials = tmp_path / ".env"
    credentials.write_text("UNRELATED=$(ignored)\nexport tvly_api_key='test-value'\n", encoding="utf-8")
    assert parse_credentials_env(credentials, "tvly_api_key") == "test-value"


def test_credentials_parser_rejects_duplicate_or_executable_values(tmp_path):
    credentials = tmp_path / ".env"
    credentials.write_text("tvly_api_key=one\ntvly_api_key=two\n", encoding="utf-8")
    try:
        parse_credentials_env(credentials, "tvly_api_key")
    except SmokeConfigurationError as exc:
        assert "more than once" in str(exc)
    else:
        raise AssertionError("duplicate credentials were accepted")

    credentials.write_text("tvly_api_key=$(unsafe)\n", encoding="utf-8")
    try:
        parse_credentials_env(credentials, "tvly_api_key")
    except SmokeConfigurationError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("executable dotenv syntax was accepted")


def test_smoke_user_content_can_include_an_image(tmp_path):
    image = tmp_path / "sample.png"
    image.write_bytes(b"small-fixture")
    content = build_user_content("search with image", image)
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_smoke_result_schema_is_strict():
    valid = {
        "elapsed_time": 0.1,
        "data": [{"title": "T", "link": "", "snippet": "S", "date": None}],
    }
    assert valid_result_schema(valid)
    assert not valid_result_schema({"elapsed_time": float("nan"), "data": []})
    assert not valid_result_schema({"elapsed_time": 0.1, "data": [{"title": "T"}]})


def test_result_fingerprint_ignores_timing_and_code_state_is_well_formed():
    first = {"elapsed_time": 0.1, "data": [{"title": "T", "link": "", "snippet": "S", "date": None}]}
    second = {"elapsed_time": 9.9, "data": first["data"]}

    assert result_fingerprint(first) == result_fingerprint(second)
    state = code_state()
    assert len(state["sha"]) == 40
    assert isinstance(state["dirty"], bool)
    assert len(state["worktree_fingerprint"]) == 16


def test_local_retriever_to_real_agent_subprocess(tmp_path):
    _RetrieverHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RetrieverHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = tmp_path / "search.yaml"
        config.write_text(
            "\n".join(
                [
                    "backend: retriever",
                    "top_k: 1",
                    "timeout_s: 2",
                    "max_retries: 0",
                    "retry_budget_s: 3",
                    "backoff_initial_s: 0",
                    "backoff_max_s: 0",
                    "trust_env: false",
                    "max_response_bytes: 4096",
                    "max_observation_chars: 4096",
                    "retriever:",
                    f"  url: http://127.0.0.1:{server.server_port}/retrieve",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        report = tmp_path / "report.json"
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(REPO_ROOT),
            "NO_PROXY": "127.0.0.1,localhost",
        }
        completed = subprocess.run(
            [
                sys.executable,
                str(SMOKE_SCRIPT),
                "--mode",
                "offline-agent",
                "--config",
                str(config),
                "--query",
                "local e2e query",
                "--report",
                str(report),
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert payload["status"] == "passed"
        assert payload["backend"] == "retriever"
        assert payload["attempt_count"] == 1
        assert payload["result_count"] == 1
        assert _RetrieverHandler.requests == [{"queries": ["offline smoke marker"], "topk": 1, "return_scores": True}]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _run_agent_wrapper(
    tmp_path: Path, *, retriever_port: int, max_retries: int
) -> tuple[subprocess.CompletedProcess, dict]:
    ScriptedModelHandler.requests = []
    model = ThreadingHTTPServer(("127.0.0.1", 0), ScriptedModelHandler)
    model_thread = threading.Thread(target=model.serve_forever, daemon=True)
    model_thread.start()
    try:
        search_config = tmp_path / "search config.yaml"
        search_config.write_text(
            "\n".join(
                [
                    "backend: retriever",
                    "top_k: 1",
                    "timeout_s: 2",
                    f"max_retries: {max_retries}",
                    "retry_budget_s: 3",
                    "backoff_initial_s: 0",
                    "backoff_max_s: 0",
                    "trust_env: false",
                    "max_response_bytes: 4096",
                    "max_observation_chars: 4096",
                    "retriever:",
                    f"  url: http://127.0.0.1:{retriever_port}/retrieve",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        sandbox_config = tmp_path / "sandbox config.yaml"
        sandbox_config.write_text(f"image: {tmp_path / 'unused.sif'}\nnv: false\n", encoding="utf-8")
        input_path = tmp_path / "input.json"
        output_path = tmp_path / "output.json"
        input_path.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": UNIFIED_SYSTEM_PROMPT},
                        {"role": "user", "content": "Use search, then answer."},
                    ],
                    "metadata": {"data_index": "wrapper-e2e", "data_source": "search"},
                }
            ),
            encoding="utf-8",
        )
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(REPO_ROOT),
            "NO_PROXY": "127.0.0.1,localhost",
            "DEEPEYES_V2_APP_PYTHON": sys.executable,
            "DEEPEYES_V2_WEB_SEARCH_CONFIG": str(search_config),
            "SANDBOX_BACKEND": "apptainer_jupyter",
            "SANDBOX_CONFIG_PATH": str(sandbox_config),
            "RELAX_BASE_URL": f"http://127.0.0.1:{model.server_port}/v1/",
            "RELAX_SESSION_ID": "wrapper-e2e-session",
            "RELAX_INPUT_JSON": str(input_path),
            "RELAX_OUTPUT_JSON": str(output_path),
        }
        if "LD_LIBRARY_PATH" in os.environ:
            env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]

        completed = subprocess.run(
            ["bash", str(AGENT_WRAPPER)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

        output = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {}
        return completed, output
    finally:
        model.shutdown()
        model.server_close()
        model_thread.join(timeout=5)


def test_run_agent_wrapper_propagates_live_retriever_config(tmp_path):
    _RetrieverHandler.requests = []
    retriever = ThreadingHTTPServer(("127.0.0.1", 0), _RetrieverHandler)
    retriever_thread = threading.Thread(target=retriever.serve_forever, daemon=True)
    retriever_thread.start()
    try:
        completed, output = _run_agent_wrapper(tmp_path, retriever_port=retriever.server_port, max_retries=0)

        assert completed.returncode == 0, completed.stderr
        assert output["metadata"]["stop_reason"] == "env_done"
        assert output["metadata"]["web_search_attempt_count"] == 1
        assert output["metadata"]["web_search_result_count"] == 1
        assert output["metadata"]["web_search_backend"] == "retriever"
        assert output["metadata"]["web_search_elapsed_time_s"] >= 0
        assert len(output["metadata"]["web_search_observation_fingerprints"]) == 1
        assert output["metadata"]["web_search_runtime_metrics"] == {
            "cache_hit_count": 0,
            "circuit_open_count": 0,
            "failure_count": 0,
            "http_attempt_count": 1,
            "retry_count": 0,
            "search_count": 1,
        }
        assert len(ScriptedModelHandler.requests) == 2
        assert all(request["max_tokens"] == 4096 for request in ScriptedModelHandler.requests)
        assert _RetrieverHandler.requests == [{"queries": ["offline smoke marker"], "topk": 1, "return_scores": True}]
    finally:
        retriever.shutdown()
        retriever.server_close()
        retriever_thread.join(timeout=5)


def test_retriever_failure_retries_and_agent_still_answers(tmp_path):
    _FailingRetrieverHandler.request_count = 0
    retriever = ThreadingHTTPServer(("127.0.0.1", 0), _FailingRetrieverHandler)
    retriever_thread = threading.Thread(target=retriever.serve_forever, daemon=True)
    retriever_thread.start()
    try:
        completed, output = _run_agent_wrapper(tmp_path, retriever_port=retriever.server_port, max_retries=1)

        assert completed.returncode == 0, completed.stderr
        assert _FailingRetrieverHandler.request_count == 2
        assert len(ScriptedModelHandler.requests) == 2
        assert output["metadata"]["stop_reason"] == "env_done"
        assert output["metadata"]["last_error"] == "search_failed"
        assert output["metadata"]["final_answer"] == "search observation missing"
        assert output["metadata"]["web_search_attempt_count"] == 1
        assert output["metadata"]["web_search_result_count"] == 0
        assert output["metadata"]["web_search_runtime_metrics"]["failure_count"] == 1
        assert output["metadata"]["web_search_runtime_metrics"]["http_attempt_count"] == 2
        assert output["metadata"]["web_search_runtime_metrics"]["retry_count"] == 1
    finally:
        retriever.shutdown()
        retriever.server_close()
        retriever_thread.join(timeout=5)
