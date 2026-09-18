#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reproducible smoke checks for DeepEyes V2 text-search integration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


EXAMPLE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXAMPLE_DIR.parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from app.prompt import UNIFIED_SYSTEM_PROMPT  # noqa: E402
from app.search_backends import SearchConfigurationError, resolve_search_config  # noqa: E402
from app.search_utils import search  # noqa: E402


class SmokeConfigurationError(ValueError):
    pass


def config_fingerprint(config: dict[str, Any]) -> str:
    safe_config = {key: value for key, value in config.items() if not key.startswith("_")}
    encoded = json.dumps(safe_config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def valid_result_schema(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    elapsed = result.get("elapsed_time")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
        return False
    rows = result.get("data")
    if not isinstance(rows, list):
        return False
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"title", "link", "snippet", "date"}:
            return False
        if not all(isinstance(row[key], str) for key in ("title", "link", "snippet")):
            return False
        if row["date"] is not None and not isinstance(row["date"], str):
            return False
    return True


def result_fingerprint(result: dict[str, Any]) -> str:
    encoded = json.dumps(result["data"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def check_search_prerequisites(config: dict[str, Any]) -> None:
    if config["backend"] != "external":
        return
    auth = config["external"].get("auth")
    if not auth:
        return
    if not os.environ.get(auth["env"]):
        raise FileNotFoundError(f"missing search credential environment variable: {auth['env']}")


class _ScriptedModelHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            type(self).requests.append(request)
            messages = request.get("messages", [])
            if len(type(self).requests) == 1:
                content = (
                    '<think>I need current evidence.</think><tool_call>{"name":"search",'
                    '"arguments":{"query":"offline smoke marker"}}</tool_call>'
                )
            else:
                tool_messages = [message for message in messages if message.get("role") == "tool"]
                observed = any("## Web Results" in str(message.get("content")) for message in tool_messages)
                answer = "offline smoke passed" if observed else "search observation missing"
                content = f"<think>I inspected the result.</think><answer>{answer}</answer>"
            body = {
                "id": f"smoke-{len(type(self).requests)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "scripted-smoke-model",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            encoded = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        except Exception as exc:  # noqa: BLE001
            self.send_error(500, type(exc).__name__)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def _run_agent(*, query: str) -> tuple[dict[str, Any], int]:
    _ScriptedModelHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedModelHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        with tempfile.TemporaryDirectory(prefix="task10-smoke-") as temp_dir:
            temp = Path(temp_dir)
            sandbox_config = temp / "sandbox.yaml"
            sandbox_config.write_text(f"image: {temp / 'unused.sif'}\nnv: false\n", encoding="utf-8")
            input_path = temp / "input.json"
            output_path = temp / "output.json"
            input_path.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": UNIFIED_SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": (
                                    "Call the search tool exactly once using this query, inspect its observation, "
                                    f"then answer briefly: {query}"
                                ),
                            },
                        ],
                        "metadata": {"data_index": "task10-smoke", "data_source": "search"},
                    }
                ),
                encoding="utf-8",
            )
            child_env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": str(REPO_ROOT),
                "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                "OPENAI_API_KEY": "offline-smoke",
                "OPENAI_MODEL": "scripted-smoke-model",
                "SANDBOX_CONFIG_PATH": str(sandbox_config),
                "SANDBOX_BACKEND": "apptainer_jupyter",
                "NO_PROXY": "127.0.0.1,localhost",
            }
            for name in (
                "LD_LIBRARY_PATH",
                "DEEPEYES_V2_WEB_SEARCH_CONFIG",
                "DEEPEYES_V2_WEB_SEARCH_BACKEND",
                "DEEPEYES_V2_WEB_SEARCH_URL",
                "DEEPEYES_V2_WEB_SEARCH_TOP_K",
                "DEEPEYES_V2_WEB_SEARCH_TIMEOUT_S",
                "DEEPEYES_V2_WEB_SEARCH_MAX_RETRIES",
                "TAVILY_API_KEY",
            ):
                if name in os.environ:
                    child_env[name] = os.environ[name]
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "app.agent",
                    "--input-json",
                    str(input_path),
                    "--output-json",
                    str(output_path),
                ],
                cwd=EXAMPLE_DIR,
                env=child_env,
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"agent subprocess failed with exit code {completed.returncode}")
            if not output_path.exists():
                raise RuntimeError("agent subprocess did not write output")
            return json.loads(output_path.read_text(encoding="utf-8")), len(_ScriptedModelHandler.requests)
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline-agent", "tool"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--query", default="What is the capital of France?")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-live", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "schema_version": 2,
        "mode": args.mode,
        "backend": None,
        "transport_kind": "offline" if args.mode == "offline-agent" else "http",
        "model_kind": "scripted" if args.mode == "offline-agent" else "none",
        "config_fingerprint": None,
        "status": "failed",
        "checks": {},
        "attempt_count": 0,
        "result_count": 0,
        "result_fingerprint": None,
        "elapsed_time": 0.0,
        "error_category": None,
        "prerequisites_missing": [],
    }
    exit_code = 1
    started = time.monotonic()
    try:
        if args.config is not None:
            os.environ["DEEPEYES_V2_WEB_SEARCH_CONFIG"] = str(args.config.resolve())
        config = resolve_search_config()
        report["backend"] = config["backend"]
        report["transport_kind"] = "offline" if config["backend"] == "mock" else "http"
        report["config_fingerprint"] = config_fingerprint(config)
        if args.require_live and config["backend"] == "mock":
            raise SmokeConfigurationError("--require-live cannot be used with the mock backend")
        if args.mode == "offline-agent" and config["backend"] != "mock":
            endpoint = (
                config["retriever"]["url"] if config["backend"] == "retriever" else config["external"]["endpoint"]
            )
            if urlsplit(endpoint).hostname not in {"127.0.0.1", "localhost", "::1"}:
                raise SmokeConfigurationError("offline-agent permits only mock or loopback HTTP search")
        if args.require_live:
            check_search_prerequisites(config)

        if args.mode == "tool":
            result = search(args.query, config=config)
            report["attempt_count"] = 1
            if result == "Error":
                raise RuntimeError("search backend returned Error")
            report["result_count"] = len(result["data"])
            report["result_fingerprint"] = result_fingerprint(result)
            serialized = json.dumps(result["data"], ensure_ascii=False).lower()
            report["checks"] = {
                "search_succeeded": True,
                "schema_valid": valid_result_schema(result),
                "non_placeholder_results": config["backend"] == "mock"
                or ("placeholder" not in serialized and "offline mock result" not in serialized),
            }
            if not all(report["checks"].values()):
                raise RuntimeError("tool smoke checks failed")
        else:
            output, model_requests = _run_agent(query=args.query)
            metadata = output.get("metadata", {})
            branch_counts = metadata.get("branch_counts", {})
            report["attempt_count"] = metadata.get("web_search_attempt_count", 0)
            report["result_count"] = metadata.get("web_search_result_count", 0)
            report["checks"] = {
                "agent_exit": True,
                "search_exercised": report["attempt_count"] >= 1,
                "result_observed": report["result_count"] >= 1,
                "final_answer": bool(metadata.get("final_answer")),
                "env_done": metadata.get("stop_reason") == "env_done",
                "no_tool_error": metadata.get("last_error") is None,
                "model_followup": branch_counts.get("tool_call", 0) >= 1 and branch_counts.get("answer", 0) >= 1,
            }
            if not all(report["checks"].values()):
                raise RuntimeError("agent smoke checks failed")
        report["status"] = "passed"
        exit_code = 0
    except FileNotFoundError as exc:
        report["status"] = "skipped"
        report["error_category"] = type(exc).__name__
        report["prerequisites_missing"] = [str(exc)]
        exit_code = 3
    except (SmokeConfigurationError, SearchConfigurationError, OSError, ValueError) as exc:
        report["status"] = "failed"
        report["error_category"] = type(exc).__name__
        exit_code = 2
    except Exception as exc:  # noqa: BLE001
        report["status"] = "failed"
        report["error_category"] = type(exc).__name__
        exit_code = 1
    finally:
        report["elapsed_time"] = round(time.monotonic() - started, 6)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
