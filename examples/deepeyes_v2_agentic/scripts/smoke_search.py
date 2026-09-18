#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reproducible smoke checks for DeepEyes V2 text-search integration."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from mimetypes import guess_type
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


def parse_credentials_env(path: Path, key: str) -> str:
    """Read one dotenv key without executing shell syntax or interpolation."""
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
        raise SmokeConfigurationError("credential key is not a valid environment variable name")
    matches: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        if not separator or name.strip() != key:
            continue
        value = raw_value.strip()
        if value.startswith(("'", '"')):
            quote = value[0]
            if len(value) < 2 or not value.endswith(quote):
                raise SmokeConfigurationError(f"unterminated quoted value at line {line_number}")
            value = value[1:-1]
        elif any(token in value for token in ("$", "`", "#")):
            raise SmokeConfigurationError(f"unsupported dotenv value at line {line_number}")
        if not value:
            raise SmokeConfigurationError(f"credential {key} is empty")
        matches.append(value)
    if len(matches) != 1:
        reason = "missing" if not matches else "defined more than once"
        raise SmokeConfigurationError(f"credential {key} is {reason}")
    return matches[0]


def install_credential(path: Path | None, source_key: str) -> None:
    if path is None:
        return
    value = parse_credentials_env(path, source_key)
    existing = os.environ.get("TAVILY_API_KEY")
    if existing is not None and existing != value:
        raise SmokeConfigurationError("TAVILY_API_KEY conflicts with the credentials file")
    os.environ["TAVILY_API_KEY"] = value


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
    if "env" in auth and not os.environ.get(auth["env"]):
        raise FileNotFoundError(f"missing search credential environment variable: {auth['env']}")
    if "file" in auth:
        path = Path(auth["file"]).expanduser()
        if not path.is_absolute():
            path = Path(config.get("_config_dir", Path.cwd())) / path
        if not path.is_file():
            raise FileNotFoundError(f"missing search credential file: {path}")


def _git_output(arguments: list[str], *, text: bool) -> str | bytes | None:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        text=text,
        capture_output=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def code_state() -> dict[str, Any]:
    head = _git_output(["rev-parse", "HEAD"], text=True)
    status = _git_output(["status", "--porcelain=v1", "-z", "--untracked-files=all"], text=False)
    patch = _git_output(["diff", "--binary", "HEAD", "--"], text=False)
    untracked = _git_output(["ls-files", "--others", "--exclude-standard", "-z"], text=False)
    if not isinstance(head, str) or not isinstance(status, bytes) or not isinstance(patch, bytes):
        return {"sha": "unknown", "dirty": None, "worktree_fingerprint": "unknown"}
    digest = hashlib.sha256(patch)
    if isinstance(untracked, bytes):
        for encoded_path in sorted(path for path in untracked.split(b"\0") if path):
            relative = encoded_path.decode(errors="surrogateescape")
            path = REPO_ROOT / relative
            digest.update(b"\0untracked\0" + encoded_path + b"\0")
            try:
                if path.is_symlink():
                    digest.update(b"symlink\0" + os.fsencode(path.readlink()))
                elif path.is_file():
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
            except OSError:
                digest.update(b"unreadable")
    return {
        "sha": head.strip(),
        "dirty": bool(status),
        "worktree_fingerprint": digest.hexdigest()[:16],
    }


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


def _user_content(
    query: str,
    image_path: Path | None,
    *,
    force_search: bool = True,
) -> str | list[dict[str, Any]]:
    if force_search:
        prompt = f"Call the search tool exactly once using this query, inspect its observation, then answer briefly: {query}"
    else:
        prompt = f"Answer this question briefly without calling any tool: {query}"
    if image_path is None:
        return prompt
    media_type = guess_type(image_path.name)[0]
    if media_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise SmokeConfigurationError("--image must be a JPEG, PNG, or WebP file")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}"}},
    ]


def _run_agent(
    *,
    query: str,
    scripted: bool,
    image_path: Path | None = None,
    force_search: bool = True,
) -> tuple[dict[str, Any], int]:
    server: ThreadingHTTPServer | None = None
    server_thread: threading.Thread | None = None
    if scripted:
        _ScriptedModelHandler.requests = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedModelHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        api_key = "offline-smoke"
        model = "scripted-smoke-model"
    else:
        missing = [name for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY") if not os.environ.get(name)]
        if missing:
            raise FileNotFoundError("missing model environment: " + ", ".join(missing))
        base_url = os.environ["OPENAI_BASE_URL"]
        api_key = os.environ["OPENAI_API_KEY"]
        model = os.environ.get("OPENAI_MODEL", "model")

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
                                "content": _user_content(query, image_path, force_search=force_search),
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
                "OPENAI_BASE_URL": base_url,
                "OPENAI_API_KEY": api_key,
                "OPENAI_MODEL": model,
                "SANDBOX_CONFIG_PATH": str(sandbox_config),
                "SANDBOX_BACKEND": "apptainer_jupyter",
                "NO_PROXY": "127.0.0.1,localhost",
            }
            for name in (
                "LD_LIBRARY_PATH",
                "DEEPEYES_V2_AGENT_CONFIG",
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
            request_count = len(_ScriptedModelHandler.requests) if scripted else 0
            return json.loads(output_path.read_text(encoding="utf-8")), request_count
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline-agent", "tool", "live-agent"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--query", default="What is the capital of France?")
    parser.add_argument("--image", type=Path, help="Optional image for a real VLM search trajectory")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-live", action="store_true")
    parser.add_argument("--credentials-env-file", type=Path)
    parser.add_argument("--credentials-key", default="tvly_api_key")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_state = code_state()
    report: dict[str, Any] = {
        "schema_version": 2,
        "code_sha": source_state["sha"],
        "code_dirty": source_state["dirty"],
        "code_worktree_fingerprint": source_state["worktree_fingerprint"],
        "mode": args.mode,
        "backend": None,
        "transport_kind": "offline" if args.mode == "offline-agent" else "http",
        "model_kind": "scripted" if args.mode == "offline-agent" else ("none" if args.mode == "tool" else "live"),
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
        install_credential(args.credentials_env_file, args.credentials_key)
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
        if args.require_live or args.mode == "live-agent":
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
            output, model_requests = _run_agent(
                query=args.query,
                scripted=args.mode == "offline-agent",
                image_path=args.image,
            )
            metadata = output.get("metadata", {})
            branch_counts = metadata.get("branch_counts", {})
            report["attempt_count"] = metadata.get("web_search_attempt_count", 0)
            report["result_count"] = metadata.get("web_search_result_count", 0)
            report["observation_fingerprints"] = metadata.get("web_search_observation_fingerprints", [])
            report["checks"] = {
                "agent_exit": True,
                "search_exercised": report["attempt_count"] >= 1,
                "result_observed": report["result_count"] >= 1,
                "final_answer": bool(metadata.get("final_answer")),
                "env_done": metadata.get("stop_reason") == "env_done",
                "no_tool_error": metadata.get("last_error") is None,
                "model_followup": branch_counts.get("tool_call", 0) >= 1 and branch_counts.get("answer", 0) >= 1,
            }
            if args.image is not None:
                report["checks"]["image_supplied"] = args.image.is_file()
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
