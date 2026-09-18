# DeepEyes V2 — agentic example

DeepEyes V2 ([paper](https://arxiv.org/abs/2511.05271),
[upstream](https://github.com/Visual-Agent/DeepEyesV2)) on Relax's agentic
stack. Multimodal RL with three action channels:

- `<code>...</code>` — Python in a Jupyter sandbox (`image_1` pre-loaded)
- `<tool_call>{"name": "search" | "image_search", ...}</tool_call>` — text/image search
- `<answer>...</answer>` — terminate

Reward = `0.8 * acc + 0.2 * format`, accuracy via LLM-judge, routed on
`extra_info.data_source` ∈ {perception, reason, search, vstar-test}.

## One env var: `DATA_DIR`

`scripts/prepare.sh` keeps the high-concurrency app environment on node-local
storage and lays out shared data assets under `DATA_DIR`:

```
${DEEPEYES_V2_APP_ENV_ROOT:-/tmp/deepeyes-v2-app-env}/
└── .venv/                           (host-side agent environment)

${DATA_DIR}/
├── sif/
│   └── deepeyes_v2_kernel.sif       (~115 MiB, built locally)
└── data/
    ├── raw/                          (~10 GiB, raw HF download — kept for re-conversion)
    │   └── {perception_all_1..5,reason,search,vstar_test}.parquet
    ├── {perception_all_1..5,reason,search,vstar_test}.parquet   (data_source-injected)
    └── smoke.parquet                 (4 synthetic rows, all 4 data_sources)
```

## Configure once: `env.sh`

Machine-local paths / endpoints / proxies / mirrors live in `env.sh`
(gitignored). Every script in this example auto-sources it.

```bash
cp examples/deepeyes_v2_agentic/env.sh.example examples/deepeyes_v2_agentic/env.sh
# edit env.sh — at minimum set DATA_DIR; see comments in the file for the
# optional knobs (OPENAI_BASE_URL, HF_HTTP_PROXY, BOOTSTRAP_FROM_IMAGE, …)
```

## Prep (one-time)

```bash
bash examples/deepeyes_v2_agentic/scripts/prepare.sh
```

What it does (all idempotent — skips done work on re-run):

1. **App environment** — creates a reusable node-local `uv` virtual environment
   with system site packages and installs the host-side `jupyter_client` used
   to control the sandbox kernel. Its default location is
   `/tmp/deepeyes-v2-app-env`; prepare the same path on every training node.
2. **SIF** — `apptainer build` from `apptainer_env/deepeyes_v2_kernel.def`, falls
   back to `--fakeroot`, verifies kernel deps inside the SIF.
3. **Train parquets** — downloads `honglyhly/DeepEyesV2_RL` (8 files, ~10 GiB)
   from the configured Hugging Face endpoint (default `https://huggingface.co`),
   then runs `convert_tool/rl_data_convert.py` to inject
   `extra_info.data_source`.
4. **Smoke parquet** — 4 synthetic rows covering all `data_source` values.

Skip individual steps with `SKIP_SIF=1 SKIP_TRAIN=1 SKIP_SMOKE=1`.

## Smoke (single sample, no Ray)

One trajectory through the full agent app (`app/agent.py` + sandbox +
tools) against an OpenAI-compatible chat endpoint. Run this FIRST to
verify the agent loop / sandbox / message wiring before the cluster
launch. Needs `OPENAI_BASE_URL` + `OPENAI_API_KEY` in `env.sh`:

```bash
bash examples/deepeyes_v2_agentic/scripts/smoke.sh
# or a different row from smoke.parquet:
SMOKE_ROW=2 bash examples/deepeyes_v2_agentic/scripts/smoke.sh
```

Pretty-prints the output JSON + a summary. Healthy run:
`stop_reason=env_done`, `branch_counts.code≥1`, `final_answer` non-null,
`last_error=null`. After: `apptainer instance list` must be empty (else
`env.close()` didn't run on some exit path — file a bug).

For ad-hoc debug with a synthetic input (no parquet needed):

```bash
source examples/deepeyes_v2_agentic/env.sh
${DEEPEYES_V2_APP_ENV_ROOT:-/tmp/deepeyes-v2-app-env}/.venv/bin/python \
    examples/deepeyes_v2_agentic/scripts/run_single_session.py
```

## Text web search backends

Text search is pluggable and uses one stable result contract:
`{"elapsed_time": float, "data": [{"title", "link", "snippet", "date"}]}`.
The committed default is deterministic `mock`, so tests and first-run smoke
checks do not need network access or credentials. Complete configurations are
provided in `configs/search/`:

- `mock.yaml` — deterministic offline results.
- `retriever.yaml` — Search-R1-compatible `POST /retrieve`; replace the
  loopback URL when the service is on another node.
- `external-tavily.yaml` — generic mapped external API example. It reads
  `TAVILY_API_KEY` at request time and contains no secret.

Select a configuration without editing the committed agent YAML:

```bash
export DEEPEYES_V2_WEB_SEARCH_CONFIG="$PWD/examples/deepeyes_v2_agentic/configs/search/retriever.yaml"
```

`DEEPEYES_V2_WEB_SEARCH_BACKEND`, `DEEPEYES_V2_WEB_SEARCH_URL`,
`DEEPEYES_V2_WEB_SEARCH_TOP_K`, `DEEPEYES_V2_WEB_SEARCH_TIMEOUT_S`, and
`DEEPEYES_V2_WEB_SEARCH_MAX_RETRIES` are the only scalar environment
overrides. Explicitly empty overrides fail configuration validation. External
production endpoints require HTTPS; loopback HTTP is accepted for fixtures.
Search-R1 URLs may use HTTP on a trusted internal network.

Backend failures, timeouts, invalid JSON, unsafe redirects, and invalid
mappings become a non-terminal `search_failed` tool
observation so the agent can continue reasoning. Search result text is escaped
before entering the model context and the complete observation is bounded by
`max_observation_chars`. Search-R1 documents without a source URL render as a
plain title rather than a fabricated link.

Each agent owns one reusable HTTP client so repeated tool calls share a
connection pool without adding global state.

Run the dedicated smoke entry point:

```bash
# Actual agent subprocess + scripted local model + default mock search
python examples/deepeyes_v2_agentic/scripts/smoke_search.py \
  --mode offline-agent --report /tmp/task10-offline.json

# Direct live backend check
python examples/deepeyes_v2_agentic/scripts/smoke_search.py \
  --mode tool \
  --config examples/deepeyes_v2_agentic/configs/search/retriever.yaml \
  --query 'test query' --require-live \
  --report /tmp/task10-retriever.json
```

For the external backend, provide the configured credential through the
environment and run the same direct backend check:

```bash
export TAVILY_API_KEY=...
python examples/deepeyes_v2_agentic/scripts/smoke_search.py \
  --mode tool \
  --config examples/deepeyes_v2_agentic/configs/search/external-tavily.yaml \
  --query 'test query' --require-live --report /tmp/task10-tavily.json
```

The JSON report is credential-free. Reproducible Task 10 acceptance results
are summarized in
[`docs/validation-results.md`](./docs/validation-results.md).

## Train (cluster)

`DATA_DIR` comes from `env.sh`. Also export `MODEL_DIR` + `SAVE_DIR`:

```bash
export MODEL_DIR=...          # contains Qwen3.6-35B-A3B/ and Qwen2.5-1.5B-Instruct/
export SAVE_DIR=...

bash examples/deepeyes_v2_agentic/run_deepeyes_v2_agentic.sh
```

The launcher auto-resolves `APPTAINER_IMAGE_PATH = ${DATA_DIR}/sif/deepeyes_v2_kernel.sif`
and propagates it to every Ray worker via `--runtime-env-json`. Set
`APPTAINER_IMAGE_PATH` explicitly to override (e.g. shared NFS path for
multi-node).

## Image-search cache (optional, only for the `search` split)

The `<tool_call>image_search</tool_call>` branch hits a precomputed
MMSearch-R1 cache, not live Google. If you skip this, the search backend
returns a benign `Error`, the env surfaces it as `search_failed`, and training
keeps moving — fine for any split that isn't `search`.

To enable it, get the MMSearch raw cache (separate dataset, not bundled) and:

```bash
python examples/deepeyes_v2_agentic/convert_tool/cache_convert.py \
    --input_json_path  ${MMSEARCH_CACHE_JSON} \
    --output_json_path ${DEEPEYES_V2_SEARCH_CACHE_PATHS} \
    --data_path        ${MMSEARCH_IMAGE_ROOT}
```

then `export DEEPEYES_V2_SEARCH_CACHE_PATHS=...` before launching.

## Layout

| Path                             | Role                                                                                     |
| -------------------------------- | ---------------------------------------------------------------------------------------- |
| `env.sh.example`                 | Template for the gitignored `env.sh` — set DATA_DIR + optional knobs                     |
| `scripts/prepare.sh`             | Single prep entry point — app environment + SIF + train data + smoke parquet             |
| `scripts/prepare_app_env.sh`     | Reusable host-side agent environment                                                     |
| `scripts/build_smoke_parquet.py` | Synthetic 4-row parquet generator (called by prepare.sh; also runnable standalone)       |
| `scripts/run_single_session.py`  | Single-trajectory harness (parquet row or synthetic input)                               |
| `app/agent.py`                   | Per-session agent driver                                                                 |
| `app/env_deepeyes_v2.py`         | Tool handlers (exec_code / exec_tool / close)                                            |
| `app/prompt.py`                  | Observation templates + sandbox init code                                                |
| `app/search_utils.py`            | Text + image-search backend                                                              |
| `app/search_backends.py`         | Search configuration, HTTP policy, and mock/retriever/external adapters                  |
| `configs/search/`                | Copyable mock, Search-R1, and Tavily backend configurations                              |
| `app/sandboxes/`                 | Jupyter sandbox abstraction + apptainer backend                                          |
| `reward_deepeyes_v2.py`          | Post-trajectory scorer (data_source-routed, LLM-judge)                                   |
| `convert_tool/`                  | `rl_data_convert.py` (data_source injection) + `cache_convert.py` (search cache rewrite) |
| `apptainer_env/`                 | Apptainer image def + sandbox YAML config                                                |
| `run_deepeyes_v2_agentic.sh`     | Full GRPO launch (Qwen3.6-35B-A3B, colocate)                                             |
| `scripts/smoke.sh`               | Single-sample smoke wrapper (one trajectory through the full app, no Ray)                |
| `scripts/smoke_search.py`        | Search-specific offline-agent/tool smoke and redacted JSON report                        |
| `run_agent_app.sh`               | Per-session wrapper invoked by Relax for each rollout                                    |
| `sglang_judge_service.sh`        | Stands up the LLM-judge SGLang server                                                    |

## Pitfalls — read before debugging

Adapting DeepEyes V2 on Relax's agentic stack has a set of recurring
footguns (SGLang mamba IMA, "step 1 keeps looping" caused by
`--use-fault-tolerance` silently masking errors,
reward tool-bonus divergence from upstream, …). Before opening py-spy,
read [`PITFALLS.md`](./PITFALLS.md) — the top entry (don't enable
`--use-fault-tolerance` during adaptation) alone will save hours.

## How it differs from `examples/deepeyes_agentic/` (V1)

Three action branches vs V1's single `image_zoom_in_tool`; stateful Jupyter
sandbox per trajectory; reward routed on data_source. Same agentic stack +
OpenAI-SDK driver pattern.

## Phase 1 scope notes

- Only the **apptainer** backend ships in Phase 1; `nexsandbox_backend.py` is
  intentionally absent (Phase 1.5 plan in `docs/superpowers/plans/`).
- Sandbox abstraction is example-local at `app/sandboxes/`; will move to
  `relax/runtime/sandbox/` in Phase 1.5 once smoke validates the design.
- Cold-start SFT out of scope (Phase 2). First runs will see low reward
  without a community V2 SFT checkpoint.
