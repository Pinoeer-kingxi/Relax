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

Backend failures, timeouts, invalid JSON, unsafe redirects, oversized
responses, and invalid mappings become a non-terminal `search_failed` tool
observation so the agent can continue reasoning. Search result text is escaped
before entering the model context and the complete observation is bounded by
`max_observation_chars`. Search-R1 documents without a source URL render as a
plain title rather than a fabricated link.

Each agent owns one reusable HTTP client. `cache_ttl_s` and
`cache_max_entries` configure its bounded per-agent LRU cache; setting the TTL
to zero disables caching. `circuit_breaker_failures` consecutive failures open
the circuit for `circuit_breaker_reset_s`, preventing a degraded upstream from
stalling every turn. Transport attempts, retries, cache hits, failures, and
open-circuit rejections are included in output metadata under
`web_search_runtime_metrics`.

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

For an explicitly selected local credentials file, the smoke script can map
one dotenv key without sourcing or evaluating the file:

```bash
python examples/deepeyes_v2_agentic/scripts/smoke_search.py \
  --mode live-agent \
  --config examples/deepeyes_v2_agentic/configs/search/external-tavily.yaml \
  --credentials-env-file /path/to/.env --credentials-key tvly_api_key \
  --query 'test query' --require-live --report /tmp/task10-live-agent.json
```

The JSON report is deliberately credential-free. `live-agent` also requires
`OPENAI_BASE_URL`, `OPENAI_API_KEY`, and optionally `OPENAI_MODEL`; it fails if
the model never exercises search. Pass `--image /path/to/image.png` to exercise
the VLM image-input path in the same search trajectory. GPU/model/retriever
resource boundaries and validation levels are documented in
[`docs/search-e2e-resources.md`](./docs/search-e2e-resources.md). Reproducible
Task 10 acceptance and six-GPU smoke results are summarized in
[`docs/validation-results.md`](./docs/validation-results.md).
When the endpoint is `transformers serve`, set `DEEPEYES_V2_AGENT_CONFIG` to
`configs/agent-transformers-validation.yaml`; that server rejects SGLang's
otherwise production-default `stop_token_ids` request extension.

Measure retrieval quality on JSONL, JSON, or Parquet QA data with answer-hit
rate and MRR (the report stores query hashes, not raw queries or answers):

```bash
python examples/deepeyes_v2_agentic/scripts/evaluate_search_quality.py \
  --input /path/to/eval.jsonl \
  --config examples/deepeyes_v2_agentic/configs/search/retriever.yaml \
  --top-k 5 --require-all-success --min-answer-hit-rate 0.5 \
  --report /tmp/task10-search-quality.json
```

Rows default to `prompt` plus `extra_info.answers`; override those dotted paths
with `--query-field` and `--answers-field` (the raw Search-R1 parquet uses
`--query-field question --answers-field golden_answers`). The report also pins
the input hash, resolved config fingerprint, Git commit, and dirty-worktree
fingerprint.

Use `--group-field extra_info.source --max-examples-per-group N` for a
deterministic stratified sample. `--baseline-report previous.json` rejects an
incompatible baseline and reports answer-hit/MRR/latency deltas for comparable
runs.

Evaluate actual final answers—not only retrieved snippets—against the same QA
formats by running a live local or remote model endpoint:

```bash
python examples/deepeyes_v2_agentic/scripts/evaluate_agent_search.py \
  --input /path/to/eval.jsonl --condition search \
  --config examples/deepeyes_v2_agentic/configs/search/retriever.yaml \
  --group-field extra_info.source --max-examples-per-group 5 \
  --baseline-report /tmp/task10-agent-no-search.json \
  --require-all-success --report /tmp/task10-agent-search.json
```

The prompt-injection check serves correct evidence containing three hostile
instruction variants from a loopback Search-R1 endpoint, then verifies that a
live agent uses the fact and excludes the attack marker:

```bash
python examples/deepeyes_v2_agentic/scripts/evaluate_prompt_injection.py \
  --min-pass-rate 1.0 --report /tmp/task10-prompt-injection.json
```

## Train (cluster)

`DATA_DIR` comes from `env.sh`. Also export `MODEL_DIR` + `SAVE_DIR`:

```bash
export MODEL_DIR=...          # contains Qwen3.6-35B-A3B/ and Qwen2.5-1.5B-Instruct/
export SAVE_DIR=...

bash examples/deepeyes_v2_agentic/run_deepeyes_v2_agentic.sh
```

The launcher auto-resolves `APPTAINER_IMAGE_PATH = ${DATA_DIR}/sif/deepeyes_v2_kernel.sif`
and propagates it to every Ray worker via a mode-`0600` Ray runtime-env file. Set
`APPTAINER_IMAGE_PATH` explicitly to override (e.g. shared NFS path for
multi-node).

Before allocating Ray workers, generate a machine-readable readiness report:

```bash
python examples/deepeyes_v2_agentic/scripts/preflight_training.py \
  --model-dir "$MODEL_DIR" --data-dir "$DATA_DIR" --save-dir "$SAVE_DIR" \
  --training-image "$RELAX_TRAINING_IMAGE_PATH" \
  --report /tmp/deepeyes-v2-training-preflight.json
```

Pass `--sif-path "$APPTAINER_IMAGE_PATH"` when training should use an exact
shared image path instead of `${DATA_DIR}/sif/deepeyes_v2_kernel.sif`.

The resource-complete validation profile requires eight GPUs that are actually
idle, the exact Qwen3.6-35B-A3B checkpoint with its weight index and all
referenced shards, the Qwen2.5-1.5B-Instruct judge model, all eight converted
DeepEyesV2_RL parquets, sandbox SIF, pinned Relax training SIF, app environment,
Ray, PyTorch, and Apptainer. The
model checks pin the expected Hugging Face revisions, Qwen3.6's 26 safetensors
shards with exact sizes and SHA-256 values, the judge weight size and SHA-256,
and key config fields. A checkout
may prove revision with `git rev-parse HEAD`; non-git downloads must include a
revision marker such as `.hf_revision`. The data check requires
`${DATA_DIR}/data/raw/.hf_revision`, valid converted parquet metadata, non-zero
rows, required columns, and the injected `extra_info.data_source` values. The
sandbox check runs bounded `apptainer inspect --json` and imports `ipykernel`,
`PIL`, `matplotlib`, `autopep8`, and `numpy` inside the SIF. The training image
must have an adjacent `.oci-digest` file containing the pinned official digest.
Its check runs `apptainer exec --nv` and imports the real Ray/Torch/SGLang/
Megatron/TransformerEngine/FlashAttention stack, plus
`cuda.bindings.driver`/`cuda.bindings.runtime`, while requiring CUDA. If a
host-mounted Python user site shadows the image's CUDA bindings, configure the
optional `RELAX_CUDA_PYTHON_OVERRIDE`/`APPTAINERENV_*` variables documented in
`env.sh.example`. Enable the documented NCCL transport fallbacks only when a
real multi-GPU collective fails with the defaults on that host. The
launcher currently selects two training files
plus `vstar_test.parquet`, but the preflight deliberately validates the
complete prepared bundle. A blocked preflight must not be relabeled as a
reduced recipe validation.

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

| Path                                   | Role                                                                                     |
| -------------------------------------- | ---------------------------------------------------------------------------------------- |
| `env.sh.example`                       | Template for the gitignored `env.sh` — set DATA_DIR + optional knobs                     |
| `scripts/prepare.sh`                   | Single prep entry point — app environment + SIF + train data + smoke parquet             |
| `scripts/prepare_app_env.sh`           | Reusable host-side agent environment                                                     |
| `scripts/build_smoke_parquet.py`       | Synthetic 4-row parquet generator (called by prepare.sh; also runnable standalone)       |
| `scripts/build_search_fixture.py`      | Reproducible small E5/FAISS native-retriever fixture builder                             |
| `scripts/evaluate_search_quality.py`   | Answer-hit/MRR/latency evaluator for JSONL, JSON, or Parquet QA data                     |
| `scripts/evaluate_agent_search.py`     | Live-agent final-answer evaluator with search/no-search conditions                       |
| `scripts/evaluate_prompt_injection.py` | Live-agent hostile-search-evidence resistance evaluator                                  |
| `scripts/run_single_session.py`        | Single-trajectory harness (parquet row or synthetic input)                               |
| `app/agent.py`                         | Per-session agent driver                                                                 |
| `app/env_deepeyes_v2.py`               | Tool handlers (exec_code / exec_tool / close)                                            |
| `app/prompt.py`                        | Observation templates + sandbox init code                                                |
| `app/search_utils.py`                  | Text + image-search backend                                                              |
| `app/search_backends.py`               | Search configuration, HTTP policy, and mock/retriever/external adapters                  |
| `configs/search/`                      | Copyable mock, Search-R1, and Tavily backend configurations                              |
| `app/sandboxes/`                       | Jupyter sandbox abstraction + apptainer backend                                          |
| `reward_deepeyes_v2.py`                | Post-trajectory scorer (data_source-routed, LLM-judge)                                   |
| `convert_tool/`                        | `rl_data_convert.py` (data_source injection) + `cache_convert.py` (search cache rewrite) |
| `apptainer_env/`                       | Apptainer image def + sandbox YAML config                                                |
| `run_deepeyes_v2_agentic.sh`           | Full GRPO launch (Qwen3.6-35B-A3B, colocate)                                             |
| `scripts/smoke.sh`                     | Single-sample smoke wrapper (one trajectory through the full app, no Ray)                |
| `scripts/smoke_search.py`              | Search-specific offline/tool/live-agent smoke and redacted JSON report                   |
| `run_agent_app.sh`                     | Per-session wrapper invoked by Relax for each rollout                                    |
| `sglang_judge_service.sh`              | Stands up the LLM-judge SGLang server                                                    |

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
