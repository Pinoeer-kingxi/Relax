#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

###############################################################################
#                                 ENVIRONMENT                                 #
###############################################################################

TIMESTAMP=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source env.sh if present (gitignored, machine-specific overrides).
# shellcheck source=/dev/null
[ -f "${SCRIPT_DIR}/env.sh" ] && source "${SCRIPT_DIR}/env.sh"

# Shell tracing is opt-in and starts only after machine-local environment files
# have been sourced. Secret-bearing commands below disable tracing explicitly.
if [ "${DEEPEYES_TRACE:-0}" = "1" ]; then
    set -x
fi

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
MODEL_CONFIG_NAME="${MODEL_CONFIG_NAME:-qwen36-35B-A3B}"
MODEL_CONFIG_PATH="${MODEL_CONFIG_DIR}/${MODEL_CONFIG_NAME}.sh"
if [ ! -f "${MODEL_CONFIG_PATH}" ]; then
    echo "ERROR: model config not found at ${MODEL_CONFIG_PATH}."
    exit 1
fi
# shellcheck source=/dev/null
source "${MODEL_CONFIG_PATH}"

###############################################################################
#                                    DIRS                                     #
###############################################################################

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/deepeyes-v2}"
MODEL_CHECKPOINT_NAME="${MODEL_CHECKPOINT_NAME:-Qwen3.6-35B-A3B}"
MODEL_RUN_NAME="${MODEL_RUN_NAME:-${MODEL_CONFIG_NAME}}"
EXP_NAME="${MODEL_RUN_NAME}-deepeyes-v2-agentic${TRAIN_PROFILE_SUFFIX:-}-${TIMESTAMP}"

if [ -z "${MODEL_DIR:-}" ] || [ -z "${DATA_DIR:-}" ] || [ -z "${SAVE_DIR:-}" ]; then
    echo "ERROR: MODEL_DIR, DATA_DIR, and SAVE_DIR must be set."
    echo "Example: MODEL_DIR=/path/to/models DATA_DIR=/path/to/data SAVE_DIR=/path/to/save bash $0"
    exit 1
fi
mkdir -p ${SAVE_DIR}

###############################################################################
#                             STARTUP CLEANUP                                 #
###############################################################################
# Sessions that die by SIGKILL / OOM / uncaught crash never run
# ApptainerJupyterSession.close(), so their /tmp/relax-apptainer-* dirs
# (kernel conn files + bind-mounted imgs/inputs holding base64-decoded
# dataset images) leak. Over 2000 rollouts × 32 batch × 8 samples ≈ 512K
# sessions, even a 1% leak fills the container disk and triggers eviction.
# mtime > 240 min is well above any single session's max wall-clock (bounded
# by max_turns × code_timeout_s ≈ tens of minutes), so live sessions are
# never touched.
find /tmp -maxdepth 1 -name 'relax-apptainer-*' -mmin +240 -exec rm -rf {} + 2>/dev/null || true
# Per-run agent log dirs accumulate one file per session; keep 3 days.
find "${SCRIPT_DIR}/log/agent" -maxdepth 1 -type d -mtime +3 -exec rm -rf {} + 2>/dev/null || true

# SIF path under DATA_DIR layout; user may override APPTAINER_IMAGE_PATH to
# point at a shared NFS copy elsewhere.
export APPTAINER_IMAGE_PATH="${APPTAINER_IMAGE_PATH:-${DATA_DIR}/sif/deepeyes_v2_kernel.sif}"
if [ ! -f "${APPTAINER_IMAGE_PATH}" ]; then
    echo "ERROR: SIF not found at ${APPTAINER_IMAGE_PATH}."
    echo "Run: DATA_DIR=${DATA_DIR} bash ${SCRIPT_DIR}/scripts/prepare.sh"
    exit 1
fi
DEEPEYES_V2_APP_PYTHON="${DEEPEYES_V2_APP_ENV_ROOT:-/tmp/deepeyes-v2-app-env}/.venv/bin/python"
if [ ! -x "${DEEPEYES_V2_APP_PYTHON}" ]; then
    echo "ERROR: DeepEyes V2 app environment not found at ${DEEPEYES_V2_APP_PYTHON}."
    echo "Run: bash ${SCRIPT_DIR}/scripts/prepare_app_env.sh"
    exit 1
fi

###############################################################################
#                              JUDGE MODEL API                                #
###############################################################################

if [ "${START_LOCAL_JUDGE:-1}" = "1" ]; then
    source "${SCRIPT_DIR}/sglang_judge_service.sh"
else
    trace_was_enabled=0
    if [[ $- == *x* ]]; then
        trace_was_enabled=1
        set +x
    fi
    : "${DEEPEYES_JUDGE_BASE_URL:?DEEPEYES_JUDGE_BASE_URL is required when START_LOCAL_JUDGE=0}"
    : "${DEEPEYES_JUDGE_MODELS:?DEEPEYES_JUDGE_MODELS is required when START_LOCAL_JUDGE=0}"
    export DEEPEYES_JUDGE_API_KEY="${DEEPEYES_JUDGE_API_KEY:-EMPTY}"
    judge_models_url="${DEEPEYES_JUDGE_BASE_URL%/}/models"
    if ! curl --fail --silent --show-error --max-time 10 \
        -H "Authorization: Bearer ${DEEPEYES_JUDGE_API_KEY}" \
        "${judge_models_url}" >/dev/null; then
        echo "ERROR: configured external judge is not ready at ${judge_models_url}."
        exit 1
    fi
    if [ "${trace_was_enabled}" = "1" ]; then
        set -x
    fi
    echo "Using configured external judge at ${DEEPEYES_JUDGE_BASE_URL}."
fi

###############################################################################
#                                  MODEL CONFIG                               #
###############################################################################

HF_CHECKPOINT_PATH="${HF_CHECKPOINT_PATH:-${MODEL_DIR}/${MODEL_CHECKPOINT_NAME}}"
REF_CHECKPOINT_PATH="${REF_CHECKPOINT_PATH:-${HF_CHECKPOINT_PATH}}"
if [ ! -f "${HF_CHECKPOINT_PATH}/config.json" ]; then
    echo "ERROR: Hugging Face checkpoint not found at ${HF_CHECKPOINT_PATH}."
    exit 1
fi

CKPT_ARGS=(
    --hf-checkpoint "${HF_CHECKPOINT_PATH}"
    --ref-load "${REF_CHECKPOINT_PATH}"
    --warm-hf-checkpoint-page-cache
    --megatron-to-hf-mode bridge
)
if [ "${DISABLE_CHECKPOINT_SAVE:-0}" != "1" ]; then
    CKPT_ARGS+=(
        --save "${SAVE_DIR}/${CHECKPOINT_NAME:-${MODEL_CHECKPOINT_NAME}-Checkpoint_v2}"
        --save-interval "${SAVE_INTERVAL:-200}"
        --max-actor-ckpt-to-keep 1
    )
fi

LORA_ARGS=()
if [ "${ENABLE_LORA:-0}" = "1" ]; then
    LORA_ARGS+=(
        --lora-rank "${LORA_RANK:-16}"
        --lora-alpha "${LORA_ALPHA:-32}"
        --lora-scope "${LORA_SCOPE:-all}"
        --lora-target-modules linear_qkv linear_proj in_proj out_proj linear_fc1 linear_fc2
        --lora-dropout "${LORA_DROPOUT:-0.0}"
        --lora-merge-mode
    )
fi

###############################################################################
#                                  DATASETS                                   #
###############################################################################

# Layout produced by scripts/prepare.sh:
#   ${DATA_DIR}/data/{perception_all_*,reason,search,vstar_test}.parquet
#   ${DATA_DIR}/sif/deepeyes_v2_kernel.sif
TRAIN_FILES=(
    "'${DATA_DIR}/data/perception_all_1.parquet@[0:5000]'"
    "'${DATA_DIR}/data/reason.parquet@[0:5000]'"
)
TEST_FILES=("${DATA_DIR}/data/vstar_test.parquet@[0:256]")
PROMPT_SET="[$(IFS=,; echo "${TRAIN_FILES[*]}")]"

###############################################################################
#                               ROLLOUT CONFIG                                #
###############################################################################

NUM_ROLLOUT="${NUM_ROLLOUT:=2000}"

# Sandbox env vars propagated into every Ray worker so the per-session
# agent process can find apptainer / search cache.
# SANDBOX_CONFIG_PATH is required — the agent reads it in _build_executor
# to find the apptainer backend YAML config (image path, bind paths, etc).
# Build a protected runtime-env file with a real encoder so paths containing
# spaces or quotes remain valid. Passing a file path keeps credentials out of
# shell traces and process arguments. Search credentials are deliberately
# excluded; use an auth.file on shared storage or provision the credential in
# the worker environment.
RUNTIME_ENV_FILE="$(mktemp "${TMPDIR:-/tmp}/deepeyes-v2-runtime-env.XXXXXX.json")"
chmod 600 "${RUNTIME_ENV_FILE}"
cleanup_deepeyes_launcher() {
    local exit_status=$?
    rm -f -- "${RUNTIME_ENV_FILE:-}"
    if declare -F cleanup_sglang_judge >/dev/null; then
        cleanup_sglang_judge || true
    fi
    trap - EXIT
    exit "${exit_status}"
}
trap cleanup_deepeyes_launcher EXIT
(
    SANDBOX_CONFIG_PATH="${SCRIPT_DIR}/apptainer_env/apptainer_config.yaml" \
    APPTAINER_IMAGE_PATH="${APPTAINER_IMAGE_PATH}" \
    RUNTIME_APP_PYTHON="${DEEPEYES_V2_APP_PYTHON}" \
    "${DEEPEYES_V2_APP_PYTHON}" - <<'PY'
import json
import os

env_vars = {
    "SANDBOX_BACKEND": "apptainer_jupyter",
    "SANDBOX_CONFIG_PATH": os.environ["SANDBOX_CONFIG_PATH"],
    "APPTAINER_IMAGE_PATH": os.environ["APPTAINER_IMAGE_PATH"],
    "DEEPEYES_V2_APP_PYTHON": os.environ["RUNTIME_APP_PYTHON"],
    "PYTHONUNBUFFERED": "1",
    "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
    "CUDA_DEVICE_MAX_CONNECTIONS": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "1"),
    "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "24"),
    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "24"),
    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "24"),
}
optional_names = (
    "NCCL_P2P_DISABLE",
    "NCCL_IB_DISABLE",
    "NCCL_NVLS_ENABLE",
    "NVSHMEM_DISABLE_NCCL",
    "APPTAINERENV_PYTHONPATH",
    "APPTAINERENV_PYTHONNOUSERSITE",
    "DEEPEYES_V2_SEARCH_CACHE_PATHS",
    "DEEPEYES_V2_AGENT_CONFIG",
    "DEEPEYES_V2_WEB_SEARCH_CONFIG",
    "DEEPEYES_V2_WEB_SEARCH_BACKEND",
    "DEEPEYES_V2_WEB_SEARCH_URL",
    "DEEPEYES_V2_WEB_SEARCH_TOP_K",
    "DEEPEYES_V2_WEB_SEARCH_TIMEOUT_S",
    "DEEPEYES_V2_WEB_SEARCH_MAX_RETRIES",
    "DEEPEYES_JUDGE_BASE_URL",
    "DEEPEYES_JUDGE_MODELS",
    "DEEPEYES_JUDGE_API_KEY",
)
for name in optional_names:
    value = os.environ.get(name)
    if value is not None:
        env_vars[name] = value
print(json.dumps({"env_vars": env_vars}))
PY
) >"${RUNTIME_ENV_FILE}"

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_SET}"
    --input-key prompt
    --label-key reward_model
    --multimodal-keys '{"image":"images"}'
    --reward-key score
    --metadata-key extra_info
    --custom-rm-path examples.deepeyes_v2_agentic.reward_deepeyes_v2.reward_func
    --use-agentic-rollout
    --agent-command ". ${SCRIPT_DIR}/run_agent_app.sh"
    --agent-cwd "${SCRIPT_DIR}"
    # Per-run agent log dir: every session's stdout/stderr is tee'd to
    # ${dir}/${session_id}.log (run_agent_app.sh). Successful AND failed
    # sessions are both kept, which is what makes hang/timeout diagnosis
    # possible — Relax's own tmpdir-based capture drops both.
    --agent-env "AGENT_DEBUG_LOG_DIR=${SCRIPT_DIR}/log/agent/${TIMESTAMP}"
    --num-rollout ${NUM_ROLLOUT}
    --rollout-batch-size ${ROLLOUT_BATCH_SIZE:-32}
    --micro-batch-size 1
    --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT:-8}
    --rollout-max-context-len ${ROLLOUT_MAX_CONTEXT_LEN:-16384}
    --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN:-4096}
    --rollout-max-prompt-len ${ROLLOUT_MAX_PROMPT_LEN:-4096}
    --rollout-temperature 1
    --global-batch-size ${GLOBAL_BATCH_SIZE:-256}
    --rollout-shuffle
    --use-streaming-dataset
)

###############################################################################
#                                EVAL CONFIG                                  #
###############################################################################

EVAL_ARGS=(
    --skip-eval-before-train
    --eval-interval 500
    --eval-prompt-data vstar ${TEST_FILES}
    --n-samples-per-eval-prompt 8
    --eval-max-response-len 4096
    --eval-top-p 0.7
    --agentic-eval-concurrency ${AGENTIC_EVAL_CONCURRENCY:-128}
)

###############################################################################
#                              ALGORITHM CONFIG                               #
###############################################################################

GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
    --eps-clip-c 3
    --use-tis
)

###############################################################################
#                              OPTIMIZER CONFIG                               #
###############################################################################

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --no-rope-fusion
)

###############################################################################
#                               SGLANG CONFIG                                 #
###############################################################################

if [ -z "${SGLANG_MODEL_LOADER_EXTRA_CONFIG:-}" ]; then
    SGLANG_MODEL_LOADER_EXTRA_CONFIG='{}'
fi
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}
    --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC:-0.8}
    --sglang-mamba-scheduler-strategy extra_buffer
    --sglang-model-loader-extra-config "${SGLANG_MODEL_LOADER_EXTRA_CONFIG}"
)

###############################################################################
#                               LOGGING CONFIG                                #
###############################################################################

LOG_ARGS=(
    --tb-project-name ${PROJECT_NAME}
    --tb-experiment-name ${EXP_NAME}
)
if [ "${USE_CLEARML:-1}" = "1" ]; then
    LOG_ARGS+=(--use-clearml)
fi
if [ "${USE_METRICS_SERVICE:-1}" = "1" ]; then
    LOG_ARGS+=(--use-metrics-service)
else
    LOG_ARGS+=(--no-use-metrics-service)
fi

###############################################################################
#                              MEGATRON CONFIG                                #
###############################################################################

TRAIN_GPU_COUNT=${TRAIN_GPU_COUNT:-8}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-4}
PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-1}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-8}
EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}
MODEL_NUM_EXPERTS=${MODEL_NUM_EXPERTS:-256}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}

if (( TRAIN_GPU_COUNT % (TENSOR_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE) != 0 )); then
    echo "ERROR: TRAIN_GPU_COUNT must be divisible by TP * PP * CP."
    exit 1
fi
if (( MODEL_NUM_EXPERTS > 0 )); then
    if (( TRAIN_GPU_COUNT % (EXPERT_TENSOR_PARALLEL_SIZE * EXPERT_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE) != 0 )); then
        echo "ERROR: TRAIN_GPU_COUNT must be divisible by expert TP * EP * PP."
        exit 1
    fi
    if (( MODEL_NUM_EXPERTS % EXPERT_MODEL_PARALLEL_SIZE != 0 )); then
        echo "ERROR: MODEL_NUM_EXPERTS must be divisible by EP."
        exit 1
    fi
elif (( EXPERT_MODEL_PARALLEL_SIZE != 1 || EXPERT_TENSOR_PARALLEL_SIZE != 1 )); then
    echo "ERROR: dense models require expert TP=1 and EP=1."
    exit 1
fi
if (( TRAIN_GPU_COUNT % ROLLOUT_NUM_GPUS_PER_ENGINE != 0 )); then
    echo "ERROR: TRAIN_GPU_COUNT must be divisible by rollout GPUs per engine."
    exit 1
fi
if (( GLOBAL_BATCH_SIZE != ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT )); then
    echo "ERROR: GLOBAL_BATCH_SIZE must equal ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT."
    exit 1
fi

MEGATRON_ARGS=(
    --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE}
    --pipeline-model-parallel-size ${PIPELINE_MODEL_PARALLEL_SIZE}
    --context-parallel-size ${CONTEXT_PARALLEL_SIZE}
    --expert-model-parallel-size ${EXPERT_MODEL_PARALLEL_SIZE}
    --expert-tensor-parallel-size ${EXPERT_TENSOR_PARALLEL_SIZE}
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-16384}
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend "${ATTENTION_BACKEND:-flash}"
    --use-dynamic-batch-size
)
if (( TENSOR_MODEL_PARALLEL_SIZE > 1 )); then
    MEGATRON_ARGS+=(--sequence-parallel)
fi
if [ -n "${DECODER_FIRST_PIPELINE_NUM_LAYERS:-}" ]; then
    MEGATRON_ARGS+=(--decoder-first-pipeline-num-layers "${DECODER_FIRST_PIPELINE_NUM_LAYERS}")
fi
if [ -n "${DECODER_LAST_PIPELINE_NUM_LAYERS:-}" ]; then
    MEGATRON_ARGS+=(--decoder-last-pipeline-num-layers "${DECODER_LAST_PIPELINE_NUM_LAYERS}")
fi

###############################################################################
#                              RESOURCE CONFIG                                #
###############################################################################

RAY_RESOURCE_ARGS=(
    --resource "{\"actor\": [1, ${TRAIN_GPU_COUNT}], \"rollout\": [1, ${TRAIN_GPU_COUNT}]}"
    --max-staleness 0
    --num-data-storage-units 1
    # --use-health-check
    --colocate
)

###############################################################################
#                                 LAUNCH JOB                                  #
###############################################################################

mkdir -p logs

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
    --runtime-env "${RUNTIME_ENV_FILE}" \
    -- python3 relax/entrypoints/train.py \
    "${RAY_RESOURCE_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${LORA_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${LOG_ARGS[@]}" \
    "${MEGATRON_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    2>&1 | tee logs/${EXP_NAME}.log
