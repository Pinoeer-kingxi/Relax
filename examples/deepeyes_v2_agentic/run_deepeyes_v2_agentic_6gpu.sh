#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Six-card Qwen3.6 profile for 48 GiB cards. PP=3 is required because the
# BF16 actor consumes more than 47 GiB/rank with TP=2/PP=1/EP=2. Attention
# uses DP=2 across three pipeline stages; experts use EP=2 within each stage.
# The first/last stages carry the vision/embedding and output-head overhead,
# so the 40 decoder layers are distributed 12/16/12.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6}"
export NUM_GPUS=6
export TRAIN_GPU_COUNT=6
export TENSOR_MODEL_PARALLEL_SIZE=1
export PIPELINE_MODEL_PARALLEL_SIZE=3
export CONTEXT_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=2
export EXPERT_TENSOR_PARALLEL_SIZE=1
export DECODER_FIRST_PIPELINE_NUM_LAYERS="${DECODER_FIRST_PIPELINE_NUM_LAYERS:-12}"
export DECODER_LAST_PIPELINE_NUM_LAYERS="${DECODER_LAST_PIPELINE_NUM_LAYERS:-12}"
export ROLLOUT_NUM_GPUS_PER_ENGINE=2
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.88}"
# The 35B checkpoint is stored as 26 large shards on a rotational data disk.
# SGLang otherwise starts eight loader threads per TP rank (48 concurrent
# readers here), turning sequential reads into pathological random I/O.
if [ -z "${SGLANG_MODEL_LOADER_EXTRA_CONFIG:-}" ]; then
    export SGLANG_MODEL_LOADER_EXTRA_CONFIG='{"enable_multithread_load":false}'
fi
export TRAIN_PROFILE_SUFFIX="-6gpu"
export ENABLE_LORA="${ENABLE_LORA:-1}"
export LORA_RANK="${LORA_RANK:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_SCOPE="${LORA_SCOPE:-all}"
# Qwen3.6 uses 256-wide attention heads. Transformer Engine 2.14 disables
# FlashAttention 2 for that head size on sm86 (RTX A6000), while the auto
# selector correctly falls back to UnfusedDotProductAttention. Keep this
# profile portable by allowing an explicit override on newer architectures.
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"

# Defaults preserve the eight-sample GRPO group and make the effective batch
# divisible by the attention data-parallel size (6 / PP3 = 2).
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-24}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-192}"

exec bash "${SCRIPT_DIR}/run_deepeyes_v2_agentic.sh" "$@"
