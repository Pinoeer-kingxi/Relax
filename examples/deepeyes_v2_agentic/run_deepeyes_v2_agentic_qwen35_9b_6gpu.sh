#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Six-card dense Qwen3.5-9B profile. TP=2 and PP=3 consume all six cards
# with DP=1. The first stage has eight decoder layers because it also owns
# the vision stack; the remaining two stages receive twelve layers each.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6}"
export NUM_GPUS=6
export TRAIN_GPU_COUNT=6
export MODEL_CONFIG_NAME=qwen35-9B
export MODEL_CHECKPOINT_NAME=Qwen3.5-9B
export MODEL_RUN_NAME=qwen35-9B
export MODEL_NUM_EXPERTS=0
export HF_CHECKPOINT_PATH="${HF_CHECKPOINT_PATH:-/data01/LWX/Qwen3.5-9B}"
export TENSOR_MODEL_PARALLEL_SIZE=2
export PIPELINE_MODEL_PARALLEL_SIZE=3
export CONTEXT_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=1
export EXPERT_TENSOR_PARALLEL_SIZE=1
export DECODER_FIRST_PIPELINE_NUM_LAYERS="${DECODER_FIRST_PIPELINE_NUM_LAYERS:-8}"
unset DECODER_LAST_PIPELINE_NUM_LAYERS
export ROLLOUT_NUM_GPUS_PER_ENGINE=2
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.60}"
export TRAIN_PROFILE_SUFFIX=-qwen35-9b-6gpu
export ENABLE_LORA="${ENABLE_LORA:-0}"

# Defaults preserve the eight-sample GRPO group. DP=1 imposes no additional
# divisibility constraint, so this matches the full DeepEyes-V2 batch shape.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"

exec bash "${SCRIPT_DIR}/run_deepeyes_v2_agentic.sh" "$@"
