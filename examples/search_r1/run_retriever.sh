#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -eu

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SEARCH_R1_DATA_ROOT="${SEARCH_R1_DATA_ROOT:?Set SEARCH_R1_DATA_ROOT to the prepared Search-R1 asset root.}"
RETRIEVAL_DIR="${SEARCH_R1_DATA_ROOT}/retrieval/wiki18_e5_flat"
MODEL_DIR="${SEARCH_R1_DATA_ROOT}/models/e5-base-v2"
RETRIEVER_ENV_ROOT="${SEARCH_R1_RETRIEVER_ENV_ROOT:-${TMPDIR:-/tmp}/search-r1-retriever}"
RETRIEVER_PYTHON="${RETRIEVER_ENV_ROOT}/.venv/bin/python"

exec "${RETRIEVER_PYTHON}" "${SCRIPT_DIR}/retrieval_server.py" \
    --index_path "${RETRIEVAL_DIR}/e5_Flat.index" \
    --corpus_path "${RETRIEVAL_DIR}/wiki-18.jsonl" \
    --topk "${SEARCH_R1_RETRIEVER_TOP_K:-3}" \
    --port "${SEARCH_R1_RETRIEVER_PORT:-17389}" \
    --batch_wait_ms "${SEARCH_R1_RETRIEVER_BATCH_WAIT_MS:-5}" \
    --max_batch_queries "${SEARCH_R1_RETRIEVER_MAX_BATCH_QUERIES:-512}" \
    --max_pending_requests "${SEARCH_R1_RETRIEVER_MAX_PENDING_REQUESTS:-4096}" \
    --queue_timeout_s "${SEARCH_R1_RETRIEVER_QUEUE_TIMEOUT_S:-1}" \
    --max_topk "${SEARCH_R1_RETRIEVER_MAX_TOP_K:-50}" \
    --retriever_model "${MODEL_DIR}"
