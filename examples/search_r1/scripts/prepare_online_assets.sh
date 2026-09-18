#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

DATA_ROOT="${SEARCH_R1_DATA_ROOT:?Set SEARCH_R1_DATA_ROOT to the directory that will store all Search-R1 assets.}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

QA_REPO="PeterJinGo/nq_hotpotqa_train"
QA_REVISION="b7d80abfee334a7a91cb377544f09180d58b34f6"
CORPUS_REPO="PeterJinGo/wiki-18-corpus"
CORPUS_REVISION="69c1c00ffe7c5554c68d8548355cb22e46aabc51"
CORPUS_MEMBER="data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl"
INDEX_REPO="PeterJinGo/wiki-18-e5-index"
INDEX_REVISION="a4d31160a035f30764604f4827cd8f1d0315eb86"
INDEX_PART_A_BYTES=42949672960
INDEX_PART_B_BYTES=21609402413
INDEX_BYTES=$((INDEX_PART_A_BYTES + INDEX_PART_B_BYTES))
RETRIEVER_MODEL="intfloat/e5-base-v2"
RETRIEVER_MODEL_REVISION="f52bf8ec8c7124536f0efb74aca902b2995e5bcd"

mkdir -p "${DATA_ROOT}"
DATA_ROOT="$(cd -- "${DATA_ROOT}" && pwd -P)"

QA_DIR="${DATA_ROOT}/qa/nq_hotpotqa_train"
RETRIEVAL_DIR="${DATA_ROOT}/retrieval/wiki18_e5_flat"
RETRIEVER_MODEL_DIR="${DATA_ROOT}/models/e5-base-v2"
HF_CACHE_DIR="${DATA_ROOT}/.hf_cache"

mkdir -p "${QA_DIR}" "${RETRIEVAL_DIR}" "${RETRIEVER_MODEL_DIR}" "${HF_CACHE_DIR}"
export HF_HOME="${HF_CACHE_DIR}"

echo "Preparing Search-R1 assets under ${DATA_ROOT}"

echo "Downloading preprocessed NQ + HotpotQA train and seven-dataset evaluation data..."
hf download "${QA_REPO}" \
    train.parquet test.parquet \
    --repo-type dataset \
    --revision "${QA_REVISION}" \
    --local-dir "${QA_DIR}"

python "${SCRIPT_DIR}/prepare_data.py" \
    --input-dir "${QA_DIR}" \
    --output-dir "${DATA_ROOT}/qa/nq_hotpotqa_train_relax"

echo "Downloading the Wikipedia 2018 passage corpus..."
hf download "${CORPUS_REPO}" \
    wiki-18.jsonl.gz \
    --repo-type dataset \
    --revision "${CORPUS_REVISION}" \
    --local-dir "${RETRIEVAL_DIR}"

CORPUS_ARCHIVE="${RETRIEVAL_DIR}/wiki-18.jsonl.gz"
CORPUS_PATH="${RETRIEVAL_DIR}/wiki-18.jsonl"
if [[ ! -f "${CORPUS_PATH}" ]]; then
    CORPUS_TMP="$(mktemp "${RETRIEVAL_DIR}/wiki-18.jsonl.tmp.XXXXXX")"
    trap 'rm -f "${CORPUS_TMP}"' EXIT
    gzip -dc "${CORPUS_ARCHIVE}" | tar -xOf - "${CORPUS_MEMBER}" > "${CORPUS_TMP}"
    mv "${CORPUS_TMP}" "${CORPUS_PATH}"
    trap - EXIT
fi

echo "Downloading the E5 Flat index parts..."
INDEX_PATH="${RETRIEVAL_DIR}/e5_Flat.index"
INDEX_PARTIAL="${INDEX_PATH}.partial"
if [[ -f "${INDEX_PATH}" ]] && [[ "$(stat -c %s "${INDEX_PATH}")" -ne "${INDEX_BYTES}" ]]; then
    echo "ERROR: ${INDEX_PATH} has an unexpected size; refusing to overwrite it." >&2
    exit 1
fi
if [[ ! -f "${INDEX_PATH}" ]]; then
    if [[ ! -f "${INDEX_PARTIAL}" ]]; then
        hf download "${INDEX_REPO}" \
            part_aa \
            --repo-type dataset \
            --revision "${INDEX_REVISION}" \
            --local-dir "${RETRIEVAL_DIR}"
        if [[ "$(stat -c %s "${RETRIEVAL_DIR}/part_aa")" -ne "${INDEX_PART_A_BYTES}" ]]; then
            echo "ERROR: downloaded first E5 index part has an unexpected size." >&2
            exit 1
        fi
        mv "${RETRIEVAL_DIR}/part_aa" "${INDEX_PARTIAL}"
    fi
    partial_bytes="$(stat -c %s "${INDEX_PARTIAL}")"
    if [[ "${partial_bytes}" -lt "${INDEX_PART_A_BYTES}" ]] || [[ "${partial_bytes}" -gt "${INDEX_BYTES}" ]]; then
        echo "ERROR: ${INDEX_PARTIAL} cannot be resumed safely." >&2
        exit 1
    fi
    appended_bytes=$((partial_bytes - INDEX_PART_A_BYTES))
    if [[ "${appended_bytes}" -lt "${INDEX_PART_B_BYTES}" ]]; then
        hf download "${INDEX_REPO}" \
            part_ab \
            --repo-type dataset \
            --revision "${INDEX_REVISION}" \
            --local-dir "${RETRIEVAL_DIR}"
        if [[ "$(stat -c %s "${RETRIEVAL_DIR}/part_ab")" -ne "${INDEX_PART_B_BYTES}" ]]; then
            echo "ERROR: downloaded second E5 index part has an unexpected size." >&2
            exit 1
        fi
        tail -c +$((appended_bytes + 1)) "${RETRIEVAL_DIR}/part_ab" >> "${INDEX_PARTIAL}"
    fi
    if [[ "$(stat -c %s "${INDEX_PARTIAL}")" -ne "${INDEX_BYTES}" ]]; then
        echo "ERROR: assembled E5 index has an unexpected size." >&2
        exit 1
    fi
    mv "${INDEX_PARTIAL}" "${INDEX_PATH}"
    rm -f "${RETRIEVAL_DIR}/part_aa" "${RETRIEVAL_DIR}/part_ab"
fi

echo "Downloading the E5 query encoder..."
hf download "${RETRIEVER_MODEL}" \
    config.json model.safetensors special_tokens_map.json tokenizer.json tokenizer_config.json vocab.txt \
    --revision "${RETRIEVER_MODEL_REVISION}" \
    --local-dir "${RETRIEVER_MODEL_DIR}"

echo "Search-R1 assets are ready under ${DATA_ROOT}."
