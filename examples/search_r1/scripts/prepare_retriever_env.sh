#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -eu

RETRIEVER_ENV_ROOT="${SEARCH_R1_RETRIEVER_ENV_ROOT:-${TMPDIR:-/tmp}/search-r1-retriever}"
VENV_DIR="${RETRIEVER_ENV_ROOT}/.venv"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

mkdir -p "${RETRIEVER_ENV_ROOT}"
export UV_CACHE_DIR="${RETRIEVER_ENV_ROOT}/uv-cache"
export UV_ISOLATED=1
if [ ! -x "${VENV_DIR}/bin/python" ]; then
    uv venv --python "$(command -v python3)" --system-site-packages "${VENV_DIR}"
fi

# This CUDA-12 build is verified with the repository host's Torch 2.6/CUDA
# 12.4 stack. The generic ``faiss-gpu`` 1.14.3 wheel can load incompatible
# cuBLAS symbols before Torch and fail on the first encoder inference.
uv pip install --python "${VENV_DIR}/bin/python" faiss-gpu-cu12==1.14.1.post1

"${VENV_DIR}/bin/python" -c '
import faiss
import sys
import torch

faiss.StandardGpuResources()
probe = torch.ones((2, 2), device="cuda")
torch.mm(probe, probe)
torch.cuda.synchronize()
print(f"python={sys.executable}")
print(f"torch={torch.__version__}, cuda={torch.version.cuda}, gpu={torch.cuda.get_device_name(0)}")
'
"${VENV_DIR}/bin/python" "${EXAMPLE_DIR}/retrieval_server.py" --help >/dev/null

echo "Search-R1 retriever environment is ready."
echo "Environment: ${VENV_DIR}"
echo "Server: ${EXAMPLE_DIR}/retrieval_server.py"
echo "Start with: bash examples/search_r1/run_retriever.sh"
