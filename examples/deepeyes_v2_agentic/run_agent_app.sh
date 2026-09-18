#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -eu

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}"

# Strip the trailing slash so clients that append /chat/completions do not
# produce a double-slash URL.
export OPENAI_BASE_URL="${RELAX_BASE_URL%/}"
export OPENAI_API_KEY="${RELAX_SESSION_ID}"

# httpx does not consistently honor CIDR entries such as 10.0.0.0/8 in
# NO_PROXY.  Ray advertises the agentic API by its concrete node IP, so add
# that host explicitly or requests may be sent through an ambient HTTP proxy
# and fail with an empty 502 response.  Preserve proxy access for genuinely
# external search APIs.
RELAX_API_AUTHORITY="${OPENAI_BASE_URL#*://}"
RELAX_API_HOST="${RELAX_API_AUTHORITY%%[:/]*}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${RELAX_API_HOST}"
export no_proxy="${no_proxy:+${no_proxy},}${RELAX_API_HOST}"

# Persist each session's stdout and stderr without spawning helper processes.
# Functional agent output is written separately through RELAX_OUTPUT_JSON.
if [ -n "${AGENT_DEBUG_LOG_DIR:-}" ]; then
    mkdir -p "${AGENT_DEBUG_LOG_DIR}"
    AGENT_LOG_FILE="${AGENT_DEBUG_LOG_DIR}/${RELAX_SESSION_ID:-unknown}.log"
    exec >> "${AGENT_LOG_FILE}" 2>&1
fi

exec "${DEEPEYES_V2_APP_PYTHON}" -m app.agent \
    --input-json "${RELAX_INPUT_JSON}" \
    --output-json "${RELAX_OUTPUT_JSON}"
