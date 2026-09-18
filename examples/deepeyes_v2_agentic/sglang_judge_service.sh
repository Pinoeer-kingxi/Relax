#!/bin/bash
# SGLang Judge 服务：脚本启动时自动部署，结束时自动终止
# 用法: source "$(dirname "${BASH_SOURCE[0]}")/sglang_judge_service.sh"
# 依赖: MODEL_DIR, TIMESTAMP 需在 source 前已定义

# 设置默认端口，确保变量有初始值
SGLANG_JUDGE_PORT=${SGLANG_JUDGE_PORT:-30000}
SGLANG_JUDGE_MEM_FRACTION_STATIC=${SGLANG_JUDGE_MEM_FRACTION_STATIC:-0.10}
SGLANG_JUDGE_MODEL_PATH="${MODEL_DIR}/Qwen2.5-1.5B-Instruct"
SGLANG_JUDGE_PID=""
LOG_FILE="logs/sglang_judge_${TIMESTAMP}.log"

cleanup_sglang_judge() {
    if [ -n "$SGLANG_JUDGE_PID" ] && kill -0 "$SGLANG_JUDGE_PID" 2>/dev/null; then
        echo "Stopping sglang judge service (PID: $SGLANG_JUDGE_PID)..."
        kill -9 "$SGLANG_JUDGE_PID" 2>/dev/null || true
        echo "Sglang judge service stopped successfully."
    fi
}

trap cleanup_sglang_judge EXIT

# 检查必要依赖是否存在
if ! command -v curl &> /dev/null; then
    echo "Error: curl is required but not installed."
    exit 1
fi

# 检查模型路径是否存在
if [ ! -d "$SGLANG_JUDGE_MODEL_PATH" ]; then
    echo "Error: Model path not found - $SGLANG_JUDGE_MODEL_PATH"
    exit 1
fi

mkdir -p logs || { echo "Error: Failed to create logs directory"; exit 1; }

# 启动 SGLang 服务
echo "Starting sglang judge service on port $SGLANG_JUDGE_PORT..."
echo "Model path: $SGLANG_JUDGE_MODEL_PATH"
echo "Log file: $LOG_FILE"
python -m sglang.launch_server \
    --model-path "$SGLANG_JUDGE_MODEL_PATH" \
    --port "$SGLANG_JUDGE_PORT" \
    --api-key "EMPTY" \
    --mem-fraction-static "$SGLANG_JUDGE_MEM_FRACTION_STATIC" \
    --mm-feature-transport cpu \
    --disable-cuda-graph \
    > "$LOG_FILE" 2>&1 &

SGLANG_JUDGE_PID=$!
if [ -z "$SGLANG_JUDGE_PID" ]; then
    echo "Error: Failed to get PID for sglang judge service"
    exit 1
fi
echo "Sglang judge service started with PID: $SGLANG_JUDGE_PID"

# 等待服务就绪
wait_for_sglang_ready() {
    local max_attempts=60       # 最大等待次数（总计5分钟）
    local attempt=1
    local retry_interval=5      # 重试间隔（秒）
    local url="http://127.0.0.1:${SGLANG_JUDGE_PORT}/health"

    echo "Waiting for sglang judge service to be ready (max wait: $((max_attempts * retry_interval)) seconds)..."

    # 健康检查
    while [ $attempt -le $max_attempts ]; do
        local http_status
        if ! kill -0 "$SGLANG_JUDGE_PID" 2>/dev/null; then
            echo "Error: Sglang judge process exited before becoming ready."
            tail -n 40 "$LOG_FILE" 2>/dev/null || true
            return 1
        fi
        if ! http_status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "$url" 2>/dev/null); then
            http_status="000"
        fi

        if [ -z "$http_status" ]; then
            http_status="000"
        fi

        if [ "$http_status" = "200" ] || [ "$http_status" = "204" ]; then
            echo "Sglang judge service is ready (health check HTTP 200)."
            return 0
        fi

        # 仅在非最后一次尝试时输出重试信息
        if [ $attempt -lt $max_attempts ]; then
            echo "  Attempt $attempt/$max_attempts: Waiting for judge model service (HTTP status: $http_status), retrying in ${retry_interval}s..."
        fi

        sleep $retry_interval
        attempt=$((attempt + 1))
    done

    # 超时处理
    echo "Error: Sglang judge service failed to start within timeout (${max_attempts} attempts)."
    cleanup_sglang_judge
    return 1
}

# 执行等待逻辑，失败则退出
if ! wait_for_sglang_ready; then
    echo "Aborting due to sglang judge service startup failure."
    exit 1
fi


export DEEPEYES_JUDGE_API_KEY="EMPTY"
export DEEPEYES_JUDGE_BASE_URL="http://127.0.0.1:${SGLANG_JUDGE_PORT}/v1"
export DEEPEYES_JUDGE_MODELS="Qwen2.5-1.5B-Instruct"

echo "Sglang judge service is fully ready for use."
