#!/usr/bin/env bash
set -euo pipefail

# Serve a local Hugging Face model directory through the OpenAI-compatible vLLM API.
# Example:
#   MODEL_PATH=/models/Qwen2.5-7B-Instruct MODEL_NAME=qwen2.5-7b-instruct \
#   bash scripts/gfm-rag/serve_vllm_llm.sh

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the uploaded local model directory}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH}")}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-EMPTY}"
DTYPE="${DTYPE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CMD=(
  "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server
  --model "${MODEL_PATH}"
  --served-model-name "${MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --api-key "${API_KEY}"
  --dtype "${DTYPE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
)

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  CMD+=(--trust-remote-code)
fi

if [[ -n "${MAX_MODEL_LEN}" ]]; then
  CMD+=(--max-model-len "${MAX_MODEL_LEN}")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  # Split on whitespace intentionally so users can pass raw vLLM flags.
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi

echo "Serving ${MODEL_NAME} from ${MODEL_PATH} on http://${HOST}:${PORT}/v1"
exec "${CMD[@]}"
