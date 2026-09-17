#!/usr/bin/env bash
# Launch vLLM for PyroDash-4B-SFT-0902 (OpenAI-compatible /v1).
#
# Example:
#   bash examples/coding_agent_rl/sft/launch_vllm_pyrodash4b_sft0902.sh
#   PORT=8066 CUDA_VISIBLE_DEVICES=7 bash .../launch_vllm_pyrodash4b_sft0902.sh
#
# Then either:
#   bash examples/coding_agent_rl/sft/run_infer_sft_traj.sh          # generate traj + think entropy
#   bash examples/coding_agent_rl/sft/run_annotate_reward1_entropy.sh  # offline annotate existing runs

set -euo pipefail

MODEL="${MODEL:-/workspace/models/pyromind/PyroDash-4B-SFT-0916}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8066}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/vllm_logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/pyrodash4b_sft0902_${PORT}_$(date +%Y%m%d_%H%M%S).log"

echo "======================================================================"
echo "vLLM serve PyroDash-4B-SFT-0902"
echo "  MODEL=${MODEL}"
echo "  SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "  HOST=${HOST} PORT=${PORT} TP=${TP}"
echo "  MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  LOG=${LOG_FILE}"
echo "======================================================================"

export CUDA_VISIBLE_DEVICES

exec vllm serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TP}" \
  --dtype auto \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  2>&1 | tee "${LOG_FILE}"
