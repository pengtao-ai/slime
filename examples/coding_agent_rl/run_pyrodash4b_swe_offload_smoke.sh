#!/usr/bin/env bash
# Tiny real rollout smoke: 1 prompt × 1 sample × 1 rollout, docker async + offload.
#
# Prerequisites:
#   export DASHSCOPE_API_KEY=...
#   export DASHSCOPE_BASE_URL=http://host:8000/v1   # OpenAI-compatible GLM
#   docker + PyroDash HF/torch_dist (defaults in parent script)
#
#   bash examples/coding_agent_rl/run_pyrodash4b_swe_offload_smoke.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

if [[ -z "${DASHSCOPE_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: set DASHSCOPE_API_KEY (or OPENAI_API_KEY) before smoke train." >&2
  exit 1
fi

export SLIME_AGENT_OFFLOAD=1
export PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/data/swe_smoke_preliz_docker.jsonl}"
export EXP_TAG="${EXP_TAG:-agent_offload_pyrodash4b_smoke}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-1}"
export SWE_AGENT_TIME_BUDGET_SEC="${SWE_AGENT_TIME_BUDGET_SEC:-300}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-180}"
# Keep smoke on fewer GPUs if desired (still uses parent async defaults unless overridden).
export NUM_GPUS="${NUM_GPUS:-8}"
export ACTOR_GPUS="${ACTOR_GPUS:-6}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"

echo "======================================================================"
echo "Offload SMOKE train (1×1×1)"
echo "  PROMPT_DATA=${PROMPT_DATA}"
echo "  NUM_ROLLOUT=${NUM_ROLLOUT} BATCH=${ROLLOUT_BATCH_SIZE} N=${N_SAMPLES_PER_PROMPT}"
echo "  DASHSCOPE_BASE_URL=${DASHSCOPE_BASE_URL:-}"
echo "======================================================================"

exec bash "${SCRIPT_DIR}/run_pyrodash4b_swe_offload_1node_docker_async.sh"
