#!/usr/bin/env bash
# PyroDash-4B coding-agent RL with mid-turn LLM offload (async + local Docker).
#
# Builds on the black-box agent path (claude-code -> AnthropicAdapter -> SGLang):
# when the actor emits <|llm_offload|>N<|/llm_offload|>, the adapter calls a
# remote GLM (N selects reasoning_effort: 0=off, 1-5=high, 6-9=max) and returns
# the continuation so the agent can keep editing. Train reward is
#   solved ? (1 - λ * cost_ratio) : 0
# Empty patches never count as solved (see offload.py / swe.run_evaluation).
#
# Prerequisites:
#   bash examples/coding_agent_rl/convert_pyrodash4b_to_torch_dist.sh
#   DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL pointing at an OpenAI-compatible GLM
#   docker sandboxes + pod IP (same as run_qwen35_4b_swe_1node_docker_async.sh)
#
#   export DASHSCOPE_API_KEY=...
#   export DASHSCOPE_BASE_URL=http://host:8000/v1
#   bash examples/coding_agent_rl/run_pyrodash4b_swe_offload_1node_docker_async.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ---- mid-turn offload ----
export SLIME_AGENT_OFFLOAD="${SLIME_AGENT_OFFLOAD:-1}"
export OFFLOAD_EFFICIENCY_LAMBDA="${OFFLOAD_EFFICIENCY_LAMBDA:-0.05}"
export DASHSCOPE_BASE_URL="${DASHSCOPE_BASE_URL:-http://208.64.254.187:8000/v1}"
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-sk-6137d26281697017ef07ef4da0823dc16d32acaad253ecac}"
export DASHSCOPE_MODEL="${DASHSCOPE_MODEL:-glm-5.2-fp8}"
export OFFLOAD_MAX_TOKENS="${OFFLOAD_MAX_TOKENS:-32768}"
# PyroDash-4B-SFT-0723: <|llm_offload|>=248077, <|/llm_offload|>=248078.
# Stop on the *close* tag so open + digit N can be emitted first.
export OFFLOAD_STOP_TOKEN_ID="${OFFLOAD_STOP_TOKEN_ID:-248078}"
export ROLLOUT_STOP_TOKEN_IDS="${ROLLOUT_STOP_TOKEN_IDS:-248046 248044 ${OFFLOAD_STOP_TOKEN_ID}}"
# Fewer TOKEN_FORK segments via REALIGN / rewrite-merge (passed through to Ray workers).
export SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS="${SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS:-4096}"

if [[ -z "${DASHSCOPE_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "WARNING: DASHSCOPE_API_KEY (or OPENAI_API_KEY) is unset; offload calls will fail at runtime." >&2
fi

# ---- PyroDash checkpoints ----
# SGLang loads HF vocab rows; Megatron torch_dist is padded to 248320. Use the
# padded HF copy so initial weight sync matches (see *_pad248320).
export HF_CHECKPOINT="${HF_CHECKPOINT:-/workspace/models/pyromind/PyroDash-4B-SFT-0727_pad248320}"
export REF_MODEL_PATH="${REF_MODEL_PATH:-/workspace/models/pyromind/PyroDash-4B-SFT-0727_torch_dist}"
export EXP_TAG="${EXP_TAG:-agent_offload_pyrodash4b_docker_async}"

echo "======================================================================"
echo "PyroDash coding-agent OFFLOAD (async docker)"
echo "  SLIME_AGENT_OFFLOAD=${SLIME_AGENT_OFFLOAD}"
echo "  HF_CHECKPOINT=${HF_CHECKPOINT}"
echo "  REF_MODEL_PATH=${REF_MODEL_PATH}"
echo "  ROLLOUT_STOP_TOKEN_IDS=${ROLLOUT_STOP_TOKEN_IDS}"
echo "  DASHSCOPE_BASE_URL=${DASHSCOPE_BASE_URL}"
echo "  DASHSCOPE_MODEL=${DASHSCOPE_MODEL}"
echo "  OFFLOAD_EFFICIENCY_LAMBDA=${OFFLOAD_EFFICIENCY_LAMBDA}"
echo "  SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS=${SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS}"
echo "======================================================================"

exec bash "${SCRIPT_DIR}/run_qwen35_4b_swe_1node_docker_async.sh"
