#!/usr/bin/env bash
# PyroDash-4B coding-agent RL with mid-turn LLM offload (async + local Docker).
#
# Builds on the black-box agent path (claude-code -> AnthropicAdapter -> SGLang):
# when the actor emits <|llm_offload|>N<|/llm_offload|>, the adapter calls a
# remote LLM (deepseek-v4-flash; N selects reasoning_effort: 0=off, 1-3=low,
# 4-6=high, 7-9=max via chat_template_kwargs.thinking) and returns the
# continuation so the agent can keep editing. Default train reward is
# help_seeking (OFFLOAD_REWARD_MODE) with OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG:
# α only when every sibling in the GRPO group failed and this traj did
# in-think offload; otherwise unsolved→0 / solved→(1-λ*cost_ratio).
# Empty patches never count as solved.
#
# Prerequisites:
#   bash examples/coding_agent_rl/convert_pyrodash4b_to_torch_dist.sh
#   DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL pointing at OpenAI-compatible deepseek
#   docker sandboxes + pod IP (same as run_qwen35_4b_swe_1node_docker_async.sh)
#
# Precision: Megatron train + SGLang rollout both use BF16 (padded HF / torch_dist).
# Optional: SGLANG_KV_CACHE_DTYPE=fp8_e4m3 for longer agent contexts.
#
#   export DASHSCOPE_API_KEY=...
#   export DASHSCOPE_BASE_URL=http://host:8000/v1
#   bash examples/coding_agent_rl/run_pyrodash4b_swe_offload_1node_docker_async.sh
#
# Train-only CUDA memory snapshot (default ON; no SGLang / Docker agents):
#   DEBUG_TRAIN_MEM=1 bash ...   # default
#   DEBUG_TRAIN_MEM=0 bash ...   # full async RL with Docker
#   LOAD_DEBUG_ROLLOUT_DATA=/path/to/rollout_{rollout_id}.pt bash ...

set -euo pipefail

# NCCL runtime env (passed through to downstream exec'd script).
export NCCL_DEBUG=INFO
export NCCL_CUMEM_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# TF32 (Ampere+): enable via env var so it overrides any internal PyTorch default.
# This targets cuBLAS matmul; for cuDNN, prefer torch.backends.cudnn.allow_tf32 in code if needed.
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
# ---- mid-turn offload ----
export SLIME_AGENT_OFFLOAD=1
export OFFLOAD_EFFICIENCY_LAMBDA=0.3
# help_seeking + only-all-wrong: α only if the whole GRPO group failed
# (see offload.shape_group_help_seeking_rewards). Else do not encourage offload.
# Set OFFLOAD_REWARD_MODE=cost_aware to restore the old "fail → 0" shaping.
export OFFLOAD_REWARD_MODE=help_seeking
export OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG=1
# export OFFLOAD_REWARD_MODE="${OFFLOAD_REWARD_MODE:-cost_aware}"
export OFFLOAD_SEEK_ALPHA=0.1
export OFFLOAD_SEEK_EMPTY_SCALE=0.5
export OFFLOAD_UNIQUE_SOLVER_BONUS=0.15
export DASHSCOPE_BASE_URL=http://208.64.254.187:8001/v1
export DASHSCOPE_API_KEY=sk-6137d26281697017ef07ef4da0823dc16d32acaad253ecac
export DASHSCOPE_MODEL=deepseek-v4-flash-0731
export OFFLOAD_MAX_TOKENS="${OFFLOAD_MAX_TOKENS:-32768}"
# PyroDash-4B-SFT-0723: <|llm_offload|>=248077, <|/llm_offload|>=248078.
# Stop on the *close* tag so open + digit N can be emitted first.
export OFFLOAD_STOP_TOKEN_ID="${OFFLOAD_STOP_TOKEN_ID:-248078}"
export ROLLOUT_STOP_TOKEN_IDS="${ROLLOUT_STOP_TOKEN_IDS:-248046 248044 ${OFFLOAD_STOP_TOKEN_ID}}"
# Fewer TOKEN_FORK segments via REALIGN / rewrite-merge (passed through to Ray workers).
export SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS="${SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS:-160000}"
# Embed GLM continuation into Sample.tokens with loss_mask=0 (default on).
export SLIME_OFFLOAD_EMBED_IN_TRAJECTORY="${SLIME_OFFLOAD_EMBED_IN_TRAJECTORY:-1}"
# export SLIME_OFFLOAD_EMBED_MAX_TOKENS="${SLIME_OFFLOAD_EMBED_MAX_TOKENS:-8192}"

if [[ -z "${DASHSCOPE_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "WARNING: DASHSCOPE_API_KEY (or OPENAI_API_KEY) is unset; offload calls will fail at runtime." >&2
fi

# ---- PyroDash checkpoints (BF16 train + BF16 rollout) ----
# SGLang loads padded HF vocab rows; Megatron torch_dist is padded to 248320.
export HF_CHECKPOINT="${HF_CHECKPOINT:-/workspace/models/pyromind/PyroDash-4B-SFT-0803}"
export REF_MODEL_PATH="${REF_MODEL_PATH:-/workspace/models/pyromind/PyroDash-4B-SFT-0803_torch_dist}"
export EXP_TAG="${EXP_TAG:-agent_offload_pyrodash4b_docker_async}"
# FP8 KV cache for longer agent decode contexts (rollout only; weights stay BF16).
export SGLANG_KV_CACHE_DTYPE="${SGLANG_KV_CACHE_DTYPE:-fp8_e4m3}"

# Pre-baked ScaleSWE agent images (Node22 + Claude Code + pre_commands).
# Override with PROMPT_DATA=.../swe_train_scaleswe_200.jsonl for the raw bases.
# export PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/data/swe_train_scaleswe_200_baked.jsonl}"
export PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/data/mixed400.jsonl}"

# ---- train-only CUDA memory snapshot (exclude SGLang rollout) ----
# Default ON for this launcher while debugging train OOM. Set DEBUG_TRAIN_MEM=0 for full RL.
export DEBUG_TRAIN_MEM="${DEBUG_TRAIN_MEM:-0}"
if [[ "${DEBUG_TRAIN_MEM}" == "1" ]]; then
  # 32-sample dump (trimmed from the large async dump) for fast mem profiling.
  _DEFAULT_DUMP="${SLIME_DIR}/runs/debug_rollout_32/rollout_dumps/rollout_{rollout_id}.pt"
  export LOAD_DEBUG_ROLLOUT_DATA="${LOAD_DEBUG_ROLLOUT_DATA:-${_DEFAULT_DUMP}}"
  export RECORD_MEMORY_HISTORY="${RECORD_MEMORY_HISTORY:-1}"
  # Dump after N train calls (rollout_id == N-1). Keep NUM_ROLLOUT >= N.
  export MEMORY_SNAPSHOT_NUM_STEPS="${MEMORY_SNAPSHOT_NUM_STEPS:-4}"
  export NUM_ROLLOUT="${NUM_ROLLOUT:-4}"
  export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
  export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
  # Match dump size: 32 samples / step.
  export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
  export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
  export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
  # Same train parallel as docker_async defaults; no rollout engines.
  export NUM_GPUS="${NUM_GPUS:-8}"
  export ACTOR_GPUS="${ACTOR_GPUS:-6}"
  export ROLLOUT_GPUS="${ROLLOUT_GPUS:-0}"
  export TP_SIZE="${TP_SIZE:-1}"
  export CP_SIZE="${CP_SIZE:-6}"
  export DEBUG_TRAIN_ONLY=1
  if [[ ! -f "${LOAD_DEBUG_ROLLOUT_DATA/\{rollout_id\}/0}" ]]; then
    echo "ERROR: LOAD_DEBUG_ROLLOUT_DATA missing rollout_0.pt:" >&2
    echo "  ${LOAD_DEBUG_ROLLOUT_DATA}" >&2
    echo "  Override LOAD_DEBUG_ROLLOUT_DATA=/path/to/rollout_{rollout_id}.pt" >&2
    exit 1
  fi
fi

echo "======================================================================"
echo "PyroDash coding-agent OFFLOAD (async docker, BF16 train + BF16 rollout)"
echo "  SLIME_AGENT_OFFLOAD=${SLIME_AGENT_OFFLOAD}"
echo "  HF_CHECKPOINT=${HF_CHECKPOINT}"
echo "  REF_MODEL_PATH=${REF_MODEL_PATH}"
echo "  SGLANG_KV_CACHE_DTYPE=${SGLANG_KV_CACHE_DTYPE:-<unset>}"
echo "  PROMPT_DATA=${PROMPT_DATA}"
echo "  ROLLOUT_STOP_TOKEN_IDS=${ROLLOUT_STOP_TOKEN_IDS}"
echo "  DASHSCOPE_BASE_URL=${DASHSCOPE_BASE_URL}"
echo "  DASHSCOPE_MODEL=${DASHSCOPE_MODEL}"
echo "  OFFLOAD_EFFICIENCY_LAMBDA=${OFFLOAD_EFFICIENCY_LAMBDA}"
echo "  OFFLOAD_REWARD_MODE=${OFFLOAD_REWARD_MODE}"
echo "  OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG=${OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG}"
echo "  OFFLOAD_SEEK_ALPHA=${OFFLOAD_SEEK_ALPHA}"
echo "  SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS=${SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS}"
echo "  TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE}"
echo "  DEBUG_TRAIN_MEM=${DEBUG_TRAIN_MEM}"
if [[ "${DEBUG_TRAIN_MEM}" == "1" ]]; then
  echo "  LOAD_DEBUG_ROLLOUT_DATA=${LOAD_DEBUG_ROLLOUT_DATA}"
  echo "  RECORD_MEMORY_HISTORY=${RECORD_MEMORY_HISTORY} NUM_ROLLOUT=${NUM_ROLLOUT} NUM_STEPS_PER_ROLLOUT=${NUM_STEPS_PER_ROLLOUT} MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}"
  echo "  ACTOR_GPUS=${ACTOR_GPUS} ROLLOUT_GPUS=${ROLLOUT_GPUS} CP_SIZE=${CP_SIZE}"
fi
echo "======================================================================"

if [[ "${DEBUG_TRAIN_MEM}" != "1" ]]; then
  docker ps -aq --filter name=slime-sb- | xargs -r docker rm -f
fi

exec bash "${SCRIPT_DIR}/run_qwen35_4b_swe_1node_docker_async.sh"
