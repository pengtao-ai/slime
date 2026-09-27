#!/usr/bin/env bash
# PyroDash-4B coding-agent RL with mid-turn LLM offload (async + local Docker).
#
# Builds on the black-box agent path (claude-code -> AnthropicAdapter -> SGLang):
# when the actor emits <|llm_offload|>N<|/llm_offload|>, the adapter calls a
# remote LLM (deepseek-v4-flash; N selects reasoning_effort: 0=off, 1-3=low,
# 4-6=high, 7-9=max via chat_template_kwargs.thinking) and returns the
# continuation so the agent can keep editing. With OFFLOAD_CONSTRAINED_DECODE=1
# (default), SGLang Free-phase stops on OPEN and bans bare CLOSE; a second
# /generate completes digit+CLOSE via ebnf so spans stay well-formed.
# Default train reward is
# help_seeking (OFFLOAD_REWARD_MODE): failed episode R=0; turn α (default 1.0)
# on valid in-think offload for failed seekers (SEEK_ONLY_WHEN_ALL_WRONG=0);
# malformed/outside-think -β (stacks with α when both on one turn).
# solved→max(floor, 1-λ*cost_ratio). Empty patches never count as solved.
#
# Prerequisites:
#   bash examples/coding_agent_rl/scripts/convert_pyrodash4b_to_torch_dist.sh
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
# sleep 3600
# NCCL runtime env (passed through to downstream exec'd script).
export NCCL_DEBUG=INFO
export NCCL_CUMEM_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export ADAPTER_MAX_TURNS_PER_SID="${ADAPTER_MAX_TURNS_PER_SID:-150}"
export SWE_AGENT_TIME_BUDGET_SEC="${SWE_AGENT_TIME_BUDGET_SEC:-600}"
echo "ADAPTER_MAX_TURNS_PER_SID=${ADAPTER_MAX_TURNS_PER_SID}"
# TF32 (Ampere+): enable via env var so it overrides any internal PyTorch default.
# This targets cuBLAS matmul; for cuDNN, prefer torch.backends.cudnn.allow_tf32 in code if needed.
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
# Train/rollout context 180k; Claude Code autoCompact at 160k (20k headroom).
export MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-180000}"
# ---- mid-turn offload ----
export SLIME_AGENT_OFFLOAD=1
export OFFLOAD_EFFICIENCY_LAMBDA=0.05
# help_seeking: fail R=0; turn α on valid in-think digit offload; malformed/outside
# -β=0.08 (stacks with α). In-think orphan CLOSE / bad payload still call GLM (N=3)
# without α; orphan OPEN and outside-think digit spans do not.
export OFFLOAD_REWARD_MODE=help_seeking
export OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG=0
export OFFLOAD_SEEK_ALPHA="${OFFLOAD_SEEK_ALPHA:-1.0}"
export OFFLOAD_MALFORMED_PENALTY="${OFFLOAD_MALFORMED_PENALTY:-0.08}"
export OFFLOAD_SEEK_BUDGET_TURN_K="${OFFLOAD_SEEK_BUDGET_TURN_K:-0}"
export OFFLOAD_SEEK_BUDGET_DECAY="${OFFLOAD_SEEK_BUDGET_DECAY:-0.5}"
export OFFLOAD_SEEK_OVERAGE_PENALTY="${OFFLOAD_SEEK_OVERAGE_PENALTY:-0}"
# Solved: r = max(floor, 1 - λ*cost_ratio).
export OFFLOAD_SOLVED_REWARD_FLOOR="${OFFLOAD_SOLVED_REWARD_FLOOR:-0.3}"
# GiGPO: A_t = A_E + w*A_S + w_r*(r_t - mean r). w_r=0 drops the turn residual.
export GIGPO_W="${GIGPO_W:-1.0}"
export GIGPO_GAMMA="${GIGPO_GAMMA:-0.95}"
export GIGPO_TURN_RESIDUAL_W="${GIGPO_TURN_RESIDUAL_W:-1.0}"
# export ADAPTER_MAX_TURNS_PER_SID="${ADAPTER_MAX_TURNS_PER_SID:-50}"
export DASHSCOPE_BASE_URL=http://208.64.254.189:8000/v1
export DASHSCOPE_API_KEY=sk-6137d26281697017ef07ef4da0823dc16d32acaad253ecac
export DASHSCOPE_MODEL=deepseek-v4-flash-0731
export OFFLOAD_MAX_TOKENS="${OFFLOAD_MAX_TOKENS:-32768}"
# PyroDash-4B: <|llm_offload|>=248077, <|/llm_offload|>=248078.
# Constrained decode (default on): Free phase stops on OPEN and logit-bias-bans
# CLOSE; a second /generate uses ebnf [0-9] CLOSE. Do not put CLOSE in free stops.
export OFFLOAD_OPEN_TOKEN_ID="${OFFLOAD_OPEN_TOKEN_ID:-248077}"
export OFFLOAD_CLOSE_TOKEN_ID="${OFFLOAD_CLOSE_TOKEN_ID:-248078}"
export OFFLOAD_STOP_TOKEN_ID="${OFFLOAD_STOP_TOKEN_ID:-${OFFLOAD_CLOSE_TOKEN_ID}}"
export OFFLOAD_CONSTRAINED_DECODE="${OFFLOAD_CONSTRAINED_DECODE:-1}"
export OFFLOAD_CLOSE_LOGIT_BIAS="${OFFLOAD_CLOSE_LOGIT_BIAS:--100}"
export ROLLOUT_STOP_TOKEN_IDS="${ROLLOUT_STOP_TOKEN_IDS:-248046 248044 ${OFFLOAD_OPEN_TOKEN_ID}}"
# Fewer TOKEN_FORK segments via REALIGN / rewrite-merge (passed through to Ray workers).
export SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS="${SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS:-8192}"
# Keep server-side prompt_ids+output_ids append-only across agent turns (THUDM#2287).
export SLIME_PRESERVE_REASONING_HISTORY="${SLIME_PRESERVE_REASONING_HISTORY:-1}"
# Embed GLM continuation into Sample.tokens with loss_mask=0 (default on).
export SLIME_OFFLOAD_EMBED_IN_TRAJECTORY="${SLIME_OFFLOAD_EMBED_IN_TRAJECTORY:-1}"
# export SLIME_OFFLOAD_EMBED_MAX_TOKENS="${SLIME_OFFLOAD_EMBED_MAX_TOKENS:-8192}"

if [[ -z "${DASHSCOPE_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "WARNING: DASHSCOPE_API_KEY (or OPENAI_API_KEY) is unset; offload calls will fail at runtime." >&2
fi

# ---- PyroDash checkpoints (BF16 train + BF16 rollout) ----
# SGLang loads padded HF vocab rows; Megatron torch_dist is padded to 248320.
# Convert once if missing:
#   HF_CHECKPOINT=/workspace/models/pyromind/PyroDash-4B-SFT-0918 \
#   SAVE=/workspace/models/pyromind/PyroDash-4B-SFT-0918_torch_dist \
#   bash examples/coding_agent_rl/scripts/convert_pyrodash4b_to_torch_dist.sh
export HF_CHECKPOINT="${HF_CHECKPOINT:-/workspace/models/pyromind/PyroDash-4B-SFT-0918}"
export REF_MODEL_PATH="${REF_MODEL_PATH:-/workspace/models/pyromind/PyroDash-4B-SFT-0918_torch_dist}"
export EXP_TAG="${EXP_TAG:-agent_offload_pyrodash4b_sft0918_gigpo}"
# W&B (picked up by run_qwen35_4b_swe_1node_async.sh when WANDB_API_KEY is set).
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_PROJECT="${WANDB_PROJECT:-swe-slime}"
export WANDB_GROUP="${WANDB_GROUP:-${EXP_TAG}}"
# FP8 KV cache for longer agent decode contexts (rollout only; weights stay BF16).
export SGLANG_KV_CACHE_DTYPE="${SGLANG_KV_CACHE_DTYPE:-fp8_e4m3}"

# Pre-baked ScaleSWE agent images (Node22 + Claude Code + pre_commands).
# Override with PROMPT_DATA=.../swe_train_scaleswe_200.jsonl for the raw bases.
# export PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/data/swe_train_scaleswe_200_baked.jsonl}"
# export PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/data/release/mixed_reward1_agents_baked.jsonl}"
# Aliyun VPC images (pyromind-registry-vpc.../scaleswe-agent|tmax-agent).
export PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/data/release/mixed_reward1_agents_baked_ali_exist.jsonl}"

# Sandbox pool (default on): prefetch next rollout's containers during current step.
# Same image: first warm gated, then parallel. Max defaults to 2× rollout sample
# count (current in-flight + next-step warm) via the async launcher.
export SANDBOX_POOL="${SANDBOX_POOL:-1}"

# Fan-out for this launcher (overrides docker_async.sh default batch=4).
# 16 prompts × 8 samples = 128 / step; GLOBAL_BATCH follows unless overridden.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"

# Multi-agent CLI packages for mixed_*_agents.jsonl (codex/pi/opencode/miniswe).
# Claude Code + Node are set in run_qwen35_4b_swe_1node_async.sh; these four must
# also reach Ray workers or non-CC rows abort with KeyError at boot.
_TB="${SCRIPT_DIR}/tarballs"
export SLIME_AGENT_CODEX_TARBALL="${SLIME_AGENT_CODEX_TARBALL:-${_TB}/openai-codex-local.tgz}"
export SLIME_AGENT_PI_TARBALL="${SLIME_AGENT_PI_TARBALL:-${_TB}/pi-coding-agent-local.tgz}"
export SLIME_AGENT_OPENCODE_TARBALL="${SLIME_AGENT_OPENCODE_TARBALL:-${_TB}/opencode-ai-local-linux-x64.tgz}"
export SLIME_AGENT_MINISWE_WHEEL="${SLIME_AGENT_MINISWE_WHEEL:-${_TB}/miniswe-wheels}"
# 判断agent包是否存在，新镜像已预安装 CODEX / PI / OPENCODE / MINISWE
for _agent_pkg in \
  SLIME_AGENT_CODEX_TARBALL \
  SLIME_AGENT_PI_TARBALL \
  SLIME_AGENT_OPENCODE_TARBALL \
  SLIME_AGENT_MINISWE_WHEEL
do
  _path="${!_agent_pkg}"
  if [[ ! -e "${_path}" ]]; then
    echo "ERROR: ${_agent_pkg} missing: ${_path}" >&2
    echo "  Mixed-agent PROMPT_DATA requires host tarballs under ${_TB}/ (see README)." >&2
    # exit 1
  fi
done
unset _TB _agent_pkg _path

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
  export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
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
echo "  WANDB_PROJECT=${WANDB_PROJECT:-<unset>} WANDB_GROUP=${WANDB_GROUP:-<unset>} WANDB_API_KEY=${WANDB_API_KEY:+set}"
echo "  SGLANG_KV_CACHE_DTYPE=${SGLANG_KV_CACHE_DTYPE:-<unset>}"
echo "  PROMPT_DATA=${PROMPT_DATA}"
echo "  ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE} N_SAMPLES=${N_SAMPLES_PER_PROMPT} GLOBAL_BATCH=${GLOBAL_BATCH_SIZE}"
echo "  SLIME_AGENT_CODEX_TARBALL=${SLIME_AGENT_CODEX_TARBALL}"
echo "  SLIME_AGENT_PI_TARBALL=${SLIME_AGENT_PI_TARBALL}"
echo "  SLIME_AGENT_OPENCODE_TARBALL=${SLIME_AGENT_OPENCODE_TARBALL}"
echo "  SLIME_AGENT_MINISWE_WHEEL=${SLIME_AGENT_MINISWE_WHEEL}"
echo "  ROLLOUT_STOP_TOKEN_IDS=${ROLLOUT_STOP_TOKEN_IDS}"
echo "  DASHSCOPE_BASE_URL=${DASHSCOPE_BASE_URL}"
echo "  DASHSCOPE_MODEL=${DASHSCOPE_MODEL}"
echo "  MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN}"
echo "  OFFLOAD_EFFICIENCY_LAMBDA=${OFFLOAD_EFFICIENCY_LAMBDA}"
echo "  OFFLOAD_REWARD_MODE=${OFFLOAD_REWARD_MODE}"
echo "  OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG=${OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG}"
echo "  OFFLOAD_SEEK_ALPHA=${OFFLOAD_SEEK_ALPHA}"
echo "  OFFLOAD_MALFORMED_PENALTY=${OFFLOAD_MALFORMED_PENALTY:-}"
echo "  OFFLOAD_CONSTRAINED_DECODE=${OFFLOAD_CONSTRAINED_DECODE:-1}"
echo "  OFFLOAD_OPEN_TOKEN_ID=${OFFLOAD_OPEN_TOKEN_ID} OFFLOAD_CLOSE_TOKEN_ID=${OFFLOAD_CLOSE_TOKEN_ID}"
echo "  ROLLOUT_STOP_TOKEN_IDS=${ROLLOUT_STOP_TOKEN_IDS} (free: EOS+OPEN; ebnf adds CLOSE)"
echo "  OFFLOAD_SOLVED_REWARD_FLOOR=${OFFLOAD_SOLVED_REWARD_FLOOR:-}"
echo "  GIGPO_W=${GIGPO_W:-} GIGPO_GAMMA=${GIGPO_GAMMA:-} GIGPO_TURN_RESIDUAL_W=${GIGPO_TURN_RESIDUAL_W:-}"
echo "  CUSTOM_ADVANTAGE_FUNCTION_PATH=${CUSTOM_ADVANTAGE_FUNCTION_PATH:-examples.coding_agent_rl.gigpo_advantage.compute_gigpo_advantages}"
echo "  SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS=${SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS}"
echo "  SLIME_PRESERVE_REASONING_HISTORY=${SLIME_PRESERVE_REASONING_HISTORY}"
echo "  TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE}"
echo "  DEBUG_TRAIN_MEM=${DEBUG_TRAIN_MEM}"
if [[ "${DEBUG_TRAIN_MEM}" == "1" ]]; then
  echo "  LOAD_DEBUG_ROLLOUT_DATA=${LOAD_DEBUG_ROLLOUT_DATA}"
  echo "  RECORD_MEMORY_HISTORY=${RECORD_MEMORY_HISTORY} NUM_ROLLOUT=${NUM_ROLLOUT} NUM_STEPS_PER_ROLLOUT=${NUM_STEPS_PER_ROLLOUT} MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}"
  echo "  ACTOR_GPUS=${ACTOR_GPUS} ROLLOUT_GPUS=${ROLLOUT_GPUS} CP_SIZE=${CP_SIZE}"
fi
echo "======================================================================"

# ray job stop leaves the head/workers alive; they keep appending session logs
# under /tmp/ray (worker.*.err, raylet.out, job-driver, spill). Kill + purge
# before relaunch so stale sessions do not fill the overlay disk.
echo "Stopping leftover Ray and cleaning /tmp/ray ..."
ray stop --force 2>/dev/null || true
pkill -9 -f 'ray::|gcs_server|raylet|ray\.dashboard' 2>/dev/null || true
sleep 2
rm -rf /tmp/ray
echo "  /tmp after Ray cleanup: $(df -h /tmp | awk 'NR==2{print $3" used / "$4" avail ("$5")"}')"

if [[ "${DEBUG_TRAIN_MEM}" != "1" ]]; then
  docker ps -aq --filter name=slime-sb- | xargs -r docker rm -f
fi

exec bash "${SCRIPT_DIR}/run_qwen35_4b_swe_1node_docker_async.sh"
