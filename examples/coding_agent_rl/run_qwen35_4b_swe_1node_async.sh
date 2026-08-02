#!/usr/bin/env bash
# Async (train_async.py) SWE coding-agent RL on 1 node (8 GPUs) with Qwen3.5-4B.
#
# Unlike run_qwen35_4b_swe_1node.sh (--colocate + train.py), this script:
#   * uses train_async.py so the next rollout starts while the actor trains
#   * splits GPUs between Megatron and SGLang (colocate is not supported)
#   * saves Megatron checkpoints via --save / --save-interval
#
# Default GPU split on 8 devices: 6 train + 2 rollout.
# SWE agents are usually sandbox-bound; actor_train dominated wall clock at 4+4 / GBS=128.
# Parallelism: DP = ACTOR_GPUS / (TP * PP * CP). Unset CP → max CP / DP=1.
# Docker wrapper defaults: CP=2 → DP=3 with ACTOR_GPUS=6.
#
# Run from a long-lived shell / tmux:
#   bash examples/coding_agent_rl/run_qwen35_4b_swe_1node_docker_async.sh

# Best-effort cleanup so a rerun does not collide with stale workers.
pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
sleep 3
pkill -9 ray || true

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
source "${SLIME_DIR}/scripts/models/qwen3.5-4B.sh"

# ============ GPU split (async: no colocate) ============
NUM_GPUS="${NUM_GPUS:-8}"
ACTOR_GPUS="${ACTOR_GPUS:-6}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-$((NUM_GPUS - ACTOR_GPUS))}"
# Train-only (load saved rollout dumps): no SGLang, rollout GPUs may be 0.
LOAD_DEBUG_ROLLOUT_DATA="${LOAD_DEBUG_ROLLOUT_DATA:-}"
DEBUG_TRAIN_ONLY="${DEBUG_TRAIN_ONLY:-0}"
if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA}" ]]; then
  DEBUG_TRAIN_ONLY=1
fi
if [[ "${DEBUG_TRAIN_ONLY}" == "1" ]]; then
  ACTOR_GPUS="${ACTOR_GPUS:-${NUM_GPUS}}"
  ROLLOUT_GPUS="${ROLLOUT_GPUS:-0}"
fi

if (( ACTOR_GPUS + ROLLOUT_GPUS > NUM_GPUS )); then
  echo "ERROR: ACTOR_GPUS(${ACTOR_GPUS})+ROLLOUT_GPUS(${ROLLOUT_GPUS}) > NUM_GPUS(${NUM_GPUS})" >&2
  exit 1
fi
if (( ACTOR_GPUS < 1 )); then
  echo "ERROR: ACTOR_GPUS must be >= 1" >&2
  exit 1
fi
if [[ "${DEBUG_TRAIN_ONLY}" != "1" ]] && (( ROLLOUT_GPUS < 1 )); then
  echo "ERROR: async mode needs ROLLOUT_GPUS>=1 unless DEBUG_TRAIN_ONLY / LOAD_DEBUG_ROLLOUT_DATA" >&2
  exit 1
fi

# Train parallelism: TP * PP * CP must divide ACTOR_GPUS; remainder is DP.
# Prefer larger CP for long agent trajectories; set CP_SIZE=2 for DP=3 on 6 GPUs.
export TP_SIZE="${TP_SIZE:-1}"
export PP_SIZE="${PP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-$((ACTOR_GPUS / TP_SIZE / PP_SIZE))}"
_MODEL_PARALLEL=$((TP_SIZE * PP_SIZE * CP_SIZE))
if (( _MODEL_PARALLEL < 1 || ACTOR_GPUS % _MODEL_PARALLEL != 0 )); then
  echo "ERROR: ACTOR_GPUS(${ACTOR_GPUS}) must be divisible by TP(${TP_SIZE})*PP(${PP_SIZE})*CP(${CP_SIZE})=${_MODEL_PARALLEL}" >&2
  exit 1
fi
DP_SIZE=$((ACTOR_GPUS / _MODEL_PARALLEL))
# Qwen3.5/Next GDN kernel: fla (default) or flashqla (SM90+).
export QWEN_GDN_BACKEND="${QWEN_GDN_BACKEND:-fla}"

# ============ rollout engine ============
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
ROLLOUT_MEM_UTILIZATION="${ROLLOUT_MEM_UTILIZATION:-0.7}"
if (( ROLLOUT_GPUS % ROLLOUT_TP_SIZE != 0 )); then
  echo "ERROR: ROLLOUT_GPUS(${ROLLOUT_GPUS}) must be divisible by ROLLOUT_TP_SIZE(${ROLLOUT_TP_SIZE})" >&2
  exit 1
fi

# ============ context length ============
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-160000}"
MAX_GEN_LEN="${MAX_GEN_LEN:-160000}"
# Per-GPU token budget after CP split. Sync uses MAX/4 with CP=4.
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-$((MAX_CONTEXT_LEN / CP_SIZE))}"
# Megatron asserts seq_length % (2 * CP) == 0. Align up when needed
# (e.g. CP=6 → 160000 % 12 != 0 → 200004).
SEQ_LENGTH="${SEQ_LENGTH:-${MAX_CONTEXT_LEN}}"
if (( CP_SIZE > 1 )); then
  _cp_align=$((CP_SIZE * 2))
  if (( SEQ_LENGTH % _cp_align != 0 )); then
    SEQ_LENGTH=$(( (SEQ_LENGTH + _cp_align - 1) / _cp_align * _cp_align ))
  fi
fi

# ============ paths — override before launching ============
HF_CHECKPOINT="${HF_CHECKPOINT:-/workspace/models/Qwen/Qwen3.5-4B}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/workspace/models/Qwen/Qwen3.5-4B_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/data/swe_train_scaleswe_200.jsonl}"

# Fan-out defaults (docker_async wrapper may override).
export SLIME_AGENT_E2B_USE_TEMPLATE="${SLIME_AGENT_E2B_USE_TEMPLATE:-0}"
if [[ "${SLIME_AGENT_E2B_USE_TEMPLATE}" == "1" ]]; then
  ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
  N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
  SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-4}"
else
  ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
  N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
  SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-16}"
fi

EXP_TAG="${EXP_TAG:-agent_only_qwen35_4b_async}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${SLIME_DIR}/runs/${EXP_TAG}_${STAMP}}"
SAVE_DIR="${SAVE_DIR:-${RUN_ROOT}/checkpoints}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
UPDATE_WEIGHTS_INTERVAL="${UPDATE_WEIGHTS_INTERVAL:-1}"
# eos/pad; offload launchers append <|/llm_offload|> id (e.g. 248078).
ROLLOUT_STOP_TOKEN_IDS="${ROLLOUT_STOP_TOKEN_IDS:-248046 248044}"

# ============ logging ============
LOG_DIR="${RUN_ROOT}"
mkdir -p "${LOG_DIR}/rollout_dumps" "${LOG_DIR}/timelines" "${SAVE_DIR}"
LOG_FILE="${LOG_DIR}/run.log"
echo "======================================================================"
echo "Async training log: ${LOG_FILE}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "SAVE_DIR=${SAVE_DIR}  SAVE_INTERVAL=${SAVE_INTERVAL}"
echo "ACTOR_GPUS=${ACTOR_GPUS} ROLLOUT_GPUS=${ROLLOUT_GPUS} (TP=${TP_SIZE} PP=${PP_SIZE} CP=${CP_SIZE} DP=${DP_SIZE} seq=${SEQ_LENGTH} max_tokens/gpu=${MAX_TOKENS_PER_GPU})"
echo "ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE} N_SAMPLES=${N_SAMPLES_PER_PROMPT} GLOBAL_BATCH=${GLOBAL_BATCH_SIZE}"
echo "QWEN_GDN_BACKEND=${QWEN_GDN_BACKEND}"
echo "SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS=${SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS:-160000}"
echo "======================================================================"
if (( GLOBAL_BATCH_SIZE % DP_SIZE != 0 )); then
  echo "ERROR: GLOBAL_BATCH_SIZE(${GLOBAL_BATCH_SIZE}) must be divisible by DP(${DP_SIZE})" >&2
  exit 1
fi

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_MODEL_PATH}"
   --save "${SAVE_DIR}"
   --save-interval "${SAVE_INTERVAL}"
)

ROLLOUT_ARGS=(
   --custom-generate-function-path examples.coding_agent_rl.generate.generate
   --custom-rollout-log-function-path examples.coding_agent_rl.log_rollout_timeline.log_rollout_timeline
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --num-rollout ${NUM_ROLLOUT:-50}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
   --rollout-max-context-len ${MAX_CONTEXT_LEN}
   --rollout-max-response-len ${MAX_GEN_LEN}
   --rollout-temperature 1.0
   --rollout-stop-token-ids ${ROLLOUT_STOP_TOKEN_IDS}
   --num-steps-per-rollout ${NUM_STEPS_PER_ROLLOUT:-1}
   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --micro-batch-size ${MICRO_BATCH_SIZE:-1}
   --update-weights-interval "${UPDATE_WEIGHTS_INTERVAL}"
)
if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA}" ]]; then
  ROLLOUT_ARGS+=(--load-debug-rollout-data "${LOAD_DEBUG_ROLLOUT_DATA}")
else
  ROLLOUT_ARGS+=(--save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt")
fi
if [[ "${DEBUG_TRAIN_ONLY}" == "1" && -z "${LOAD_DEBUG_ROLLOUT_DATA}" ]]; then
  ROLLOUT_ARGS+=(--debug-train-only)
fi

# Per-GPU token budget already set above (MAX_TOKENS_PER_GPU).
PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --pipeline-model-parallel-size ${PP_SIZE}
   --context-parallel-size ${CP_SIZE}
   --seq-length ${SEQ_LENGTH}
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
   --log-probs-chunk-size 1024
   --use-dynamic-batch-size
   --qwen-gdn-backend ${QWEN_GDN_BACKEND}
)
# Megatron sequence-parallel requires TP>1.
if (( TP_SIZE > 1 )); then
  PERF_ARGS+=(--sequence-parallel)
fi

ALGO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus ${ROLLOUT_GPUS}
   --rollout-num-gpus-per-engine ${ROLLOUT_TP_SIZE}
   --sglang-mem-fraction-static ${ROLLOUT_MEM_UTILIZATION}
   --sglang-tool-call-parser qwen3_coder
   --sglang-reasoning-parser qwen3
   --sglang-cuda-graph-max-bs 64
)
# Optional FP8 KV cache for long agent contexts (rollout only; does not affect Megatron BF16).
if [[ -n "${SGLANG_KV_CACHE_DTYPE:-}" ]]; then
  SGLANG_ARGS+=(--sglang-kv-cache-dtype "${SGLANG_KV_CACHE_DTYPE}")
fi

# No --colocate: train_async.py asserts against it.
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# Optional CUDA memory snapshot (rank 0 only; see slime/utils/profile_utils.py).
RECORD_MEMORY_HISTORY="${RECORD_MEMORY_HISTORY:-0}"
if [[ "${RECORD_MEMORY_HISTORY}" == "1" ]]; then
  MEMORY_SNAPSHOT_DIR="${MEMORY_SNAPSHOT_DIR:-${RUN_ROOT}/mem_snap}"
  MEMORY_SNAPSHOT_PATH="${MEMORY_SNAPSHOT_PATH:-snap.pickle}"
  MEMORY_SNAPSHOT_NUM_STEPS="${MEMORY_SNAPSHOT_NUM_STEPS:-1}"
  mkdir -p "${MEMORY_SNAPSHOT_DIR}"
  MISC_ARGS+=(
    --record-memory-history
    --memory-recorder torch
    --memory-snapshot-dir "${MEMORY_SNAPSHOT_DIR}"
    --memory-snapshot-path "${MEMORY_SNAPSHOT_PATH}"
    --memory-snapshot-num-steps "${MEMORY_SNAPSHOT_NUM_STEPS}"
    --profile-target train_overall
  )
  echo "Memory snapshot: dir=${MEMORY_SNAPSHOT_DIR} path=${MEMORY_SNAPSHOT_PATH} steps=${MEMORY_SNAPSHOT_NUM_STEPS}"
fi
if [[ "${DEBUG_TRAIN_ONLY}" == "1" ]]; then
  echo "DEBUG_TRAIN_ONLY=1 (no SGLang); LOAD_DEBUG_ROLLOUT_DATA=${LOAD_DEBUG_ROLLOUT_DATA:-<unset>}"
fi

# ============ ray cluster network ============
export MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-$(hostname -I | awk '{print $1}')}}"
export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-6379}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"

# ============ SWE / claude-code rollout knobs ============
export SWE_AGENT="${SWE_AGENT:-claude_code}"
export SWE_TRAIN_PROTOCOL="${SWE_TRAIN_PROTOCOL:-scaleswe}"
export E2B_API_KEY="${E2B_API_KEY:-e2b_0000000000000000000000000000000000000000}"
export SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY="${SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY:-image}"
export SLIME_AGENT_E2B_USE_TEMPLATE="${SLIME_AGENT_E2B_USE_TEMPLATE:-0}"
export SLIME_AGENT_NODE_TARBALL="${SLIME_AGENT_NODE_TARBALL:-${SCRIPT_DIR}/tarballs/node-v22.20.0-linux-x64.tar.xz}"
export SLIME_AGENT_CC_TARBALL="${SLIME_AGENT_CC_TARBALL:-${SCRIPT_DIR}/tarballs/anthropic-ai-claude-code-local-linux-x64.tgz}"

export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}}"
export ADAPTER_PUBLIC_URL="${ADAPTER_PUBLIC_URL:-}"
export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
export ADAPTER_PORT="${ADAPTER_PORT:-18001}"

export SWE_AGENT_TIME_BUDGET_SEC="${SWE_AGENT_TIME_BUDGET_SEC:-900}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-300}"
export SWE_BOOT_CONCURRENCY
# Higher = fewer TOKEN_FORK segments (more REALIGN / rewrite-merge); default was 1024.
export SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS="${SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS:-160000}"

SETTINGS_JSON='{"permissions":{"defaultMode":"bypassPermissions"},"autoCompactEnabled":true,"autoCompactWindow":160000}'
AGENTS_JSON='{"investigator":{"description":"Searches the repo for relevant files before any edit","prompt":"You are an investigator sub-agent. Use Grep/Read/Glob to find every file relevant to the user task, then return a short bulleted summary. Do NOT edit anything.","tools":["Grep","Read","Glob"]}}'
export SLIME_AGENT_CC_EXTRA_ARGS="--settings '${SETTINGS_JSON}' --disable-slash-commands --agents '${AGENTS_JSON}' --disallowedTools WebFetch WebSearch"
if [[ -z "${SLIME_AGENT_CC_EXTRA_ENVS:-}" ]]; then
  export SLIME_AGENT_CC_EXTRA_ENVS="{\"CLAUDE_CODE_MAX_OUTPUT_TOKENS\":\"${MAX_GEN_LEN}\"}"
fi

export no_proxy="127.0.0.1,${MASTER_ADDR},${ADAPTER_PUBLIC_HOST}"
export NO_PROXY="${no_proxy}"

cd "${SLIME_DIR}"

# ============ bring up ray cluster (single node) ============
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

echo "Waiting for Ray cluster to stabilize..."
sleep 10
ray status

export SLIME_DIR
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json, os
keys = (
    "no_proxy", "NO_PROXY",
    "SWE_AGENT",
    "E2B_API_KEY", "ADAPTER_PUBLIC_HOST", "ADAPTER_PUBLIC_URL",
    "SLIME_AGENT_NODE_TARBALL", "SLIME_AGENT_CC_TARBALL",
    "SWE_AGENT_TIME_BUDGET_SEC", "SWE_EVAL_TIMEOUT_SEC", "SWE_BOOT_CONCURRENCY",
    "ADAPTER_BIND_HOST", "ADAPTER_PORT",
    "SLIME_AGENT_CC_EXTRA_ARGS",
    "SLIME_AGENT_CC_EXTRA_ENVS",
    "SWE_CC_PROMPT",
    "SWE_TRAIN_PROTOCOL",
    "SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY",
    "SLIME_AGENT_E2B_USE_TEMPLATE",
    "SLIME_AGENT_SANDBOX_BACKEND",
    "SLIME_AGENT_DOCKER_NETWORK",
    "SLIME_AGENT_DOCKER_ADD_HOST",
    "SLIME_AGENT_DOCKER_PULL",
    "SLIME_AGENT_DOCKER_RUN_TIMEOUT_SEC",
    # coding-agent mid-turn offload
    "SLIME_AGENT_OFFLOAD",
    "DASHSCOPE_API_KEY", "OPENAI_API_KEY",
    "DASHSCOPE_BASE_URL", "DASHSCOPE_MODEL",
    "OFFLOAD_EFFICIENCY_LAMBDA", "OFFLOAD_MAX_TOKENS",
    "OFFLOAD_THINK_FORMAT_PENALTY",
    "OFFLOAD_REWARD_MODE", "OFFLOAD_SEEK_ALPHA",
    "OFFLOAD_SEEK_EMPTY_SCALE", "OFFLOAD_UNIQUE_SOLVER_BONUS",
    "OFFLOAD_STOP_TOKEN_ID", "ROLLOUT_STOP_TOKEN_IDS",
    "SLIME_AGENT_OFFLOAD_SYSTEM_APPEND",
    "SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS",
)
env = {k: os.environ[k] for k in keys if k in os.environ}
env["MASTER_ADDR"] = os.environ["MASTER_ADDR"]
env["MASTER_PORT"] = os.environ.get("MASTER_PORT", "")
env["GLOO_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["TP_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["NCCL_SOCKET_IFNAME"] = os.environ["NCCL_SOCKET_IFNAME"]
env["PYTHONPATH"] = f"/root/Megatron-LM/:{os.environ['SLIME_DIR']}"
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
env["NCCL_NVLS_ENABLE"] = "0"
print(json.dumps({"env_vars": env}))
PY
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -u train_async.py \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_GPUS}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${ALGO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   2>&1 | tee "${LOG_FILE}"

echo "RUN_ROOT=${RUN_ROOT}"
echo "SAVE_DIR=${SAVE_DIR}"
