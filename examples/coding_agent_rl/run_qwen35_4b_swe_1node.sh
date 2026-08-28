#!/usr/bin/env bash
# End-to-end SWE coding-agent RL on 1 node (8 GPUs) with Qwen3.5-4B.
# See README.md for the dataset schema, env vars, and fan-out semantics.
# Run from a long-lived shell / tmux session (a short-lived nohup launcher
# gets its Ray child processes cleaned up with it).

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

# ============ model parallelism (dense 4B, 8 GPUs) ============
export TP_SIZE="${TP_SIZE:-2}"
export PP_SIZE="${PP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-4}"

# ============ rollout engine ============
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
ROLLOUT_MEM_UTILIZATION="${ROLLOUT_MEM_UTILIZATION:-0.7}"

# ============ context length ============
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-96000}"
MAX_GEN_LEN="${MAX_GEN_LEN:-32768}"

# ============ paths — override before launching ============
HF_CHECKPOINT="${HF_CHECKPOINT:-/workspace/models/Qwen/Qwen3.5-4B}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/workspace/models/Qwen/Qwen3.5-4B_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-/workspace/work/spt/slime/examples/coding_agent_rl/data/swe_train_scaleswe_200.jsonl}"

# Public e2b.dev caps concurrent sandboxes (often 20). Template mode defaults to a
# tiny fan-out so agent (+ optional eval) sandboxes stay under that limit.
# Override any of these before launch if you have a higher quota.
export SLIME_AGENT_E2B_USE_TEMPLATE="${SLIME_AGENT_E2B_USE_TEMPLATE:-0}"
if [[ "${SLIME_AGENT_E2B_USE_TEMPLATE}" == "1" ]]; then
  ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
  N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
  SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-4}"
else
  ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
  N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
  SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-8}"
fi

EXP_TAG="${EXP_TAG:-agent_only_qwen35_4b}"
STAMP="$(date +%Y%m%d_%H%M%S)"
export RUN_ROOT="${RUN_ROOT:-${SLIME_DIR}/runs/${EXP_TAG}_${STAMP}}"
# eos/pad; offload launchers append <|/llm_offload|> id (e.g. 248078).
ROLLOUT_STOP_TOKEN_IDS="${ROLLOUT_STOP_TOKEN_IDS:-248046 248044}"

# ============ logging ============
LOG_DIR="${RUN_ROOT}"
mkdir -p "${LOG_DIR}/rollout_dumps"
LOG_FILE="${LOG_DIR}/run.log"
echo "======================================================================"
echo "Training log: ${LOG_FILE}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS=${SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS:-8192}"
echo "======================================================================"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_MODEL_PATH}"
)

ROLLOUT_ARGS=(
   --custom-generate-function-path examples.coding_agent_rl.generate.generate
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --num-rollout ${NUM_ROLLOUT:-2}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
   --rollout-max-context-len ${MAX_CONTEXT_LEN}
   --rollout-max-response-len ${MAX_GEN_LEN}
   --rollout-temperature 1.0
   --rollout-stop-token-ids ${ROLLOUT_STOP_TOKEN_IDS}
   --num-steps-per-rollout 1
   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --micro-batch-size 1
   --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
)
ROLLOUT_ARGS+=(--rollout-sample-filter-path examples.coding_agent_rl.offload.compact_and_shape_group_help_seeking_rewards)

PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --sequence-parallel
   --pipeline-model-parallel-size ${PP_SIZE}
   --context-parallel-size ${CP_SIZE}
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   # max-tokens-per-gpu is one CP rank's slice of MAX_CONTEXT_LEN; log-probs are
   # chunked along T to avoid OOM on long single trajectories.
   --max-tokens-per-gpu $((MAX_CONTEXT_LEN / CP_SIZE))
   --log-probs-chunk-size 1024
   --use-dynamic-batch-size
)

# Turn advantages: GiGPO (ScaleSWE/Tmax segment split + intent/tool step groups).
# Episode scalar reward stays help_seeking / cost-aware; cost turn-residual painting
# (offload_turn_advantage) is retired.
ALGO_ARGS=(
   --advantage-estimator grpo
   --custom-advantage-function-path examples.coding_agent_rl.gigpo.compute_advantages
   --custom-reward-post-process-path examples.coding_agent_rl.gigpo.post_process_rewards
   --gigpo-gamma "${GIGPO_GAMMA:-0.95}"
   --gigpo-step-advantage-w "${GIGPO_STEP_ADVANTAGE_W:-1.0}"
   --loss-type custom_loss
   --custom-loss-function-path examples.coding_agent_rl.grpo_sft_loss.grpo_sft_loss_function
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
   --rollout-num-gpus 8
   --rollout-num-gpus-per-engine ${ROLLOUT_TP_SIZE}
   --sglang-mem-fraction-static ${ROLLOUT_MEM_UTILIZATION}
   --sglang-tool-call-parser qwen3_coder
   --sglang-reasoning-parser qwen3
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --colocate
)

# ============ ray cluster network ============
# Set MASTER_ADDR before the SWE block: ADAPTER_PUBLIC_HOST below falls back to it.
export MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-$(hostname -I | awk '{print $1}')}}"
export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-6379}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"

# ============ SWE / claude-code rollout knobs ============

export SWE_AGENT="${SWE_AGENT:-claude_code}"
export SWE_TRAIN_PROTOCOL="${SWE_TRAIN_PROTOCOL:-scaleswe}"
export E2B_API_KEY="${E2B_API_KEY:-e2b_0000000000000000000000000000000000000000}"
# Metadata key your gateway routes images by; `image` is the neutral default.
export SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY="${SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY:-image}"
# Public e2b.dev: set to 1 and put an E2B *template* name in metadata.image
# (see examples/coding_agent_rl/smoke_public_e2b.py). Internal gateways leave this unset.
# SLIME_AGENT_E2B_USE_TEMPLATE is also read above to pick rollout fan-out defaults.
export SLIME_AGENT_E2B_USE_TEMPLATE="${SLIME_AGENT_E2B_USE_TEMPLATE:-0}"
export SLIME_AGENT_NODE_TARBALL="${SLIME_AGENT_NODE_TARBALL:-${SCRIPT_DIR}/tarballs/node-v22.20.0-linux-x64.tar.xz}"
export SLIME_AGENT_CC_TARBALL="${SLIME_AGENT_CC_TARBALL:-${SCRIPT_DIR}/tarballs/anthropic-ai-claude-code-local-linux-x64.tgz}"

# ADAPTER_PUBLIC_HOST must be routable from inside the sandbox (not 127.0.0.1).
# Docker/docker-rt on k8s: prefer the pod IP (see run_qwen35_4b_swe_1node_docker.sh).
# For public e2b.dev use a reverse proxy instead, e.g.:
#   bash examples/coding_agent_rl/start_adapter_tunnel.sh
#   export ADAPTER_PUBLIC_URL=https://....trycloudflare.com
export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}}"
export ADAPTER_PUBLIC_URL="${ADAPTER_PUBLIC_URL:-}"
export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
export ADAPTER_PORT="${ADAPTER_PORT:-18001}"

export SWE_AGENT_TIME_BUDGET_SEC="${SWE_AGENT_TIME_BUDGET_SEC:-1800}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-600}"
export SWE_BOOT_CONCURRENCY
export ADAPTER_MAX_TURNS_PER_SID="${ADAPTER_MAX_TURNS_PER_SID:-128}"
# Higher = fewer TOKEN_FORK segments (more REALIGN / rewrite-merge); default was 1024.
export SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS="${SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS:-8192}"

# autoCompactWindow (80k) < MAX_CONTEXT_LEN (96k) so the CLI compacts before any
# segment crosses the training-side cap. `investigator` is a read-only sub-agent
# (a concrete dispatch target). WebFetch/WebSearch off (no outbound internet).
SETTINGS_JSON='{"permissions":{"defaultMode":"bypassPermissions"},"autoCompactEnabled":true,"autoCompactWindow":80000}'
AGENTS_JSON='{"investigator":{"description":"Searches the repo for relevant files before any edit","prompt":"You are an investigator sub-agent. Use Grep/Read/Glob to find every file relevant to the user task, then return a short bulleted summary. Do NOT edit anything.","tools":["Grep","Read","Glob"]}}'
export SLIME_AGENT_CC_EXTRA_ARGS="--settings '${SETTINGS_JSON}' --disable-slash-commands --agents '${AGENTS_JSON}' --disallowedTools WebFetch WebSearch"
# Cap Claude Code's client-side max output (default 32k). Align with MAX_GEN_LEN.
if [[ -z "${SLIME_AGENT_CC_EXTRA_ENVS:-}" ]]; then
  export SLIME_AGENT_CC_EXTRA_ENVS="{\"CLAUDE_CODE_MAX_OUTPUT_TOKENS\":\"${MAX_GEN_LEN}\"}"
fi

# Optional: require dispatching the investigator before any edit, to maximize sub-agent fan-out.
# export SWE_CC_PROMPT="Read PROBLEM_STATEMENT.md. BEFORE editing any file, dispatch the 'investigator' sub-agent (via the Agent tool with subagent_type=investigator) to locate every file relevant to the issue. Then fix the issue and run the tests."

# ============ proxy bypass for in-cluster traffic ============
export no_proxy="127.0.0.1,${MASTER_ADDR},${ADAPTER_PUBLIC_HOST}"
export NO_PROXY="${no_proxy}"

cd "${SLIME_DIR}"

# ============ bring up ray cluster (single node) ============
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

echo "Waiting for Ray cluster to stabilize..."
sleep 10
ray status

# ============ runtime env propagated to ray workers ============
export SLIME_DIR
# Quoted heredoc: unquoted <<PY + set -x expands $CONDA_PREFIX in comments and
# can corrupt the python snippet (seen as `=: command not found`).
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json, os
keys = (
    "no_proxy", "NO_PROXY",
    "SWE_AGENT",
    "E2B_API_KEY", "ADAPTER_PUBLIC_HOST", "ADAPTER_PUBLIC_URL",
    "SLIME_AGENT_NODE_TARBALL", "SLIME_AGENT_CC_TARBALL",
    "SWE_AGENT_TIME_BUDGET_SEC", "SWE_EVAL_TIMEOUT_SEC", "SWE_BOOT_CONCURRENCY",
    "ADAPTER_BIND_HOST", "ADAPTER_PORT",
    "ADAPTER_MAX_TURNS_PER_SID",
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
    "OFFLOAD_THINK_FORMAT_PENALTY", "OFFLOAD_MALFORMED_PENALTY",
    "OFFLOAD_REWARD_MODE", "OFFLOAD_SEEK_ALPHA",
    "OFFLOAD_SEEK_EMPTY_SCALE", "OFFLOAD_UNIQUE_SOLVER_BONUS",
    "OFFLOAD_SEEK_ONLY_WHEN_ALL_WRONG",
    "OFFLOAD_NO_SEEK_PENALTY",
    "OFFLOAD_COMPACT_ORPHAN_OPEN_K", "OFFLOAD_COMPACT_OPEN_CLOSE_RATIO",
    "OFFLOAD_COMPACT_SPECIAL_TOKEN_RATIO", "OFFLOAD_COMPACT_SPECIAL_TOKEN_RUN",
    "OFFLOAD_MALFORMED_PENALTY_CAP", "OFFLOAD_MALFORMED_OPEN_RUN",
    "OFFLOAD_TRUNCATE_OPEN_RUN", "OFFLOAD_TRUNCATE_ORPHAN",
    "OFFLOAD_OPEN_TOKEN_ID", "OFFLOAD_CLOSE_TOKEN_ID",
    "OFFLOAD_STOP_TOKEN_ID", "ROLLOUT_STOP_TOKEN_IDS",
    "SLIME_AGENT_OFFLOAD_SYSTEM_APPEND",
    "SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS",
    "OFFLOAD_SFT_LAMBDA",
    "OFFLOAD_SFT_MAX_SAMPLES",
    "OFFLOAD_SFT_MAX_SEQ_LEN",
    "SLIME_OFFLOAD_EMBED_IN_TRAJECTORY",
    "SLIME_OFFLOAD_EMBED_MAX_TOKENS",
    "RUN_ROOT",
    "OFFLOAD_CONFIG_JSON",
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
# Do NOT prepend $CONDA_PREFIX/lib to LD_LIBRARY_PATH here: conda ships
# cudnn 9.10 while pip nvidia-cudnn-cu12 is 9.16; putting conda first makes
# Transformer Engine mix libcudnn_cnn/graph and fail with undefined symbol
# SdpaForwardOperation. libnuma for sgl_kernel resolves via conda python RPATH.
print(json.dumps({"env_vars": env}))
PY
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -u train.py \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
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
