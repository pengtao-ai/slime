#!/usr/bin/env bash
# PyroDash-4B LLM-offload GRPO on 1 node (8 GPUs), slime + SGLang.
#
# Reward: examples/llm_offload/reward.py (cost-aware math + remote GLM handoff
# after <|llm_offload|>). Adapted from pyroDash-training phase2/reward.py.
#
# Prerequisites:
#   1. HF + torch_dist for PyroDash-4B
#        bash examples/coding_agent_rl/convert_pyrodash4b_to_torch_dist.sh
#   2. Convert dataset once:
#        python examples/llm_offload/convert_offload_dataset.py \
#          --src /path/to/glm52_hint_8b_answers.jsonl \
#          --dst examples/llm_offload/data/offload_grpo_train.jsonl
#   3. Remote grader / GLM OpenAI-compatible API:
#        export DASHSCOPE_API_KEY=...
#        export DASHSCOPE_BASE_URL=http://host:8000/v1
#        export DASHSCOPE_MODEL=deepseek-v4-flash-0731
#        export OFFLOAD_EFFICIENCY_LAMBDA=0.6
#
# Run from a long-lived shell / tmux:
#   bash examples/llm_offload/run_pyrodash4b_offload_1node.sh

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

# ============ parallelism (dense 4B, 8 GPUs, colocate) ============
export TP_SIZE="${TP_SIZE:-2}"
export PP_SIZE="${PP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-4}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
ROLLOUT_MEM_UTILIZATION="${ROLLOUT_MEM_UTILIZATION:-0.7}"

MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-16384}"
MAX_GEN_LEN="${MAX_GEN_LEN:-8192}"

# <|llm_offload|> token id in PyroDash-4B-SFT-0723 / Qwen3.5-4B-with-offload-token
OFFLOAD_STOP_TOKEN_ID="${OFFLOAD_STOP_TOKEN_ID:-248077}"

HF_CHECKPOINT="${HF_CHECKPOINT:-/workspace/models/pyromind/PyroDash-4B-SFT-0723}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/workspace/models/pyromind/PyroDash-4B-SFT-0723_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/data/offload_grpo_train.jsonl}"

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
NUM_ROLLOUT="${NUM_ROLLOUT:-500}"

EXP_TAG="${EXP_TAG:-pyrodash4b_offload_grpo}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${SLIME_DIR}/runs/${EXP_TAG}_${STAMP}}"
mkdir -p "${RUN_ROOT}/rollout_dumps"
LOG_FILE="${RUN_ROOT}/run.log"

# Remote offload grader (consumed by examples.llm_offload.reward)
export DASHSCOPE_BASE_URL="${DASHSCOPE_BASE_URL:-http://208.64.254.187:8001/v1}"
export DASHSCOPE_MODEL="${DASHSCOPE_MODEL:-deepseek-v4-flash-0731}"
export OFFLOAD_EFFICIENCY_LAMBDA="${OFFLOAD_EFFICIENCY_LAMBDA:-0.6}"
export OFFLOAD_MAX_TOKENS="${OFFLOAD_MAX_TOKENS:-8192}"
export OFFLOAD_MAX_WORKERS="${OFFLOAD_MAX_WORKERS:-32}"
: "${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY for the remote GLM / OpenAI-compatible API}"

echo "======================================================================"
echo "PyroDash-4B offload GRPO"
echo "  HF_CHECKPOINT=${HF_CHECKPOINT}"
echo "  REF_MODEL_PATH=${REF_MODEL_PATH}"
echo "  PROMPT_DATA=${PROMPT_DATA}"
echo "  RUN_ROOT=${RUN_ROOT}"
echo "  DASHSCOPE_BASE_URL=${DASHSCOPE_BASE_URL}"
echo "  DASHSCOPE_MODEL=${DASHSCOPE_MODEL}"
echo "  OFFLOAD_EFFICIENCY_LAMBDA=${OFFLOAD_EFFICIENCY_LAMBDA}"
echo "  stop_token_id=${OFFLOAD_STOP_TOKEN_ID}"
echo "  log: ${LOG_FILE}"
echo "======================================================================"

if [[ ! -f "${PROMPT_DATA}" ]]; then
  echo "ERROR: prompt data missing: ${PROMPT_DATA}" >&2
  echo "Convert with:" >&2
  echo "  python examples/llm_offload/convert_offload_dataset.py \\" >&2
  echo "    --src /workspace/work/spt/pyroDash-training/data/phase2/glm52_hint_8b_answers.jsonl \\" >&2
  echo "    --dst ${PROMPT_DATA}" >&2
  exit 1
fi
if [[ ! -d "${REF_MODEL_PATH}" ]]; then
  echo "ERROR: torch_dist missing: ${REF_MODEL_PATH}" >&2
  echo "Run: bash examples/coding_agent_rl/convert_pyrodash4b_to_torch_dist.sh" >&2
  exit 1
fi

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_MODEL_PATH}"
   --save "${RUN_ROOT}/checkpoints"
   --save-interval "${SAVE_INTERVAL:-20}"
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --rollout-shuffle
   --group-rm
   --custom-rm-path examples.llm_offload.reward.reward_func
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-context-len "${MAX_CONTEXT_LEN}"
   --rollout-max-response-len "${MAX_GEN_LEN}"
   --rollout-temperature 1.0
   # Stop generation when the model emits <|llm_offload|> (include stop token).
   --rollout-stop "<|llm_offload|>"
   --rollout-stop-token-ids "${OFFLOAD_STOP_TOKEN_ID}"
   --num-steps-per-rollout 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --micro-batch-size 1
   --balance-data
   --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size "${PP_SIZE}"
   --context-parallel-size "${CP_SIZE}"
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-$((MAX_CONTEXT_LEN / CP_SIZE))}"
)

GRPO_ARGS=(
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
   --lr "${LR:-1e-6}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_TP_SIZE}"
   --sglang-mem-fraction-static "${ROLLOUT_MEM_UTILIZATION}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi

export MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
export PYTHONUNBUFFERED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

NUM_GPUS="${NUM_GPUS:-8}"
cd "${SLIME_DIR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "PYTHONPATH": "/root/Megatron-LM/:${SLIME_DIR}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "DASHSCOPE_API_KEY": "${DASHSCOPE_API_KEY}",
    "DASHSCOPE_BASE_URL": "${DASHSCOPE_BASE_URL}",
    "DASHSCOPE_MODEL": "${DASHSCOPE_MODEL}",
    "OFFLOAD_EFFICIENCY_LAMBDA": "${OFFLOAD_EFFICIENCY_LAMBDA}",
    "OFFLOAD_MAX_TOKENS": "${OFFLOAD_MAX_TOKENS}",
    "OFFLOAD_MAX_WORKERS": "${OFFLOAD_MAX_WORKERS}"
  }
}
EOF
)

# shellcheck disable=SC2086
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -u train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NUM_GPUS}" \
   --colocate \
   ${MODEL_ARGS[@]} \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   2>&1 | tee "${LOG_FILE}"
