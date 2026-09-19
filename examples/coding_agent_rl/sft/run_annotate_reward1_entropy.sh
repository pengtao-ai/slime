#!/usr/bin/env bash
# Annotate reward=1 CC trajectories with PyroDash entropy.
#
# Per sample: take the last requests/req_*.json, expand full chat
# (messages + response), then for EACH assistant call vLLM with the prefix
# before that assistant and write entropy onto it.
#
# Prerequisites:
#   bash examples/coding_agent_rl/sft/launch_vllm_pyrodash4b_sft0902.sh
#
# Usage:
#   bash examples/coding_agent_rl/sft/run_annotate_reward1_entropy.sh
#   DRY_RUN=1 bash examples/coding_agent_rl/sft/run_annotate_reward1_entropy.sh
#   RUN_DIRS=.../infer_cc_dsv4flash_20260806_141118 LIMIT_SAMPLES=2 \
#     bash examples/coding_agent_rl/sft/run_annotate_reward1_entropy.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

VLLM_URL="${VLLM_URL:-http://127.0.0.1:9016/v1}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_MODEL="${VLLM_MODEL:-/workspace/models/pyromind/PyroDash-4B-SFT-0918}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/trajectories_entropy_0918}"
TOP_LOGPROBS="${TOP_LOGPROBS:-20}"
ENTROPY_SCOPE="${ENTROPY_SCOPE:-thinking}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
SAMPLE_CONCURRENCY="${SAMPLE_CONCURRENCY:-8}"
ASSISTANT_CONCURRENCY="${ASSISTANT_CONCURRENCY:-8}"
TIMEOUT="${TIMEOUT:-600}"

DEFAULT_RUNS=(
  "${REPO_ROOT}/runs/infer_cc_glm_20260728_141118"
  "${REPO_ROOT}/runs/infer_cc_dsv4flash_20260806_141118"
  "${REPO_ROOT}/runs/infer_cc_tmax_20260807_162106"
)

if [[ -x /root/micromamba/envs/slime/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-/root/micromamba/envs/slime/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

EXTRA_ARGS=()
if [[ -n "${RUN_DIRS:-}" ]]; then
  # shellcheck disable=SC2206
  RUN_ARR=(${RUN_DIRS})
else
  RUN_ARR=("${DEFAULT_RUNS[@]}")
fi
for rd in "${RUN_ARR[@]}"; do
  EXTRA_ARGS+=(--run-dir "${rd}")
done

[[ -n "${LIMIT_SAMPLES:-}" ]] && EXTRA_ARGS+=(--limit-samples "${LIMIT_SAMPLES}")
[[ -n "${OFFSET_SAMPLES:-}" ]] && EXTRA_ARGS+=(--offset-samples "${OFFSET_SAMPLES}")
[[ "${DRY_RUN:-0}" == "1" ]] && EXTRA_ARGS+=(--dry-run)
[[ "${NO_TOOLS:-0}" == "1" ]] && EXTRA_ARGS+=(--no-tools)
[[ "${NO_THINKING:-0}" == "1" ]] && EXTRA_ARGS+=(--no-thinking)
[[ "${NO_RESUME:-0}" == "1" ]] && EXTRA_ARGS+=(--no-resume)
# Optional stop only when explicitly set (default: none; full Qwen response).
if [[ -n "${STOP_STR:-}" ]]; then
  EXTRA_ARGS+=(--stop "${STOP_STR}")
fi

echo "======================================================================"
echo "Annotate reward=1: last req, entropy on every assistant"
echo "  VLLM_URL=${VLLM_URL}"
echo "  VLLM_MODEL=${VLLM_MODEL}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  SAMPLE_CONCURRENCY=${SAMPLE_CONCURRENCY}"
echo "  ASSISTANT_CONCURRENCY=${ASSISTANT_CONCURRENCY}"
echo "  STOP_STR=${STOP_STR:-} (default: none, full response)"
echo "  mode=last_req_all_assistants"
echo "  runs:"
for rd in "${RUN_ARR[@]}"; do
  echo "    - ${rd}"
done
echo "======================================================================"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/annotate_reward1_entropy.py" \
  --out-dir "${OUT_DIR}" \
  --url "${VLLM_URL}" \
  --api-key "${VLLM_API_KEY}" \
  --model "${VLLM_MODEL}" \
  --top-logprobs "${TOP_LOGPROBS}" \
  --entropy-scope "${ENTROPY_SCOPE}" \
  --max-tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  --concurrency "${SAMPLE_CONCURRENCY}" \
  --assistant-concurrency "${ASSISTANT_CONCURRENCY}" \
  --timeout "${TIMEOUT}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
