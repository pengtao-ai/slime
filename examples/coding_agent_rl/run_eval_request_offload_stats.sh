#!/usr/bin/env bash
# Eval: replay saved CC request dumps on local SGLang, count offload tags.
#
# Each request's system prompt is appended with OFFLOAD_SYSTEM_PROMPT_APPEND
# from examples/coding_agent_rl/offload.py (unless NO_INJECT_OFFLOAD=1).
#
# Prerequisites — start SGLang first:
#   bash examples/coding_agent_rl/launch_sglang_pyrodash4b.sh
#
# Then:
#   bash examples/coding_agent_rl/run_eval_request_offload_stats.sh
#   RUN_DIR=runs/infer_cc_glm_20260728_094722 CONCURRENCY=8 \
#     bash examples/coding_agent_rl/run_eval_request_offload_stats.sh
#
# Outputs under ${RUN_DIR}/sglang_replay_<stamp>/:
#   summary.json                 # offload stats
#   results.jsonl                # per-request row (full request + response)
#   pairs/**/req_XX.json         # {request, response} wire payloads
#   responses/**/req_XX.json     # same + offload counters / parsed fields
#   response_texts/**/req_XX.txt # plain-text model output

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/infer_cc_glm_20260728_094722}"
SGLANG_URL="${SGLANG_URL:-http://127.0.0.1:30000/v1}"
MODEL="${SERVED_MODEL_NAME:-${MODEL:-PyroDash-4B-SFT-0729}}"
# im_end / eos / <|/llm_offload|>
STOP_TOKEN_IDS="${STOP_TOKEN_IDS:-248046,248044,248079}"
CONCURRENCY="${CONCURRENCY:-${REPLAY_CONCURRENCY:-8}}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/sglang_replay_${STAMP}}"

if [[ -x /root/micromamba/envs/slime/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-/root/micromamba/envs/slime/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

EXTRA_ARGS=()
if [[ "${NO_INJECT_OFFLOAD:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-inject-offload)
fi
if [[ -n "${LIMIT:-}" ]]; then
  EXTRA_ARGS+=(--limit "${LIMIT}")
fi

echo "======================================================================"
echo "Eval request offload stats (SGLang replay)"
echo "  RUN_DIR=${RUN_DIR}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  SGLANG_URL=${SGLANG_URL}"
echo "  MODEL=${MODEL}"
echo "  STOP_TOKEN_IDS=${STOP_TOKEN_IDS}"
echo "  CONCURRENCY=${CONCURRENCY} MAX_TOKENS=${MAX_TOKENS}"
echo "  inject_offload_system=$([ "${NO_INJECT_OFFLOAD:-0}" = 1 ] && echo false || echo true)"
echo "  no_stop_trim=true (keeps <|/llm_offload|> in output)"
echo "  OFFLOAD_SYSTEM_PROMPT_APPEND=offload.OFFLOAD_SYSTEM_PROMPT_APPEND"
echo "======================================================================"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/replay_requests_sglang_offload_stats.py" \
  --run-dir "${RUN_DIR}" \
  --out-dir "${OUT_DIR}" \
  --url "${SGLANG_URL}" \
  --model "${MODEL}" \
  --stop-token-ids "${STOP_TOKEN_IDS}" \
  --concurrency "${CONCURRENCY}" \
  --max-tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
