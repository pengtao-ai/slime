#!/usr/bin/env bash
# Inference-only Tmax: CC → adapter → remote LLM, optional inplace eval.
#
# Defaults: tmax_smoke_3.jsonl, limit=3, concurrency=3, --eval enabled.
#
# Prerequisites:
#   export DASHSCOPE_API_KEY=...   # or rely on defaults below
#   export DASHSCOPE_BASE_URL=http://host:8001/v1
#   export DASHSCOPE_MODEL=deepseek-v4-flash-0731   # optional
#   node + claude-code tarballs under examples/coding_agent_rl/tarballs/
#
# Example:
#   bash examples/coding_agent_rl/run_infer_cc_tmax_traj.sh
#   CONCURRENCY=1 TIME_BUDGET=600 bash examples/coding_agent_rl/run_infer_cc_tmax_traj.sh
#   bash examples/coding_agent_rl/run_infer_cc_tmax_traj.sh --no-thinking

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

# ---- remote LLM ----
export DASHSCOPE_BASE_URL="${DASHSCOPE_BASE_URL:-http://208.64.254.187:8001/v1}"
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-sk-6137d26281697017ef07ef4da0823dc16d32acaad253ecac}"
export DASHSCOPE_MODEL="${DASHSCOPE_MODEL:-deepseek-v4-flash-0731}"
export INFER_REASONING_EFFORT="${INFER_REASONING_EFFORT:-high}"

if [[ -z "${DASHSCOPE_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: set DASHSCOPE_API_KEY (or OPENAI_API_KEY) for remote LLM." >&2
  exit 1
fi

# ---- sandbox ↔ adapter ----
export SLIME_AGENT_OFFLOAD=0
export SLIME_AGENT_SANDBOX_BACKEND="${SLIME_AGENT_SANDBOX_BACKEND:-docker}"
export SLIME_AGENT_DOCKER_NETWORK="${SLIME_AGENT_DOCKER_NETWORK:-bridge}"
export SLIME_AGENT_DOCKER_ADD_HOST="${SLIME_AGENT_DOCKER_ADD_HOST:-host.docker.internal:host-gateway}"
export SLIME_AGENT_DOCKER_NAME_PREFIX="${SLIME_AGENT_DOCKER_NAME_PREFIX:-cc-tmax-infer}"

_POD_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-${_POD_IP}}"
if [[ -z "${ADAPTER_PUBLIC_HOST}" || "${ADAPTER_PUBLIC_HOST}" == "127.0.0.1" ]]; then
  echo "ERROR: ADAPTER_PUBLIC_HOST must be a sandbox-routable pod/node IP, not empty/127.0.0.1" >&2
  echo "  Set explicitly, e.g. ADAPTER_PUBLIC_HOST=10.244.2.72" >&2
  exit 1
fi
export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
export ADAPTER_PORT="${ADAPTER_PORT:-18021}"
: "${ADAPTER_PUBLIC_URL:=}"
export ADAPTER_PUBLIC_URL

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/infer_cc_tmax_${STAMP}}"
JSONL="${PROMPT_DATA:-${SCRIPT_DIR}/data/tmax_train_full.jsonl}"
TIME_BUDGET="${SWE_AGENT_TIME_BUDGET_SEC:-${TIME_BUDGET:-900}}"
LIMIT="${LIMIT:-${INFER_LIMIT:-15000}}"
OFFSET="${OFFSET:-${INFER_OFFSET:-0}}"
CONCURRENCY="${CONCURRENCY:-${INFER_CONCURRENCY:-8}}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-600}"

# Enable eval unless caller already passed --eval / --no-eval style args.
EXTRA_ARGS=("$@")
HAS_EVAL=0
for a in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
  if [[ "$a" == "--eval" ]]; then
    HAS_EVAL=1
    break
  fi
done
if [[ "${HAS_EVAL}" -eq 0 ]]; then
  EXTRA_ARGS+=(--eval)
fi

echo "======================================================================"
echo "Infer CC → remote LLM (tmax + inplace eval)"
echo "  OUT_DIR=${OUT_DIR}"
echo "  JSONL=${JSONL}"
echo "  LIMIT=${LIMIT} OFFSET=${OFFSET} CONCURRENCY=${CONCURRENCY}"
echo "  ADAPTER_BIND_HOST=${ADAPTER_BIND_HOST}"
echo "  ADAPTER_PUBLIC_HOST=${ADAPTER_PUBLIC_HOST}:${ADAPTER_PORT}"
echo "  SLIME_AGENT_DOCKER_NETWORK=${SLIME_AGENT_DOCKER_NETWORK}"
echo "  DASHSCOPE_BASE_URL=${DASHSCOPE_BASE_URL}"
echo "  DASHSCOPE_MODEL=${DASHSCOPE_MODEL}"
echo "  INFER_REASONING_EFFORT=${INFER_REASONING_EFFORT}"
echo "  TIME_BUDGET=${TIME_BUDGET} EVAL_TIMEOUT=${EVAL_TIMEOUT}"
echo "  EXTRA_ARGS=${EXTRA_ARGS[*]}"
echo "======================================================================"

exec python "${SCRIPT_DIR}/infer_cc_tmax_traj.py" \
  --out-dir "${OUT_DIR}" \
  --jsonl "${JSONL}" \
  --time-budget "${TIME_BUDGET}" \
  --eval-timeout "${EVAL_TIMEOUT}" \
  --limit "${LIMIT}" \
  --offset "${OFFSET}" \
  --concurrency "${CONCURRENCY}" \
  --bind-host "${ADAPTER_BIND_HOST}" \
  --bind-port "${ADAPTER_PORT}" \
  --public-host "${ADAPTER_PUBLIC_HOST}" \
  --network "${SLIME_AGENT_DOCKER_NETWORK}" \
  --force \
  "${EXTRA_ARGS[@]}"
