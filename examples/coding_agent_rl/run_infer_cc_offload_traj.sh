#!/usr/bin/env bash
# Inference-only: CC → adapter → remote LLM only (no local SLM, no offload relay).
#
# Defaults: resume into the existing run dir; skip samples with summary.json.
#
# On docker-rt / k8s (same lessons as run_qwen35_4b_swe_1node_docker.sh):
#   * ADAPTER_BIND_HOST=0.0.0.0 so sandboxes can dial this pod
#   * ADAPTER_PUBLIC_HOST=pod IP (host.docker.internal / --add-host often useless
#     under docker-rt; probe prefers --public-host)
#   * Keep CONCURRENCY modest; one flaky probe should not need to kill the batch
#     (probe failures are now per-sample RuntimeError + retries)
#
# Prerequisites:
#   export DASHSCOPE_API_KEY=...   # or rely on defaults below
#   export DASHSCOPE_BASE_URL=http://host:8001/v1
#   export DASHSCOPE_MODEL=deepseek-v4-flash-0731   # optional
#   node + claude-code tarballs under examples/coding_agent_rl/tarballs/
#
# Example:
#   bash examples/coding_agent_rl/run_infer_cc_offload_traj.sh
#   CONCURRENCY=4 bash examples/coding_agent_rl/run_infer_cc_offload_traj.sh --eval
#   OUT_DIR=/tmp/new_run bash examples/coding_agent_rl/run_infer_cc_offload_traj.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

# ---- remote LLM (deepseek-v4-flash; was GLM) ----
export DASHSCOPE_BASE_URL="${DASHSCOPE_BASE_URL:-http://208.64.254.187:8001/v1}"
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-sk-6137d26281697017ef07ef4da0823dc16d32acaad253ecac}"
export DASHSCOPE_MODEL="${DASHSCOPE_MODEL:-deepseek-v4-flash-0731}"
# Infer default: thinking on + max effort (override with --no-thinking / --reasoning-effort).
export INFER_REASONING_EFFORT="${INFER_REASONING_EFFORT:-high}"

if [[ -z "${DASHSCOPE_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: set DASHSCOPE_API_KEY (or OPENAI_API_KEY) for remote LLM." >&2
  exit 1
fi

# ---- sandbox ↔ adapter (docker / docker-rt) ----
export SLIME_AGENT_OFFLOAD=0
export SLIME_AGENT_SANDBOX_BACKEND="${SLIME_AGENT_SANDBOX_BACKEND:-docker}"
export SLIME_AGENT_DOCKER_NETWORK="${SLIME_AGENT_DOCKER_NETWORK:-bridge}"
# Harmless on docker-rt (often ignored); useful on real Docker Desktop/Engine.
export SLIME_AGENT_DOCKER_ADD_HOST="${SLIME_AGENT_DOCKER_ADD_HOST:-host.docker.internal:host-gateway}"
export SLIME_AGENT_DOCKER_NAME_PREFIX="${SLIME_AGENT_DOCKER_NAME_PREFIX:-cc-inference}"

_POD_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-${_POD_IP}}"
if [[ -z "${ADAPTER_PUBLIC_HOST}" || "${ADAPTER_PUBLIC_HOST}" == "127.0.0.1" ]]; then
  echo "ERROR: ADAPTER_PUBLIC_HOST must be a sandbox-routable pod/node IP, not empty/127.0.0.1" >&2
  echo "  Set explicitly, e.g. ADAPTER_PUBLIC_HOST=10.244.2.72" >&2
  exit 1
fi
export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
export ADAPTER_PORT="${ADAPTER_PORT:-18011}"
# Clear leftover Cloudflare URL unless user intentionally set one.
: "${ADAPTER_PUBLIC_URL:=}"
export ADAPTER_PUBLIC_URL

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/infer_cc_dsv4flash_20260806_141118}"
JSONL="${PROMPT_DATA:-${SCRIPT_DIR}/data/swe_train_scaleswe.jsonl}"
TIME_BUDGET="${SWE_AGENT_TIME_BUDGET_SEC:-${TIME_BUDGET:-900}}"
LIMIT="${LIMIT:-${INFER_LIMIT:-20000}}"
OFFSET="${OFFSET:-${INFER_OFFSET:-15000}}"
# docker-rt: keep concurrent sandboxes modest; override upward if stable.
CONCURRENCY="${CONCURRENCY:-${INFER_CONCURRENCY:-8}}"

echo "======================================================================"
echo "Infer CC → deepseek only (no SLM / no train) [resume]"
echo "  OUT_DIR=${OUT_DIR}"
echo "  JSONL=${JSONL}"
echo "  LIMIT=${LIMIT} OFFSET=${OFFSET} CONCURRENCY=${CONCURRENCY}"
echo "  ADAPTER_BIND_HOST=${ADAPTER_BIND_HOST}"
echo "  ADAPTER_PUBLIC_HOST=${ADAPTER_PUBLIC_HOST}:${ADAPTER_PORT}"
echo "  SLIME_AGENT_DOCKER_NETWORK=${SLIME_AGENT_DOCKER_NETWORK}"
echo "  SLIME_AGENT_DOCKER_NAME_PREFIX=${SLIME_AGENT_DOCKER_NAME_PREFIX}"
echo "  DASHSCOPE_BASE_URL=${DASHSCOPE_BASE_URL}"
echo "  DASHSCOPE_MODEL=${DASHSCOPE_MODEL}"
echo "  INFER_REASONING_EFFORT=${INFER_REASONING_EFFORT}"
echo "  TIME_BUDGET=${TIME_BUDGET}"
echo "======================================================================"

exec python "${SCRIPT_DIR}/infer_cc_offload_traj.py" \
  --out-dir "${OUT_DIR}" \
  --jsonl "${JSONL}" \
  --time-budget "${TIME_BUDGET}" \
  --limit "${LIMIT}" \
  --offset "${OFFSET}" \
  --concurrency "${CONCURRENCY}" \
  --bind-host "${ADAPTER_BIND_HOST}" \
  --bind-port "${ADAPTER_PORT}" \
  --public-host "${ADAPTER_PUBLIC_HOST}" \
  --network "${SLIME_AGENT_DOCKER_NETWORK}" \
  --resume \
  "$@"
