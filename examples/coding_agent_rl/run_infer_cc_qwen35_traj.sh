#!/usr/bin/env bash
# Inference-only: CC → adapter → remote vLLM (Qwen3.5-4B), no local SLM / no train.
#
# Intended for a separate traj pod that dials a vLLM serve started elsewhere, e.g.:
#   vllm serve /workspace/models/Qwen/Qwen3.5-4B \
#     --port 8000 --tensor-parallel-size 1 --max-model-len 262144 \
#     --reasoning-parser qwen3 --enable-auto-tool-choice \
#     --tool-call-parser qwen3_coder
#
# Defaults: resume; skip samples that already have summary.json.
# LLM IP defaults to 127.0.0.1 (port-forward / same-pod). Override when needed:
#   DASHSCOPE_BASE_URL=http://<vllm-pod-ip>:8000/v1
#
# On docker-rt / k8s:
#   * export DOCKER_HOST=unix:///tmp/docker-rt.sock  (auto-detected below)
#   * ADAPTER_BIND_HOST=0.0.0.0 so sandboxes can dial this pod
#   * ADAPTER_PUBLIC_HOST=pod IP (not 127.0.0.1; sandboxes must reach the adapter)
#   * Keep CONCURRENCY low (default 2); docker-rt docker cp often dies under load
#     with: "'NoneType' object has no attribute 'decode'"
#
# Prerequisites:
#   node + claude-code tarballs under examples/coding_agent_rl/tarballs/
#
# Example:
#   bash examples/coding_agent_rl/run_infer_cc_qwen35_traj.sh
#   CONCURRENCY=4 bash examples/coding_agent_rl/run_infer_cc_qwen35_traj.sh --eval
#   DASHSCOPE_BASE_URL=http://10.244.1.23:8000/v1 bash examples/coding_agent_rl/run_infer_cc_qwen35_traj.sh
#   OUT_DIR=/tmp/new_run LIMIT=10 OFFSET=0 bash examples/coding_agent_rl/run_infer_cc_qwen35_traj.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

# ---- docker / docker-rt (must work before any sandbox) ----
# This pod has no /var/run/docker.sock; use docker-rt when present.
if [[ -z "${DOCKER_HOST:-}" && -S /tmp/docker-rt.sock ]]; then
  export DOCKER_HOST=unix:///tmp/docker-rt.sock
fi
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: cannot talk to docker daemon." >&2
  echo "  Tried DOCKER_HOST=${DOCKER_HOST:-<unset>}" >&2
  echo "  On this cluster usually: export DOCKER_HOST=unix:///tmp/docker-rt.sock" >&2
  exit 1
fi

# ---- remote LLM (vLLM OpenAI-compatible) ----
# Default: localhost:8000. Change host when vLLM is on another machine/pod.
export DASHSCOPE_BASE_URL="${DASHSCOPE_BASE_URL:-http://10.244.0.186:8000/v1}"
# vLLM accepts any non-empty key when auth is disabled.
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-EMPTY}"
# Must match vLLM --served-model-name, or the model path if that flag was omitted.
export DASHSCOPE_MODEL="${DASHSCOPE_MODEL:-/workspace/models/Qwen/Qwen3.5-4B}"
# Qwen3 reasoning: keep thinking on; drop effort if the server rejects it.
export INFER_REASONING_EFFORT="${INFER_REASONING_EFFORT:-}"

if [[ -z "${DASHSCOPE_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: set DASHSCOPE_API_KEY (or OPENAI_API_KEY) for remote LLM." >&2
  exit 1
fi

# ---- sandbox ↔ adapter (docker / docker-rt) ----
export SLIME_AGENT_OFFLOAD=0
export SLIME_AGENT_SANDBOX_BACKEND="${SLIME_AGENT_SANDBOX_BACKEND:-docker}"
export SLIME_AGENT_DOCKER_NETWORK="${SLIME_AGENT_DOCKER_NETWORK:-bridge}"
export SLIME_AGENT_DOCKER_ADD_HOST="${SLIME_AGENT_DOCKER_ADD_HOST:-host.docker.internal:host-gateway}"
export SLIME_AGENT_DOCKER_NAME_PREFIX="${SLIME_AGENT_DOCKER_NAME_PREFIX:-cc-qwen35-infer}"

_POD_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-${_POD_IP}}"
if [[ -z "${ADAPTER_PUBLIC_HOST}" || "${ADAPTER_PUBLIC_HOST}" == "127.0.0.1" ]]; then
  echo "ERROR: ADAPTER_PUBLIC_HOST must be a sandbox-routable pod/node IP, not empty/127.0.0.1" >&2
  echo "  Set explicitly, e.g. ADAPTER_PUBLIC_HOST=10.244.2.72" >&2
  exit 1
fi
export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
export ADAPTER_PORT="${ADAPTER_PORT:-18031}"
: "${ADAPTER_PUBLIC_URL:=}"
export ADAPTER_PUBLIC_URL

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/infer_cc_qwen35_4b_${STAMP}}"
JSONL="${PROMPT_DATA:-${SCRIPT_DIR}/data/swe_train_scaleswe_reward1_glm_dsv4flash.jsonl}"
TIME_BUDGET="${SWE_AGENT_TIME_BUDGET_SEC:-${TIME_BUDGET:-900}}"
LIMIT="${LIMIT:-${INFER_LIMIT:-3000}}"
OFFSET="${OFFSET:-${INFER_OFFSET:-0}}"
# docker-rt: high concurrency → flaky docker cp (NoneType.decode). Start low.
CONCURRENCY="${CONCURRENCY:-${INFER_CONCURRENCY:-2}}"

echo "======================================================================"
echo "Infer CC → Qwen3.5-4B via vLLM (no SLM / no train) [resume]"
echo "  OUT_DIR=${OUT_DIR}"
echo "  JSONL=${JSONL}"
echo "  LIMIT=${LIMIT} OFFSET=${OFFSET} CONCURRENCY=${CONCURRENCY}"
echo "  DOCKER_HOST=${DOCKER_HOST:-<default sock>}"
echo "  ADAPTER_BIND_HOST=${ADAPTER_BIND_HOST}"
echo "  ADAPTER_PUBLIC_HOST=${ADAPTER_PUBLIC_HOST}:${ADAPTER_PORT}"
echo "  SLIME_AGENT_DOCKER_NETWORK=${SLIME_AGENT_DOCKER_NETWORK}"
echo "  SLIME_AGENT_DOCKER_NAME_PREFIX=${SLIME_AGENT_DOCKER_NAME_PREFIX}"
echo "  DASHSCOPE_BASE_URL=${DASHSCOPE_BASE_URL}"
echo "  DASHSCOPE_MODEL=${DASHSCOPE_MODEL}"
echo "  INFER_REASONING_EFFORT=${INFER_REASONING_EFFORT:-<unset>}"
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
