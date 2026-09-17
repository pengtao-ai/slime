#!/usr/bin/env bash
# SLM-only agent trajectories: PyroDash-4B-SFT-0902 + per-turn think entropy.
# No offload / no remote GLM teacher — every turn goes straight to local vLLM.
#
# Prerequisites:
#   1) Start vLLM (separate terminal / GPU):
#        bash examples/coding_agent_rl/sft/launch_vllm_pyrodash4b_sft0902.sh
#   2) node + claude-code tarballs under examples/coding_agent_rl/tarballs/
#
# Usage:
#   bash examples/coding_agent_rl/sft/run_infer_sft_traj.sh
#   LIMIT=2 CONCURRENCY=1 bash examples/coding_agent_rl/sft/run_infer_sft_traj.sh
#   # --eval is on by default; omit by not using this wrapper / pass through carefully
#   DASHSCOPE_BASE_URL=http://10.x.x.x:8066/v1 bash .../run_infer_sft_traj.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." &>/dev/null && pwd)"

# ---- docker / docker-rt ----
if [[ -z "${DOCKER_HOST:-}" && -S /tmp/docker-rt.sock ]]; then
  export DOCKER_HOST=unix:///tmp/docker-rt.sock
fi
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: cannot talk to docker daemon." >&2
  echo "  Tried DOCKER_HOST=${DOCKER_HOST:-<unset>}" >&2
  echo "  On this cluster usually: export DOCKER_HOST=unix:///tmp/docker-rt.sock" >&2
  exit 1
fi

# ---- local vLLM (0902) ----
export DASHSCOPE_BASE_URL="${DASHSCOPE_BASE_URL:-http://10.244.0.61:8066/v1}"
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-EMPTY}"
export DASHSCOPE_MODEL="${DASHSCOPE_MODEL:-/workspace/models/pyromind/PyroDash-4B-SFT-0902}"
export INFER_REASONING_EFFORT="${INFER_REASONING_EFFORT:-}"

# Entropy sampling (match annotate_reward1_entropy defaults)
export TEMPERATURE="${TEMPERATURE:-0.6}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:-20}"
export TOP_LOGPROBS="${TOP_LOGPROBS:-20}"
export ENTROPY_SCOPE="${ENTROPY_SCOPE:-thinking}"

# ---- sandbox ↔ adapter ----
export SLIME_AGENT_OFFLOAD=0
export SLIME_AGENT_SANDBOX_BACKEND="${SLIME_AGENT_SANDBOX_BACKEND:-docker}"
export SLIME_AGENT_DOCKER_NETWORK="${SLIME_AGENT_DOCKER_NETWORK:-bridge}"
export SLIME_AGENT_DOCKER_ADD_HOST="${SLIME_AGENT_DOCKER_ADD_HOST:-host.docker.internal:host-gateway}"
export SLIME_AGENT_DOCKER_NAME_PREFIX="${SLIME_AGENT_DOCKER_NAME_PREFIX:-sft-0902-infer}"

_POD_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-${_POD_IP}}"
if [[ -z "${ADAPTER_PUBLIC_HOST}" || "${ADAPTER_PUBLIC_HOST}" == "127.0.0.1" ]]; then
  echo "ERROR: ADAPTER_PUBLIC_HOST must be a sandbox-routable pod/node IP, not empty/127.0.0.1" >&2
  echo "  Set explicitly, e.g. ADAPTER_PUBLIC_HOST=10.244.2.72" >&2
  exit 1
fi
export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
export ADAPTER_PORT="${ADAPTER_PORT:-18041}"
: "${ADAPTER_PUBLIC_URL:=}"
export ADAPTER_PUBLIC_URL

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/examples/coding_agent_rl/sft/runs/infer_sft_0902_${STAMP}}"
JSONL="${PROMPT_DATA:-${EXAMPLE_DIR}/data/release/mixed_reward1_agents_baked.jsonl}"
TIME_BUDGET="${SWE_AGENT_TIME_BUDGET_SEC:-${TIME_BUDGET:-900}}"
LIMIT="${LIMIT:-${INFER_LIMIT:-3}}"
OFFSET="${OFFSET:-${INFER_OFFSET:-0}}"
CONCURRENCY="${CONCURRENCY:-${INFER_CONCURRENCY:-16}}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-600}"

if [[ -x /root/micromamba/envs/slime/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-/root/micromamba/envs/slime/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

# Enable eval unless caller already passed --eval.
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
echo "Infer agents → SLM-only PyroDash-4B-SFT-0902 + think entropy (no offload)"
echo "  OUT_DIR=${OUT_DIR}"
echo "  JSONL=${JSONL}"
echo "  LIMIT=${LIMIT} OFFSET=${OFFSET} CONCURRENCY=${CONCURRENCY}"
echo "  DOCKER_HOST=${DOCKER_HOST:-<default sock>}"
echo "  ADAPTER_PUBLIC_HOST=${ADAPTER_PUBLIC_HOST}:${ADAPTER_PORT}"
echo "  DASHSCOPE_BASE_URL=${DASHSCOPE_BASE_URL}"
echo "  DASHSCOPE_MODEL=${DASHSCOPE_MODEL}"
echo "  ENTROPY_SCOPE=${ENTROPY_SCOPE} TOP_LOGPROBS=${TOP_LOGPROBS}"
echo "  TEMPERATURE=${TEMPERATURE} TOP_P=${TOP_P} TOP_K=${TOP_K}"
echo "  TIME_BUDGET=${TIME_BUDGET} EVAL_TIMEOUT=${EVAL_TIMEOUT}"
echo "  EXTRA_ARGS=${EXTRA_ARGS[*]}"
echo "======================================================================"

# Quick vLLM health check (non-fatal warning only).
if ! curl -fsS -H "Authorization: Bearer ${DASHSCOPE_API_KEY}" \
  "${DASHSCOPE_BASE_URL%/}/models" >/dev/null 2>&1; then
  echo "WARN: vLLM not reachable at ${DASHSCOPE_BASE_URL}" >&2
  echo "  Start it first: bash ${SCRIPT_DIR}/launch_vllm_pyrodash4b_sft0902.sh" >&2
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/infer_sft_traj.py" \
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
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  --top-logprobs "${TOP_LOGPROBS}" \
  --entropy-scope "${ENTROPY_SCOPE}" \
  --resume \
  "${EXTRA_ARGS[@]}"
