#!/usr/bin/env bash
# Offline regrade: score existing patch.diff under an infer OUT_DIR (no agent / no GLM).
#
# Defaults target the GLM-only infer run. Skips empty patches and samples that
# already have eval_applied=true unless you pass --force / --no-skip-eval-applied.
#
# Prerequisites: a working docker / docker-rt daemon (same host as infer/train).
# No DASHSCOPE_* needed. A plain jupyter pod without /var/run/docker.sock will
# fail the preflight — do not run regrade there.
#
# Example:
#   bash examples/coding_agent_rl/run_regrade_patches.sh --dry-run
#   CONCURRENCY=4 bash examples/coding_agent_rl/run_regrade_patches.sh
#   # default: redo any sample with summary.regrade, plus never-graded ones;
#   # skips inline eval_applied=true that were never regraded.
#   bash examples/coding_agent_rl/run_regrade_patches.sh --only-regraded
#   bash examples/coding_agent_rl/run_regrade_patches.sh --indices 851,852 --force
#   LIMIT=20 bash examples/coding_agent_rl/run_regrade_patches.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

export SLIME_AGENT_SANDBOX_BACKEND="${SLIME_AGENT_SANDBOX_BACKEND:-docker}"
export SLIME_AGENT_DOCKER_NETWORK="${SLIME_AGENT_DOCKER_NETWORK:-bridge}"
export SLIME_AGENT_DOCKER_ADD_HOST="${SLIME_AGENT_DOCKER_ADD_HOST:-host.docker.internal:host-gateway}"
export SLIME_AGENT_DOCKER_NAME_PREFIX="${SLIME_AGENT_DOCKER_NAME_PREFIX:-cc-regrade}"

# Early hint (Python also preflights via `docker info`).
if [[ "${1:-}" != "--dry-run" ]] && ! printf '%s\n' "$@" | grep -qx -- '--dry-run'; then
  if [[ -z "${DOCKER_HOST:-}" && ! -S /var/run/docker.sock ]]; then
    echo "ERROR: no docker daemon here (DOCKER_HOST unset, /var/run/docker.sock missing)." >&2
    echo "  Run on the same pod/node where infer/train sandboxes work, or set DOCKER_HOST." >&2
    echo "  Quick check: docker info" >&2
    exit 1
  fi
fi

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/runs/infer_cc_glm_20260728_141118}"
JSONL="${PROMPT_DATA:-${SCRIPT_DIR}/data/swe_train_scaleswe.jsonl}"
CONCURRENCY="${CONCURRENCY:-${REGRADE_CONCURRENCY:-4}}"
EVAL_TIMEOUT="${SWE_EVAL_TIMEOUT_SEC:-${EVAL_TIMEOUT:-600}}"

EXTRA_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  EXTRA_ARGS+=(--limit "${LIMIT}")
fi
if [[ -n "${OFFSET:-}" ]]; then
  EXTRA_ARGS+=(--offset "${OFFSET}")
fi

echo "======================================================================"
echo "Regrade existing patches (no agent)"
echo "  OUT_DIR=${OUT_DIR}"
echo "  JSONL=${JSONL}"
echo "  CONCURRENCY=${CONCURRENCY}"
echo "  EVAL_TIMEOUT=${EVAL_TIMEOUT}"
echo "  SLIME_AGENT_SANDBOX_BACKEND=${SLIME_AGENT_SANDBOX_BACKEND}"
echo "  SLIME_AGENT_DOCKER_NETWORK=${SLIME_AGENT_DOCKER_NETWORK}"
echo "  SLIME_AGENT_DOCKER_NAME_PREFIX=${SLIME_AGENT_DOCKER_NAME_PREFIX}"
echo "======================================================================"

exec python "${SCRIPT_DIR}/regrade_patches.py" \
  --out-dir "${OUT_DIR}" \
  --jsonl "${JSONL}" \
  --concurrency "${CONCURRENCY}" \
  --eval-timeout "${EVAL_TIMEOUT}" \
  --network "${SLIME_AGENT_DOCKER_NETWORK}" \
  --name-prefix "${SLIME_AGENT_DOCKER_NAME_PREFIX}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
