#!/usr/bin/env bash
# Bake a small Tmax agent-image smoke set (Kaniko + proot), then verify.
#
# Credentials: copy .env.example → .env and fill DOCKERHUB_* (gitignored).
#
# Usage:
#   INPUT=.../mixed_agents_bake_smoke_tmax.jsonl \
#   OUTPUT=.../mixed_agents_bake_smoke_tmax_baked.jsonl \
#     bash examples/coding_agent_rl/docker_build/bake_tmax_smoke.sh
#   bash examples/coding_agent_rl/docker_build/bake_tmax_smoke.sh --limit 1
#   SKIP_VERIFY=1 bash .../bake_tmax_smoke.sh
#
# Env: BAKE_WORKERS (default 2), VERIFY_WORKERS (default 2)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SLIME_DIR="$(cd "${EXAMPLE_DIR}/../.." && pwd)"

ENV_FILE="${SCRIPT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [[ -z "${DOCKERHUB_USERNAME:-}" || -z "${DOCKERHUB_TOKEN:-}" ]]; then
  echo "ERROR: set DOCKERHUB_USERNAME and DOCKERHUB_TOKEN in ${ENV_FILE} (see .env.example)" >&2
  exit 1
fi

cd "${SLIME_DIR}"

INPUT="${INPUT:-${EXAMPLE_DIR}/data/mixed_agents_bake_smoke_tmax.jsonl}"
OUTPUT="${OUTPUT:-${EXAMPLE_DIR}/data/mixed_agents_bake_smoke_tmax_baked.jsonl}"
FAILURES="${FAILURES:-${SCRIPT_DIR}/verify_tmax_failures.jsonl}"
BAKE_FAILURES="${BAKE_FAILURES:-${SCRIPT_DIR}/bake_tmax_failures.jsonl}"
BAKE_WORKERS="${BAKE_WORKERS:-2}"
VERIFY_WORKERS="${VERIFY_WORKERS:-2}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"

GENERATE_ONLY=0
for arg in "$@"; do
  if [[ "${arg}" == "--generate-only" ]]; then
    GENERATE_ONLY=1
  fi
done

echo "[bake-tmax] bake workers=${BAKE_WORKERS} → ${OUTPUT}"
set +e
python3 "${SCRIPT_DIR}/bake_tmax_agent_images.py" \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --failures "${BAKE_FAILURES}" \
  --workers "${BAKE_WORKERS}" \
  "$@"
bake_rc=$?
set -e
echo "[bake-tmax] bake exit=${bake_rc} → ${BAKE_FAILURES}"

if [[ "${GENERATE_ONLY}" -eq 1 ]]; then
  echo "[bake-tmax] --generate-only: skip verify"
  exit 0
fi

if [[ "${SKIP_VERIFY}" == "1" ]]; then
  echo "[bake-tmax] SKIP_VERIFY=1: done after bake"
  exit "${bake_rc}"
fi

echo "[bake-tmax] verify → ${FAILURES}"
set +e
python3 "${SCRIPT_DIR}/verify_tmax_agent_images.py" \
  --input "${OUTPUT}" \
  --failures-out "${FAILURES}" \
  --workers "${VERIFY_WORKERS}"
verify_rc=$?
set -e

if [[ "${bake_rc}" -ne 0 || "${verify_rc}" -ne 0 ]]; then
  echo "[bake-tmax] FAIL bake_exit=${bake_rc} verify_exit=${verify_rc}" >&2
  exit 1
fi
echo "[bake-tmax] bake + verify OK"
