#!/usr/bin/env bash
# Bake ScaleSWE agent images (Kaniko + proot) for
# swe_train_scaleswe_reward1_glm_20260728.jsonl (2679 instances),
# then verify; on failures, force-rebuild + overwrite Hub tags and re-verify.
#
# Credentials: copy .env.example → .env and fill DOCKERHUB_* (gitignored).
#
# Usage:
#   bash examples/coding_agent_rl/docker_build/bake_scaleswe_reward1_glm_20260728.sh
#   bash examples/coding_agent_rl/docker_build/bake_scaleswe_reward1_glm_20260728.sh --limit 1
#   bash examples/coding_agent_rl/docker_build/bake_scaleswe_reward1_glm_20260728.sh --generate-only
#   SKIP_VERIFY=1 bash .../bake_scaleswe_reward1_glm_20260728.sh          # bake only
#   SKIP_REBUILD=1 bash .../bake_scaleswe_reward1_glm_20260728.sh          # bake + verify, no redo
#
# Env: BAKE_WORKERS (default 8), VERIFY_WORKERS (default 4), REBUILD_ROUNDS (default 1)

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

INPUT="${INPUT:-${EXAMPLE_DIR}/data/swe_train_scaleswe_reward1_glm_20260728.jsonl}"
OUTPUT="${OUTPUT:-${EXAMPLE_DIR}/data/swe_train_scaleswe_reward1_glm_20260728_baked.jsonl}"
# Keep failure logs separate from bake_scaleswe_200.sh so concurrent runs do not clash.
FAILURES="${FAILURES:-${SCRIPT_DIR}/verify_failures_reward1_glm_20260728.jsonl}"
BAKE_FAILURES="${BAKE_FAILURES:-${SCRIPT_DIR}/bake_failures_reward1_glm_20260728.jsonl}"
FILTERED_INPUT="${FILTERED_INPUT:-/tmp/scaleswe_rebuild_reward1_glm_20260728_input.jsonl}"
BAKE_WORKERS="${BAKE_WORKERS:-8}"
VERIFY_WORKERS="${VERIFY_WORKERS:-4}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"
SKIP_REBUILD="${SKIP_REBUILD:-0}"

GENERATE_ONLY=0
for arg in "$@"; do
  if [[ "${arg}" == "--generate-only" ]]; then
    GENERATE_ONLY=1
  fi
done

count_failures() {
  python3 -c '
import sys
from pathlib import Path
total = 0
for arg in sys.argv[1:]:
    p = Path(arg)
    if p.is_file():
        total += sum(1 for line in p.read_text().splitlines() if line.strip())
print(total)
' "$@"
}

if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: missing INPUT ${INPUT}" >&2
  exit 1
fi

echo "[bake_reward1] INPUT=${INPUT}"
echo "[bake_reward1] OUTPUT=${OUTPUT}"
echo "[bake_reward1] bake workers=${BAKE_WORKERS} → ${OUTPUT}"
# Non-fatal: bake exits 1 when any image fails, but those instances must still
# reach the rebuild step (they are absent from OUTPUT, so verify cannot see them).
set +e
python3 "${SCRIPT_DIR}/bake_scaleswe_agent_images.py" \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --failures "${BAKE_FAILURES}" \
  --workers "${BAKE_WORKERS}" \
  "$@"
bake_rc=$?
set -e
n_bake_fail="$(count_failures "${BAKE_FAILURES}")"
echo "[bake_reward1] bake exit=${bake_rc} bake_failures=${n_bake_fail} → ${BAKE_FAILURES}"

if [[ "${GENERATE_ONLY}" -eq 1 ]]; then
  echo "[bake_reward1] --generate-only: skip verify/rebuild"
  exit 0
fi

if [[ "${SKIP_VERIFY}" == "1" ]]; then
  echo "[bake_reward1] SKIP_VERIFY=1: done after bake"
  exit 0
fi

echo "[bake_reward1] verify → ${FAILURES}"
set +e
python3 "${SCRIPT_DIR}/verify_scaleswe_agent_images.py" \
  --input "${OUTPUT}" \
  --failures-out "${FAILURES}" \
  --workers "${VERIFY_WORKERS}"
verify_rc=$?
set -e

n_verify_fail="$(count_failures "${FAILURES}")"
n_bake_fail="$(count_failures "${BAKE_FAILURES}")"
n_fail=$((n_bake_fail + n_verify_fail))

if [[ "${n_fail}" -eq 0 ]]; then
  echo "[bake_reward1] bake + verify OK (verify exit=${verify_rc})"
  exit 0
fi

echo "[bake_reward1] pending: bake_failures=${n_bake_fail} verify_failures=${n_verify_fail}"
if [[ "${SKIP_REBUILD}" == "1" ]]; then
  echo "[bake_reward1] SKIP_REBUILD=1: not overwriting Hub tags" >&2
  exit 1
fi

echo "[bake_reward1] calling rebuild_scaleswe_failures.sh (force overwrite + re-verify)"
export INPUT OUTPUT FAILURES BAKE_FAILURES BAKE_WORKERS VERIFY_WORKERS FILTERED_INPUT
export REBUILD_ROUNDS="${REBUILD_ROUNDS:-1}"
bash "${SCRIPT_DIR}/rebuild_scaleswe_failures.sh"
