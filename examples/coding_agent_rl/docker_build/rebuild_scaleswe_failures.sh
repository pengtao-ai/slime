#!/usr/bin/env bash
# Rebuild Hub tags that failed bake or verify (force overwrite), then re-verify.
#
# Both failure lists feed the redo queue: images that never built at all are only
# in bake_failures.jsonl (they are also missing from the baked JSONL, so a plain
# verify can never see them), while built-but-broken images land in
# verify_failures.jsonl.
#
# Flow per round:
#   1. Filter source JSONL to failed instance_ids (bake + verify failures)
#   2. bake without --skip-existing (push overwrites same tags)
#   3. Merge baked rows into the main OUTPUT JSONL
#   4. verify the rebuilt set (updates verify_failures.jsonl)
# After rounds, run a full verify on OUTPUT.
#
# Usage:
#   bash examples/coding_agent_rl/docker_build/rebuild_scaleswe_failures.sh
#   REBUILD_ROUNDS=2 bash .../rebuild_scaleswe_failures.sh
#   FAILURES=.../verify_failures.jsonl INPUT=... OUTPUT=... bash .../rebuild_scaleswe_failures.sh
#
# Env: DOCKERHUB_* (or docker_build/.env), BAKE_WORKERS, VERIFY_WORKERS, REBUILD_ROUNDS (default 1)

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

INPUT="${INPUT:-${EXAMPLE_DIR}/data/swe_train_scaleswe_200.jsonl}"
OUTPUT="${OUTPUT:-${EXAMPLE_DIR}/data/swe_train_scaleswe_200_baked.jsonl}"
FAILURES="${FAILURES:-${SCRIPT_DIR}/verify_failures.jsonl}"
BAKE_FAILURES="${BAKE_FAILURES:-${SCRIPT_DIR}/bake_failures.jsonl}"
ROUNDS="${REBUILD_ROUNDS:-1}"
BAKE_WORKERS="${BAKE_WORKERS:-4}"
VERIFY_WORKERS="${VERIFY_WORKERS:-4}"
FILTERED_INPUT="${FILTERED_INPUT:-/tmp/scaleswe_rebuild_failures_input.jsonl}"
FILTERED_BAKED="${FILTERED_BAKED:-/tmp/scaleswe_rebuild_failures_baked.jsonl}"

if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: missing INPUT ${INPUT}" >&2
  exit 1
fi

filter_failures_to_input() {
  local src_path="$1"
  local out_path="$2"
  shift 2
  python3 - "${src_path}" "${out_path}" "$@" <<'PY'
import json
import sys
from pathlib import Path

src_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
fails_paths = [Path(p) for p in sys.argv[3:]]
ids: set[str] = set()
for fails_path in fails_paths:
    if not fails_path.is_file():
        continue
    for line in fails_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        iid = row.get("instance_id")
        if iid:
            ids.add(str(iid))
if not ids:
    print(0)
    raise SystemExit(0)

seen: set[str] = set()
n = 0
with src_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        md = row.get("metadata") or {}
        rem = md.get("remote_env_info") or {}
        iid = md.get("instance_id") or rem.get("instance_id") or row.get("label")
        if not iid or str(iid) not in ids or str(iid) in seen:
            continue
        seen.add(str(iid))
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
missing = sorted(ids - seen)
if missing:
    print(f"[rebuild] WARNING: {len(missing)} failure ids not in INPUT: {missing[:10]}", flush=True)
print(n)
PY
}

merge_baked_into_output() {
  local partial="$1"
  local main_out="$2"
  python3 - "${partial}" "${main_out}" <<'PY'
import json
import sys
from pathlib import Path

partial, main_out = map(Path, sys.argv[1:3])
by_id: dict[str, dict] = {}
if main_out.is_file():
    for line in main_out.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        md = row.get("metadata") or {}
        rem = md.get("remote_env_info") or {}
        iid = md.get("instance_id") or rem.get("instance_id") or row.get("label")
        if iid:
            by_id[str(iid)] = row
for line in partial.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    md = row.get("metadata") or {}
    rem = md.get("remote_env_info") or {}
    iid = md.get("instance_id") or rem.get("instance_id") or row.get("label")
    if iid:
        by_id[str(iid)] = row
main_out.parent.mkdir(parents=True, exist_ok=True)
with main_out.open("w", encoding="utf-8") as fout:
    for iid in sorted(by_id):
        fout.write(json.dumps(by_id[iid], ensure_ascii=False) + "\n")
print(f"[rebuild] merged {partial} → {main_out} ({len(by_id)} rows)", flush=True)
PY
}

failure_count() {
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

pending_count() {
  failure_count "${BAKE_FAILURES}" "${FAILURES}"
}

echo "[rebuild] INPUT=${INPUT}"
echo "[rebuild] OUTPUT=${OUTPUT}"
echo "[rebuild] FAILURES=${FAILURES}"
echo "[rebuild] BAKE_FAILURES=${BAKE_FAILURES}"
echo "[rebuild] rounds=${ROUNDS} bake_workers=${BAKE_WORKERS} verify_workers=${VERIFY_WORKERS}"

round=1
while [[ "${round}" -le "${ROUNDS}" ]]; do
  n_bake="$(failure_count "${BAKE_FAILURES}")"
  n_verify="$(failure_count "${FAILURES}")"
  if [[ "$((n_bake + n_verify))" -eq 0 ]]; then
    echo "[rebuild] no bake/verify failures; stop"
    break
  fi
  echo "[rebuild] === round ${round}/${ROUNDS}: bake_failures=${n_bake} verify_failures=${n_verify} ==="

  n_jobs="$(filter_failures_to_input "${INPUT}" "${FILTERED_INPUT}" "${BAKE_FAILURES}" "${FAILURES}")"
  if [[ "${n_jobs}" -eq 0 ]]; then
    echo "[rebuild] filtered input empty; stop" >&2
    break
  fi
  echo "[rebuild] force-bake ${n_jobs} image(s) (no --skip-existing) → overwrite Hub tags"

  # Rewrites BAKE_FAILURES with only this round's bake failures.
  set +e
  python3 "${SCRIPT_DIR}/bake_scaleswe_agent_images.py" \
    --input "${FILTERED_INPUT}" \
    --output "${FILTERED_BAKED}" \
    --failures "${BAKE_FAILURES}" \
    --workers "${BAKE_WORKERS}"
  bake_rc=$?
  set -e
  echo "[rebuild] round ${round} bake exit=${bake_rc} bake_failures=$(failure_count "${BAKE_FAILURES}")"

  merge_baked_into_output "${FILTERED_BAKED}" "${OUTPUT}"

  echo "[rebuild] verify rebuilt set"
  set +e
  python3 "${SCRIPT_DIR}/verify_scaleswe_agent_images.py" \
    --input "${FILTERED_BAKED}" \
    --failures-out "${FAILURES}" \
    --workers "${VERIFY_WORKERS}"
  verify_rc=$?
  set -e
  echo "[rebuild] round ${round} verify exit=${verify_rc} remaining=$(pending_count)"
  round=$((round + 1))
done

echo "[rebuild] full verify → ${OUTPUT}"
set +e
python3 "${SCRIPT_DIR}/verify_scaleswe_agent_images.py" \
  --input "${OUTPUT}" \
  --failures-out "${FAILURES}" \
  --workers "${VERIFY_WORKERS}"
full_rc=$?
set -e

n_left="$(pending_count)"
echo "[rebuild] done remaining bake=$(failure_count "${BAKE_FAILURES}") verify=$(failure_count "${FAILURES}") total=${n_left}"
if [[ "${n_left}" -gt 0 || "${full_rc}" -ne 0 ]]; then
  exit 1
fi
