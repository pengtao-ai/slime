#!/usr/bin/env bash
# Convert PyroDash-4B (Qwen3.5-4B arch) HF checkpoint -> Megatron torch_dist.
#
# Usage:
#   bash examples/coding_agent_rl/convert_pyrodash4b_to_torch_dist.sh
#   HF_CHECKPOINT=/path/to/hf SAVE=/path/to/out \
#     bash examples/coding_agent_rl/convert_pyrodash4b_to_torch_dist.sh
#
# Notes:
#   * Uses scripts/models/qwen3.5-4B.sh (Megatron --vocab-size 248320).
#   * PyroDash HF embed_tokens=248079 is intentional: base Qwen3.5-4B
#     tokenizer length 248077 + 2 added special tokens. We do NOT change the
#     original HF weights; staging only zero-pads rows 248079..248319 so the
#     tensor matches Megatron/qwen3.5-4B.sh padded vocab (same as upstream
#     Qwen3.5-4B HF, which already ships embed as 248320).
#   * Prefer a free GPU: CUDA_VISIBLE_DEVICES=0 bash ...

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/root/Megatron-LM}"

HF_CHECKPOINT="${HF_CHECKPOINT:-/workspace/models/pyromind/PyroDash-4B-SFT-0723}"
SAVE="${SAVE:-/workspace/models/pyromind/PyroDash-4B-SFT-0723_torch_dist}"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-29551}"
TARGET_VOCAB="${TARGET_VOCAB:-248320}"
STAGING_DIR="${STAGING_DIR:-${SAVE}.hf_padded_staging}"
KEEP_STAGING="${KEEP_STAGING:-0}"

if [[ ! -d "${HF_CHECKPOINT}" ]]; then
  echo "ERROR: HF checkpoint not found: ${HF_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -d "${MEGATRON_LM_PATH}" ]]; then
  echo "ERROR: Megatron-LM not found: ${MEGATRON_LM_PATH}" >&2
  exit 1
fi

cd "${SLIME_DIR}"
# shellcheck disable=SC1091
source "${SLIME_DIR}/scripts/models/qwen3.5-4B.sh"

echo "======================================================================"
echo "HF -> torch_dist (PyroDash-4B / Qwen3.5-4B)"
echo "  HF_CHECKPOINT=${HF_CHECKPOINT}"
echo "  STAGING_DIR=${STAGING_DIR}"
echo "  SAVE=${SAVE}"
echo "  TARGET_VOCAB=${TARGET_VOCAB}"
echo "  NPROC=${NPROC}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "======================================================================"

mkdir -p "$(dirname -- "${SAVE}")"
rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}"

# Fast copy then pad vocab-sized embeddings in staging (do not mutate the original HF).
echo "[1/3] Staging HF checkpoint..."
cp -a "${HF_CHECKPOINT}/." "${STAGING_DIR}/"

echo "[2/3] Zero-pad embed_tokens to Megatron vocab=${TARGET_VOCAB} (HF 248079 = 248077+2 tokens stays intact in rows [:248079])..."
python - "${STAGING_DIR}" "${TARGET_VOCAB}" <<'PY'
import json
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
import torch

staging = Path(sys.argv[1])
target_vocab = int(sys.argv[2])
embed_key = "model.language_model.embed_tokens.weight"

# Staging-only: Megatron/qwen3.5-4B.sh expects padded vocab 248320.
# Real tokens (incl. PyroDash's +2) remain in [:hf_rows]; new rows are zeros.
cfg_path = staging / "config.json"
cfg = json.loads(cfg_path.read_text())
text = cfg.setdefault("text_config", cfg)
old = int(text.get("vocab_size", 0))
text["vocab_size"] = target_vocab
if "vocab_size" in cfg and cfg is not text:
    cfg["vocab_size"] = target_vocab
cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
print(f"  staging config vocab_size: {old} -> {target_vocab} (original HF unchanged)")

# Find and pad the embedding shard.
shard = None
for path in sorted(staging.glob("*.safetensors")):
    with safe_open(str(path), framework="pt") as f:
        if embed_key in f.keys():
            shard = path
            weight = f.get_tensor(embed_key)
            break
if shard is None:
    raise SystemExit(f"ERROR: {embed_key} not found under {staging}")

rows, dim = weight.shape
print(f"  {embed_key}: {tuple(weight.shape)} in {shard.name}")
if rows == target_vocab:
    print("  already padded; skip weight rewrite")
    raise SystemExit(0)
if rows > target_vocab:
    raise SystemExit(f"ERROR: embed rows {rows} > target {target_vocab}")

# Reload full shard, replace embed, rewrite file.
tensors = {}
with safe_open(str(shard), framework="pt") as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

padded = torch.zeros((target_vocab, dim), dtype=weight.dtype)
padded[:rows].copy_(weight)
tensors[embed_key] = padded
save_file(tensors, str(shard))
print(f"  wrote padded embed {tuple(padded.shape)} -> {shard}")
PY

echo "[3/3] Converting with tools/convert_hf_to_torch_dist.py ..."
export PYTHONPATH="${MEGATRON_LM_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${NPROC}" -le 1 ]]; then
  python tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${STAGING_DIR}" \
    --save "${SAVE}"
else
  torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
    tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${STAGING_DIR}" \
    --save "${SAVE}"
fi

if [[ "${KEEP_STAGING}" != "1" ]]; then
  rm -rf "${STAGING_DIR}"
fi

echo "Done. torch_dist checkpoint at: ${SAVE}"
ls -la "${SAVE}"
test -f "${SAVE}/latest_checkpointed_iteration.txt"
test -d "${SAVE}/release"
