#!/usr/bin/env bash
# Launch local SGLang serving PyroDash-4B for request-replay / offload eval.
#
# Example:
#   bash examples/coding_agent_rl/launch_sglang_pyrodash4b.sh
#   CUDA_VISIBLE_DEVICES=1 PORT=30001 bash examples/coding_agent_rl/launch_sglang_pyrodash4b.sh
#   SGLANG_HOST=127.0.0.1 PORT=30000 bash examples/coding_agent_rl/launch_sglang_pyrodash4b.sh
#
# Smoke check:
#   curl http://127.0.0.1:30000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#       "model": "PyroDash-4B-SFT-0728",
#       "messages": [{"role": "user", "content": "你好"}],
#       "stop_token_ids": [248046, 248044, 248078],
#       "max_tokens": 32
#     }'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/workspace/models/pyromind/PyroDash-4B-SFT-0728}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-PyroDash-4B-SFT-0728}"
PORT="${PORT:-30000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
TP_SIZE="${TP_SIZE:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

# Bind address. Prefer SGLANG_HOST / LISTEN_HOST — do NOT blindly use $HOST.
# Activating conda/micromamba (or conda-build leftovers) often sets
# HOST=x86_64-conda-linux-gnu, which is not a resolvable listen address and
# makes torch.distributed hang on IPv6 hostname lookup.
LISTEN_HOST="${SGLANG_HOST:-${LISTEN_HOST:-}}"
if [[ -z "${LISTEN_HOST}" ]]; then
  if [[ -n "${HOST:-}" && "${HOST}" != *conda-linux-gnu* && "${HOST}" != *unknown* ]]; then
    LISTEN_HOST="${HOST}"
  else
    LISTEN_HOST="127.0.0.1"
  fi
fi
# Keep torch/c10d off the broken conda HOST triple.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
unset HOST 2>/dev/null || true

# Prefer slime env if present (has working sglang build on this machine).
if [[ -x /root/micromamba/envs/slime/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-/root/micromamba/envs/slime/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: model path not found: ${MODEL_PATH}" >&2
  exit 1
fi

# SGLang's debug dumper imports megatron.core at module load time. In the slime
# env that pulls Transformer Engine + a broken libcudnn_cnn.so.9 (undefined
# SdpaForwardOperation). Prepend a tiny stub package so inference-only serving
# never touches real Megatron / TE / cuDNN.
STUB_ROOT="${SCRIPT_DIR}/sglang_infer_stubs"
export PYTHONPATH="${STUB_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Drop editable Megatron-LM from PYTHONPATH if present.
if [[ -n "${PYTHONPATH}" ]]; then
  _filtered=""
  IFS=':' read -r -a _parts <<< "${PYTHONPATH}"
  for _p in "${_parts[@]}"; do
    [[ -z "${_p}" ]] && continue
    [[ "${_p}" == *Megatron* ]] && continue
    if [[ -z "${_filtered}" ]]; then
      _filtered="${_p}"
    else
      _filtered="${_filtered}:${_p}"
    fi
  done
  export PYTHONPATH="${_filtered}"
fi

echo "======================================================================"
echo "SGLang launch"
echo "  PYTHON=${PYTHON_BIN}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "  listen=${LISTEN_HOST}:${PORT}"
echo "  tool-call-parser=${TOOL_CALL_PARSER} reasoning-parser=${REASONING_PARSER}"
echo "  PYTHONPATH stub=${STUB_ROOT} (skip Megatron/TE/cuDNN)"
echo "======================================================================"

export CUDA_VISIBLE_DEVICES
exec "${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${LISTEN_HOST}" \
  --port "${PORT}" \
  --tp-size "${TP_SIZE}" \
  --tool-call-parser "${TOOL_CALL_PARSER}" \
  --reasoning-parser "${REASONING_PARSER}" \
  --trust-remote-code \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  "$@"
