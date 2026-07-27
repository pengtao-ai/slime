#!/usr/bin/env bash
# Qwen3.5-4B coding-agent RL on 1 node (8 GPUs) with *local Docker* sandboxes.
# Wrapper around run_qwen35_4b_swe_1node.sh with Docker-oriented defaults.
#
# Prerequisites:
#   - docker CLI works on this host
#   - dataset metadata.image is a real Docker image (not an E2B template name)
#   - images pulled (or set SLIME_AGENT_DOCKER_PULL=1)
#
# Smoke first (no full train.sh / Ray) — requires a local SGLang:
#   CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
#     --model-path /workspace/models/Qwen/Qwen3.5-4B \
#     --served-model-name qwen --host 127.0.0.1 --port 30000
#   python examples/coding_agent_rl/smoke_claude_code_docker.py
#
# Run full training from a long-lived shell / tmux:
#   bash examples/coding_agent_rl/run_qwen35_4b_swe_1node_docker.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ---- sandbox backend: local Docker ----
# Lessons from smoke_claude_code_docker.py on docker-rt / k8s:
#   * bridge network (host netns is usually the *node*, not this pod)
#   * ADAPTER_PUBLIC_HOST = pod IP (host.docker.internal often does not resolve;
#     docker-rt ignores --add-host)
#   * ADAPTER_BIND_HOST = 0.0.0.0 so sandboxes can dial the pod IP
#   * Claude Code needs CLAUDE_CODE_MAX_OUTPUT_TOKENS (default 32k client cap)
# Sandbox env/-u are handled in slime/agent/sandbox.py (export in launcher +
# runuser), not via fragile `docker exec -e/-u`.
export SLIME_AGENT_SANDBOX_BACKEND="${SLIME_AGENT_SANDBOX_BACKEND:-docker}"
export SLIME_AGENT_E2B_USE_TEMPLATE=0
export SLIME_AGENT_DOCKER_NETWORK="${SLIME_AGENT_DOCKER_NETWORK:-bridge}"
export SLIME_AGENT_DOCKER_ADD_HOST="${SLIME_AGENT_DOCKER_ADD_HOST:-host.docker.internal:host-gateway}"
export SLIME_AGENT_DOCKER_PULL="${SLIME_AGENT_DOCKER_PULL:-0}"

_POD_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-${_POD_IP}}"
if [[ -z "${ADAPTER_PUBLIC_HOST}" || "${ADAPTER_PUBLIC_HOST}" == "127.0.0.1" ]]; then
  echo "ERROR: ADAPTER_PUBLIC_HOST must be a sandbox-routable pod/node IP, not empty/127.0.0.1" >&2
  echo "  (docker-rt sandboxes cannot reach the trainer via loopback)." >&2
  exit 1
fi
export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
export ADAPTER_PORT="${ADAPTER_PORT:-18001}"
# Clear any leftover Cloudflare URL from a public-E2B session.
export ADAPTER_PUBLIC_URL="${ADAPTER_PUBLIC_URL:-}"

# Default to the single-sample Docker smoke row (image = aweaiteam/scaleswe:...).
# Override with swe_train_scaleswe_200.jsonl for larger runs (images must exist locally).
export PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/data/swe_smoke_preliz_docker.jsonl}"
export EXP_TAG="${EXP_TAG:-agent_only_qwen35_4b_docker}"

# Keep fan-out modest: each sample boots agent (+ eval) containers.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-4}"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found on PATH" >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: docker daemon not reachable (docker info failed)" >&2
  exit 1
fi

echo "======================================================================"
echo "Local Docker coding-agent RL"
echo "  SLIME_AGENT_SANDBOX_BACKEND=${SLIME_AGENT_SANDBOX_BACKEND}"
echo "  SLIME_AGENT_DOCKER_NETWORK=${SLIME_AGENT_DOCKER_NETWORK}"
echo "  ADAPTER_PUBLIC_HOST=${ADAPTER_PUBLIC_HOST}:${ADAPTER_PORT}"
echo "  ADAPTER_BIND_HOST=${ADAPTER_BIND_HOST}"
echo "  PROMPT_DATA=${PROMPT_DATA}"
echo "  ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE} N_SAMPLES=${N_SAMPLES_PER_PROMPT}"
echo "======================================================================"

exec bash "${SCRIPT_DIR}/run_qwen35_4b_swe_1node.sh"
