#!/usr/bin/env bash
# Qwen3.5-4B coding-agent RL — *async* train + *local Docker* sandboxes.
# Wrapper around run_qwen35_4b_swe_1node_async.sh.
#
# Differences vs run_qwen35_4b_swe_1node_docker.sh:
#   * train_async.py (next rollout overlaps Megatron train)
#   * no --colocate: default 6 actor + 2 rollout GPUs (train is the wall-clock bottleneck)
#   * default TP=1 CP=2 → DP=3; fan-out 3×8=24 (divisible by DP)
#   * default --qwen-gdn-backend flashqla
#   * Megatron --save every SAVE_INTERVAL steps (default 5)
#
# After warmup, wall clock ≈ max(rollout, train). Keep train <= rollout or async helps little.
#
# Prerequisites: same as the sync docker wrapper (docker CLI, images, pod IP).
#
# Run from a long-lived shell / tmux:
#   bash examples/coding_agent_rl/run_qwen35_4b_swe_1node_docker_async.sh
#
# Faster train (closer to sync):
#   ROLLOUT_BATCH_SIZE=2 N_SAMPLES_PER_PROMPT=8 \
#     bash examples/coding_agent_rl/run_qwen35_4b_swe_1node_docker_async.sh
#
# Higher fan-out if you accept longer train:
#   ROLLOUT_BATCH_SIZE=6 N_SAMPLES_PER_PROMPT=8 \
#     bash examples/coding_agent_rl/run_qwen35_4b_swe_1node_docker_async.sh
#
# Fall back to FLA GDN / max CP:
#   QWEN_GDN_BACKEND=fla CP_SIZE=6 ROLLOUT_BATCH_SIZE=8 \
#     bash examples/coding_agent_rl/run_qwen35_4b_swe_1node_docker_async.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ---- sandbox backend: local Docker ----
export SLIME_AGENT_SANDBOX_BACKEND="${SLIME_AGENT_SANDBOX_BACKEND:-docker}"
export SLIME_AGENT_E2B_USE_TEMPLATE=0
export SLIME_AGENT_DOCKER_NETWORK="${SLIME_AGENT_DOCKER_NETWORK:-bridge}"
export SLIME_AGENT_DOCKER_ADD_HOST="${SLIME_AGENT_DOCKER_ADD_HOST:-host.docker.internal:host-gateway}"
export SLIME_AGENT_DOCKER_PULL="${SLIME_AGENT_DOCKER_PULL:-0}"
# docker-rt pod scheduling can exceed 120s under load; sandbox cleans orphans on timeout.
export SLIME_AGENT_DOCKER_RUN_TIMEOUT_SEC="${SLIME_AGENT_DOCKER_RUN_TIMEOUT_SEC:-300}"

_POD_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-${_POD_IP}}"
if [[ -z "${ADAPTER_PUBLIC_HOST}" || "${ADAPTER_PUBLIC_HOST}" == "127.0.0.1" ]]; then
  echo "ERROR: ADAPTER_PUBLIC_HOST must be a sandbox-routable pod/node IP, not empty/127.0.0.1" >&2
  echo "  (docker-rt sandboxes cannot reach the trainer via loopback)." >&2
  exit 1
fi
export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
export ADAPTER_PORT="${ADAPTER_PORT:-18001}"
export ADAPTER_PUBLIC_URL="${ADAPTER_PUBLIC_URL:-}"

export PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/data/swe_train_scaleswe_200.jsonl}"
export EXP_TAG="${EXP_TAG:-agent_only_qwen35_4b_docker_async_scaleswe200}"

# Concurrent agents ≈ ROLLOUT_BATCH_SIZE * N_SAMPLES.
# Train FLOPs ≈ GBS * avg_seqlen; GBS must be divisible by DP.
# Default CP=2 → DP=3 on ACTOR_GPUS=6; N_SAMPLES=8 → ROLLOUT_BATCH_SIZE=3 so GBS=24 % 3 == 0.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-32}"
# 200 prompts / batch=3 ≈ 67 steps/epoch.
export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"

# Prefer actor GPUs: agents are sandbox-bound; train was the bottleneck on 4+4.
# TP=1 CP=2 → DP=3 on 6 actor GPUs (GDN still all-gathers full seq per CP group).
export NUM_GPUS="${NUM_GPUS:-8}"
export ACTOR_GPUS="${ACTOR_GPUS:-6}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-$((NUM_GPUS - ACTOR_GPUS))}"
export TP_SIZE="${TP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-6}"
export QWEN_GDN_BACKEND="${QWEN_GDN_BACKEND:-flashqla}"

# Checkpointing.
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
# Agent budget: 900 keeps step latency down; raise to 1200 if too many soft timeouts.
export SWE_AGENT_TIME_BUDGET_SEC="${SWE_AGENT_TIME_BUDGET_SEC:-900}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-300}"

# Train-only (load-debug-rollout-data): skip Docker sandbox checks; no SGLang.
_DEBUG_TRAIN="${DEBUG_TRAIN_ONLY:-0}"
if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA:-}" ]]; then
  _DEBUG_TRAIN=1
fi

if [[ "${_DEBUG_TRAIN}" != "1" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found on PATH" >&2
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "ERROR: docker daemon not reachable (docker info failed)" >&2
    exit 1
  fi
else
  export ACTOR_GPUS="${ACTOR_GPUS:-${NUM_GPUS}}"
  export ROLLOUT_GPUS="${ROLLOUT_GPUS:-0}"
  echo "DEBUG train-only mode: skip docker checks (ACTOR_GPUS=${ACTOR_GPUS} ROLLOUT_GPUS=${ROLLOUT_GPUS})"
fi

echo "======================================================================"
echo "Local Docker coding-agent RL (ASYNC)"
echo "  SLIME_AGENT_SANDBOX_BACKEND=${SLIME_AGENT_SANDBOX_BACKEND}"
echo "  SLIME_AGENT_DOCKER_NETWORK=${SLIME_AGENT_DOCKER_NETWORK}"
echo "  ADAPTER_PUBLIC_HOST=${ADAPTER_PUBLIC_HOST}:${ADAPTER_PORT}"
echo "  ADAPTER_BIND_HOST=${ADAPTER_BIND_HOST}"
echo "  PROMPT_DATA=${PROMPT_DATA}"
echo "  NUM_ROLLOUT=${NUM_ROLLOUT}"
echo "  ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE} N_SAMPLES=${N_SAMPLES_PER_PROMPT} GLOBAL_BATCH=${GLOBAL_BATCH_SIZE}"
echo "  SWE_BOOT_CONCURRENCY=${SWE_BOOT_CONCURRENCY}"
echo "  ACTOR_GPUS=${ACTOR_GPUS} ROLLOUT_GPUS=${ROLLOUT_GPUS} TP_SIZE=${TP_SIZE} CP_SIZE=${CP_SIZE}"
echo "  QWEN_GDN_BACKEND=${QWEN_GDN_BACKEND}"
echo "  SAVE_INTERVAL=${SAVE_INTERVAL}"
echo "  SWE_AGENT_TIME_BUDGET_SEC=${SWE_AGENT_TIME_BUDGET_SEC} SWE_EVAL_TIMEOUT_SEC=${SWE_EVAL_TIMEOUT_SEC}"
echo "  SLIME_AGENT_DOCKER_RUN_TIMEOUT_SEC=${SLIME_AGENT_DOCKER_RUN_TIMEOUT_SEC}"
echo "======================================================================"

exec bash "${SCRIPT_DIR}/run_qwen35_4b_swe_1node_async.sh"
