#!/usr/bin/env bash
# Qwen3.5-4B coding-agent RL — *async* train + *local Docker* sandboxes.
# Wrapper around run_qwen35_4b_swe_1node_async.sh.
#
# Differences vs run_qwen35_4b_swe_1node_docker.sh:
#   * train_async.py (next rollout overlaps Megatron train)
#   * no --colocate: default 6 actor + 2 rollout GPUs (train is the wall-clock bottleneck)
#   * default fan-out 8×8=64 (128 made actor_train ~73min/step; sync~6min at ~32 samples)
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
#   ROLLOUT_BATCH_SIZE=4 N_SAMPLES_PER_PROMPT=8 \
#     bash examples/coding_agent_rl/run_qwen35_4b_swe_1node_docker_async.sh
#
# Higher fan-out if you accept longer train (quota≈720):
#   ROLLOUT_BATCH_SIZE=16 N_SAMPLES_PER_PROMPT=8 ACTOR_GPUS=6 ROLLOUT_GPUS=2 \
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
# Train FLOPs ≈ GBS * avg_seqlen; 128×~43k on 4 GPUs → ~73min/step. Default 64.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-32}"
# 200 prompts / batch=8 ≈ 25 steps/epoch.
export NUM_ROLLOUT="${NUM_ROLLOUT:-50}"

# Prefer actor GPUs: agents are sandbox-bound; train was the bottleneck on 4+4.
# TP=1 / CP=ACTOR keeps long-seq memory safe (see async script).
export NUM_GPUS="${NUM_GPUS:-8}"
export ACTOR_GPUS="${ACTOR_GPUS:-6}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-$((NUM_GPUS - ACTOR_GPUS))}"
export TP_SIZE="${TP_SIZE:-1}"

# Checkpointing.
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
# Agent budget: 900 keeps step latency down; raise to 1200 if too many soft timeouts.
export SWE_AGENT_TIME_BUDGET_SEC="${SWE_AGENT_TIME_BUDGET_SEC:-900}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-300}"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found on PATH" >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: docker daemon not reachable (docker info failed)" >&2
  exit 1
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
echo "  ACTOR_GPUS=${ACTOR_GPUS} ROLLOUT_GPUS=${ROLLOUT_GPUS} TP_SIZE=${TP_SIZE}"
echo "  SAVE_INTERVAL=${SAVE_INTERVAL}"
echo "  SWE_AGENT_TIME_BUDGET_SEC=${SWE_AGENT_TIME_BUDGET_SEC} SWE_EVAL_TIMEOUT_SEC=${SWE_EVAL_TIMEOUT_SEC}"
echo "  SLIME_AGENT_DOCKER_RUN_TIMEOUT_SEC=${SLIME_AGENT_DOCKER_RUN_TIMEOUT_SEC}"
echo "======================================================================"

exec bash "${SCRIPT_DIR}/run_qwen35_4b_swe_1node_async.sh"
