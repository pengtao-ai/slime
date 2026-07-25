#!/usr/bin/env bash
# Expose local Anthropic adapter (ADAPTER_PORT, default 18001) via a Cloudflare
# quick tunnel so public E2B sandboxes can dial back.
#
# Usage:
#   bash examples/coding_agent_rl/start_adapter_tunnel.sh
#   export ADAPTER_PUBLIC_URL=https://....trycloudflare.com
#   bash examples/coding_agent_rl/run_qwen35_4b_swe_1node.sh
#
# Notes:
# - Adapter only listens after training starts the Anthropic adapter thread.
#   It is fine (and recommended) to start the tunnel *before* training.
# - GET / often returns 404; the real health check is GET /v1/models -> 200.

set -euo pipefail

PORT="${ADAPTER_PORT:-18001}"
LOCAL_URL="http://127.0.0.1:${PORT}"
BIN_DIR="${CLOUDFLARED_BIN_DIR:-/tmp/bin}"
LOG="${ADAPTER_TUNNEL_LOG:-/tmp/adapter_cloudflared.log}"
PID_FILE="${ADAPTER_TUNNEL_PID:-/tmp/adapter_cloudflared.pid}"
CF="${BIN_DIR}/cloudflared"

mkdir -p "${BIN_DIR}"

if [[ ! -x "${CF}" ]]; then
  echo "[tunnel] downloading cloudflared -> ${CF}"
  curl -fsSL --retry 3 -o "${CF}" \
    "https://github.com/cloudflare/cloudflared/releases/download/2025.11.1/cloudflared-linux-amd64"
  chmod +x "${CF}"
fi

local_code="$(curl -sS -m 3 -o /dev/null -w '%{http_code}' "${LOCAL_URL}/v1/models" || true)"
if [[ "${local_code}" != "200" ]]; then
  echo "[tunnel] note: local adapter ${LOCAL_URL}/v1/models -> HTTP ${local_code:-down}"
  echo "[tunnel]        (normal before training; tunnel can stay up and will work once adapter binds)"
fi

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "[tunnel] already running pid=$(cat "${PID_FILE}") log=${LOG}"
else
  : > "${LOG}"
  nohup "${CF}" tunnel --url "${LOCAL_URL}" --no-autoupdate >"${LOG}" 2>&1 &
  echo $! > "${PID_FILE}"
  echo "[tunnel] started pid=$(cat "${PID_FILE}")"
fi

URL=""
for _ in $(seq 1 60); do
  URL="$(grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' "${LOG}" | tail -1 || true)"
  if [[ -n "${URL}" ]]; then
    break
  fi
  sleep 0.5
done

if [[ -z "${URL}" ]]; then
  echo "[tunnel] failed to discover public URL; see ${LOG}" >&2
  tail -n 40 "${LOG}" >&2 || true
  exit 1
fi

echo "ADAPTER_PUBLIC_URL=${URL}"
echo "export ADAPTER_PUBLIC_URL=${URL}"

# Probe the Anthropic-compatible route (NOT / — that often 404s by design).
pub_code="$(curl -sS -m 15 -o /dev/null -w '%{http_code}' "${URL}/v1/models" || true)"
echo "[tunnel] public probe ${URL}/v1/models -> HTTP ${pub_code:-down}"
if [[ "${pub_code}" == "200" ]]; then
  echo "[tunnel] OK — sandboxes can use this URL"
elif [[ "${local_code}" != "200" ]]; then
  echo "[tunnel] waiting for local adapter; re-run this script after training starts to confirm 200"
else
  echo "[tunnel] unexpected public status; check ${LOG}" >&2
fi
