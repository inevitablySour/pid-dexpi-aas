#!/usr/bin/env bash
# Share the app from this PC with a public https link (free Cloudflare quick tunnel, no
# account), behind a password. Ctrl+C stops both.
#
#   ./scripts/serve_public.sh                      # random password, printed below the link
#   APP_PASSWORD='something' ./scripts/serve_public.sh
#   PORT=8601 ./scripts/serve_public.sh
#
# The app listens on 127.0.0.1 only - the tunnel is the one way in. Model settings are
# hidden (APP_PUBLIC=1) and image mode uses the GPU only when it has room (a training run
# keeps it), else the CPU. The link changes every time this script starts. Uploaded files
# stay on this PC (the system temp folder).
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8601}"
CLOUDFLARED="tools/cloudflared"

if [ ! -x "$CLOUDFLARED" ]; then
  if command -v cloudflared >/dev/null; then
    CLOUDFLARED="$(command -v cloudflared)"
  elif [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
    echo "Downloading cloudflared (once)..."
    mkdir -p tools
    curl -fsSL -o tools/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x tools/cloudflared
  else
    echo "Install cloudflared first (macOS: brew install cloudflared; Windows: winget install Cloudflare.cloudflared)."
    exit 1
  fi
fi

if [ -z "${APP_PASSWORD:-}" ]; then
  APP_PASSWORD="$(python3 -c 'import secrets; print(secrets.token_urlsafe(9))')"
fi
mkdir -p "$HOME/.streamlit"
[ -f "$HOME/.streamlit/credentials.toml" ] || printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"

LOG="$(mktemp -d)"
cleanup() { kill "${APP_PID:-}" "${TUNNEL_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

APP_PUBLIC=1 APP_PASSWORD="$APP_PASSWORD" .venv/bin/streamlit run app.py \
  --server.address 127.0.0.1 --server.port "$PORT" --server.headless true \
  --client.toolbarMode viewer --browser.gatherUsageStats false --server.maxUploadSize 50 \
  > "$LOG/app.log" 2>&1 &
APP_PID=$!

"$CLOUDFLARED" tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" > "$LOG/tunnel.log" 2>&1 &
TUNNEL_PID=$!

URL=""
for _ in $(seq 60); do
  URL="$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$LOG/tunnel.log" | head -1 || true)"
  [ -n "$URL" ] && break
  kill -0 "$TUNNEL_PID" 2>/dev/null || { cat "$LOG/tunnel.log"; exit 1; }
  sleep 1
done
[ -n "$URL" ] || { echo "No tunnel link after 60 s:"; cat "$LOG/tunnel.log"; exit 1; }

cat <<EOF

  P&ID to DEXPI converter is online
    link:      $URL
    password:  $APP_PASSWORD

  Send both to the testers. The link stops working when you press Ctrl+C or the PC sleeps,
  and changes on the next start. Logs: $LOG
EOF
wait "$APP_PID"
