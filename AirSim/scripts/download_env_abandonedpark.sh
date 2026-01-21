#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="$BASE_DIR/envs"
DEST="$ENV_ROOT/AbandonedPark"
ZIP="$ENV_ROOT/AbandonedPark.zip"
URL="https://github.com/microsoft/AirSim/releases/download/v1.8.1/AbandonedPark.zip"

if [[ -x "$DEST/LinuxNoEditor/AbandonedPark.sh" ]]; then
  echo "OK: AbandonedPark already present: $DEST"
  exit 0
fi

mkdir -p "$ENV_ROOT"

if [[ ! -f "$ZIP" ]]; then
  echo "Downloading AbandonedPark.zip ..."
  curl -L --fail --retry 3 --retry-delay 2 -o "$ZIP" "$URL"
else
  echo "Using existing zip: $ZIP"
fi

echo "Extracting..."
rm -rf "$DEST"
unzip -q "$ZIP" -d "$ENV_ROOT"

# The zip usually contains AbandonedPark/AbandonedPark/LinuxNoEditor/AbandonedPark.sh
if [[ ! -d "$DEST" && -d "$ENV_ROOT/AbandonedPark" ]]; then
  DEST="$ENV_ROOT/AbandonedPark"
fi

BIN_CANDIDATE="$ENV_ROOT/AbandonedPark/LinuxNoEditor/AbandonedPark.sh"
if [[ -f "$BIN_CANDIDATE" ]]; then
  chmod +x "$BIN_CANDIDATE" || true
else
  # Best-effort: locate any LinuxNoEditor *.sh launcher.
  BIN_CANDIDATE="$(find "$ENV_ROOT/AbandonedPark" -maxdepth 5 -type f -name '*.sh' | head -n 1 || true)"
  if [[ -n "$BIN_CANDIDATE" && -f "$BIN_CANDIDATE" ]]; then
    chmod +x "$BIN_CANDIDATE" || true
  fi
fi

cat >"$ENV_ROOT/AbandonedPark/UPSTREAM.md" <<EOF
# AbandonedPark environment (binary)

- Source: microsoft/AirSim release assets
- Version: v1.8.1
- Downloaded from: $URL
- Downloaded at (UTC): $(date -u '+%Y-%m-%d %H:%M:%S')

Note: this binary environment is large and should not be committed to git.
EOF

echo "OK: AbandonedPark ready."
