#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="$BASE_DIR/envs"
DEST="$ENV_ROOT/LandscapeMountains"
ZIP="$ENV_ROOT/LandscapeMountains.zip"
URL="https://github.com/microsoft/AirSim/releases/download/v1.8.1/LandscapeMountains.zip"

if [[ -x "$DEST/LinuxNoEditor/LandscapeMountains.sh" ]]; then
  echo "OK: LandscapeMountains already present: $DEST"
  exit 0
fi

mkdir -p "$ENV_ROOT"

if [[ -f "$ZIP" ]] && unzip -tq "$ZIP" >/dev/null 2>&1; then
  echo "OK: zip already present and valid: $ZIP"
else
  echo "Downloading LandscapeMountains.zip (resume supported) ..."
  curl -L --fail --http1.1 --retry 3 --retry-delay 2 -C - -o "$ZIP" "$URL"

  if ! unzip -tq "$ZIP" >/dev/null 2>&1; then
    echo "WARNING: zip integrity check failed; re-downloading from scratch..."
    rm -f "$ZIP"
    curl -L --fail --http1.1 --retry 3 --retry-delay 2 -o "$ZIP" "$URL"
  fi
fi

echo "Extracting..."
rm -rf "$DEST"
unzip -q "$ZIP" -d "$ENV_ROOT"

BIN_CANDIDATE="$DEST/LinuxNoEditor/LandscapeMountains.sh"
if [[ -f "$BIN_CANDIDATE" ]]; then
  chmod +x "$BIN_CANDIDATE" || true
else
  BIN_CANDIDATE="$(find "$DEST" -maxdepth 6 -type f -name '*.sh' | head -n 1 || true)"
  if [[ -n "$BIN_CANDIDATE" && -f "$BIN_CANDIDATE" ]]; then
    chmod +x "$BIN_CANDIDATE" || true
  fi
fi

cat >"$DEST/UPSTREAM.md" <<EOF
# LandscapeMountains environment (binary)

- Source: microsoft/AirSim release assets
- Version: v1.8.1
- Downloaded from: $URL
- Downloaded at (UTC): $(date -u '+%Y-%m-%d %H:%M:%S')

Note: this binary environment is large and should not be committed to git.
EOF

echo "OK: LandscapeMountains ready."
