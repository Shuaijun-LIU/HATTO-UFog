#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_DIR="${1:-$BASE_DIR/runs_airsim}"
WIDTH="${GIF_WIDTH:-960}"
FPS="${GIF_FPS:-12}"
MAX_S="${GIF_MAX_S:-10}"

if [[ ! -d "$RUNS_DIR" ]]; then
  echo "ERROR: runs dir not found: $RUNS_DIR" >&2
  exit 2
fi

count=0
while IFS= read -r -d '' mp4; do
  run_dir="$(dirname "$(dirname "$mp4")")"
  gif="$run_dir/artifacts/video.gif"
  if [[ -f "$gif" ]]; then
    continue
  fi
  echo "GIF: $gif"
  python "$BASE_DIR/scripts/make_gif.py" --input "$mp4" --output "$gif" --width "$WIDTH" --fps "$FPS" --max_s "$MAX_S"
  count=$((count + 1))
done < <(find "$RUNS_DIR" -type f -path "*/artifacts/video.mp4" -print0)

echo "OK: generated $count gif(s)."

