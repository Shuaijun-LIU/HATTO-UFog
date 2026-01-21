#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="$ROOT_DIR/refs/external_repos.lock.json"
OUT_DIR="${1:-$ROOT_DIR/external}"

python - <<'PY' "$LOCK_FILE" "$OUT_DIR"
import json
import subprocess
import sys
from pathlib import Path

lock_path = Path(sys.argv[1]).resolve()
out_dir = Path(sys.argv[2]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

lock = json.loads(lock_path.read_text(encoding="utf-8"))
repos = lock.get("repos", [])
if not repos:
    raise SystemExit("No repos found in lock file.")

for repo in repos:
    name = repo["name"]
    url = repo["url"]
    commit = repo["commit"]
    dest = out_dir / name
    if dest.exists():
        print(f"[skip] {name}: {dest} exists")
        continue
    print(f"[clone] {name} -> {dest}")
    subprocess.check_call(["git", "clone", "--no-tags", "--depth", "1", url, str(dest)])
    subprocess.check_call(["git", "fetch", "--depth", "1", "origin", commit], cwd=str(dest))
    subprocess.check_call(["git", "checkout", "--detach", commit], cwd=str(dest))
    print(f"[ok] {name} pinned to {commit}")
PY

echo "[done] External repos in: $OUT_DIR"

