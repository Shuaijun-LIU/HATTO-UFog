from __future__ import annotations

import json
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now_compact() -> str:
    # Example: 20260116_014233Z
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def slugify(text: str) -> str:
    safe = []
    for ch in text.strip():
        if ch.isalnum():
            safe.append(ch.lower())
        elif ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("_")
    slug = "".join(safe).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "run"


def make_run_id(prefix: str) -> str:
    return f"{slugify(prefix)}_{_utc_now_compact()}"


def _find_git_root(start: Path) -> Optional[Path]:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / ".git").exists():
            return p
    return None


def _git_info(start_dir: Path) -> Dict[str, Any]:
    root = _find_git_root(start_dir)
    if root is None:
        return {"git_root": None, "git_head": None, "git_dirty": None}
    try:
        head = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        head = None
    try:
        dirty = (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=str(root), stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        dirty_flag = bool(dirty)
    except Exception:
        dirty_flag = None
    return {"git_root": str(root), "git_head": head, "git_dirty": dirty_flag}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def pip_freeze() -> str:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.STDOUT).decode()
        return out
    except Exception as e:
        return f"# pip freeze failed: {type(e).__name__}: {e}\n"


def try_read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def infer_isaac_root() -> Optional[Path]:
    env = os.environ.get("ISAACSIM_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    try:
        import isaacsim

        # .../isaacsim/python_packages/isaacsim/__init__.py -> isaacsim root
        return Path(isaacsim.__file__).resolve().parents[2]
    except Exception:
        return None


@dataclass(frozen=True)
class RunContext:
    run_id: str
    slug: str
    run_dir: Path


def prepare_run(
    *,
    output_root: Path,
    name: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> RunContext:
    run_id = make_run_id(name)
    slug = slugify(name)
    run_dir = (output_root / f"{run_id}__{slug}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    cmd = shlex.join(sys.argv)
    isaac_root_path = infer_isaac_root()
    isaac_root = str(isaac_root_path) if isaac_root_path else None
    isaac_version = try_read_text(isaac_root_path / "VERSION") if isaac_root_path else None
    meta = {
        "run_id": run_id,
        "name": name,
        "cmd": cmd,
        "cwd": str(Path.cwd()),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "env": {
            # keep minimal to avoid leaking secrets
            "ISAACSIM_ROOT": isaac_root,
            "ISAACSIM_VERSION": isaac_version,
        },
        **_git_info(Path.cwd()),
    }
    if extra_meta:
        meta["extra"] = extra_meta

    write_json(run_dir / "meta.json", meta)
    write_text(run_dir / "run.sh", cmd + "\n")
    write_text(run_dir / "env_pip_freeze.txt", pip_freeze())
    append_jsonl(output_root / "index.jsonl", meta)
    return RunContext(run_id=run_id, slug=slug, run_dir=run_dir)
