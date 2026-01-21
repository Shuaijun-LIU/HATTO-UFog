from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List


def _iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def main() -> int:
    parser = argparse.ArgumentParser(description="Locate useful Isaac Sim standalone examples in a local install.")
    parser.add_argument("--root", default="", help="Isaac Sim root. Defaults to $ISAACSIM_ROOT.")
    parser.add_argument(
        "--keywords",
        default="quadcopter,camera,replicator,px4",
        help="Comma-separated keywords to search for in paths.",
    )
    args = parser.parse_args()

    root = args.root.strip() or os.environ.get("ISAACSIM_ROOT", "")
    if not root:
        raise SystemExit("Missing Isaac Sim root. Set ISAACSIM_ROOT or pass --root.")

    isaac_root = Path(root).expanduser().resolve()
    examples_dir = isaac_root / "standalone_examples"
    if not examples_dir.exists():
        raise SystemExit(f"standalone_examples not found under: {isaac_root}")

    keywords = [k.strip().lower() for k in str(args.keywords).split(",") if k.strip()]
    hits: List[str] = []
    for p in _iter_files(examples_dir):
        rel = str(p.relative_to(isaac_root))
        lower = rel.lower()
        if any(k in lower for k in keywords):
            hits.append(rel)

    if not hits:
        print("[no hits] searched:", examples_dir)
        return 1

    for rel in sorted(hits):
        print(rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

