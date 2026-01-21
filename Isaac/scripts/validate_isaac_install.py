from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


def _infer_isaac_root() -> Optional[Path]:
    env = os.environ.get("ISAACSIM_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    try:
        import isaacsim

        # .../isaacsim/python_packages/isaacsim/__init__.py -> isaacsim root
        return Path(isaacsim.__file__).resolve().parents[2]
    except Exception:
        return None


def _try_read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Isaac Sim installation.")
    parser.add_argument("--start_app", action="store_true", help="Also start a headless SimulationApp (slow).")
    args = parser.parse_args()

    isaac_root = _infer_isaac_root()
    if isaac_root is None:
        print(
            "[error] Could not infer ISAACSIM_ROOT. Set env var and run with $ISAACSIM_ROOT/python.sh.",
            file=sys.stderr,
        )
        return 2

    version = _try_read_text(isaac_root / "VERSION")
    license_exists = (isaac_root / "LICENSE.txt").exists()
    python_sh_exists = (isaac_root / "python.sh").exists()

    print(f"ISAACSIM_ROOT={isaac_root}")
    print(f"VERSION={version}")
    print(f"python.sh={'yes' if python_sh_exists else 'no'}")
    print(f"LICENSE.txt={'yes' if license_exists else 'no'}")

    try:
        import isaacsim  # noqa: F401
        from isaacsim import SimulationApp  # noqa: F401
    except Exception as exc:
        print(f"[error] Import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    if not args.start_app:
        print("[ok] Imports look good. (Use --start_app for a full headless startup test.)")
        return 0

    # Full startup (slow): required before importing isaacsim.core.* modules.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        from isaacsim.core.api import World

        world = World(backend="numpy", stage_units_in_meters=1.0)
        world.reset()
        world.stop()
        print("[ok] SimulationApp started and World.reset() succeeded.")
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

