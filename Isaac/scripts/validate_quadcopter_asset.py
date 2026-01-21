from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate that a quadcopter USD can be resolved and referenced.")
    parser.add_argument(
        "--usd",
        default="",
        help="Explicit USD path. If empty, uses get_assets_root_path() + default relative quadcopter path.",
    )
    parser.add_argument(
        "--relative",
        default="/Isaac/Robots/IsaacSim/Quadcopter/quadcopter.usd",
        help="Relative asset path under assets root (used when --usd is empty).",
    )
    parser.add_argument("--prim", default="/World/UAV", help="Prim path to reference the asset into.")
    args = parser.parse_args()

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.utils.stage import add_reference_to_stage

        usd_path = args.usd.strip()
        if not usd_path:
            from isaacsim.storage.native import get_assets_root_path

            assets_root = get_assets_root_path()
            if not assets_root:
                print("[error] get_assets_root_path() returned empty. Assets/Nucleus may not be set up.", file=sys.stderr)
                return 2
            usd_path = assets_root + args.relative

        sim = SimulationContext()
        add_reference_to_stage(usd_path, args.prim)
        sim.initialize_physics()
        sim.play()
        sim.step(render=False)
        print(f"[ok] Referenced USD: {usd_path}")
        return 0
    except Exception as exc:
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
