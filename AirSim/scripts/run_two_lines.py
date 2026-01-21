from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run both AirSim lines (mainline + auxline) and produce exactly 2 MP4 outputs (one per line)."
    )
    parser.add_argument("--output_root", default="runs_airsim")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--speed_m_s", type=float, default=2.0)
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--vehicle", default="Drone1")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    scripts_dir = base_dir / "scripts"

    cmd_main = [
        sys.executable,
        str(scripts_dir / "run_mainline.py"),
        "--output_root",
        args.output_root,
        "--steps",
        str(args.steps),
        "--dt",
        str(args.dt),
        "--fps",
        str(args.fps),
        "--speed_m_s",
        str(args.speed_m_s),
        "--ip",
        args.ip,
        "--port",
        str(args.port),
        "--vehicle",
        args.vehicle,
    ]
    print("=== Mainline (ExternalPhysicsEngine) ===")
    print("Command:", " ".join(cmd_main))
    subprocess.check_call(cmd_main, cwd=str(base_dir))

    print("")
    print("=== Switch AirSim settings ===")
    print("Now restart AirSim with auxline settings (no ExternalPhysicsEngine).")
    print("Then press Enter to run auxline...")
    try:
        input()
    except KeyboardInterrupt:
        return 1

    cmd_aux = [
        sys.executable,
        str(scripts_dir / "run_auxline.py"),
        "--output_root",
        args.output_root,
        "--steps",
        str(args.steps),
        "--dt",
        str(args.dt),
        "--fps",
        str(args.fps),
        "--speed_m_s",
        str(args.speed_m_s),
        "--ip",
        args.ip,
        "--port",
        str(args.port),
        "--vehicle",
        args.vehicle,
    ]
    print("=== Auxline (AirSim dynamics) ===")
    print("Command:", " ".join(cmd_aux))
    subprocess.check_call(cmd_aux, cwd=str(base_dir))

    print("")
    print("OK: produced 2 runs (mainline + auxline), each with `artifacts/video.mp4` (split-screen FPV|chase).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
