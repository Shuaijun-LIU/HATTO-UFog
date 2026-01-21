from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge.import_airsim import import_airsim


def main() -> int:
    logging.getLogger("tornado.general").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Validate AirSim RPC connection + basic camera access.")
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--timeout_s", type=int, default=10)
    parser.add_argument("--vehicle", default="Drone1")
    parser.add_argument("--rpc_only", action="store_true", help="Only validate RPC + basic calls; skip image capture smoke test.")
    parser.add_argument("--save_dir", default="", help="Optional: write one RGB frame per camera here.")
    args = parser.parse_args()

    airsim = import_airsim()
    client = airsim.VehicleClient(ip=args.ip, port=args.port, timeout_value=int(args.timeout_s))
    try:
        client.confirmConnection()
    except Exception as e:
        print("")
        print("ERROR: Failed to connect to AirSim RPC server.")
        print(f"- ip={args.ip} port={args.port} timeout_s={args.timeout_s}")
        print("- Make sure AirSim is running (e.g., Blocks), and RPC is enabled on this port.")
        print(f"- Under Linux, AirSim reads settings from: ~/Documents/AirSim/settings.json")
        print(f"  (our templates are in: {Path(__file__).resolve().parents[1] / 'configs/airsim_settings'})")
        print("")
        print(f"Exception: {type(e).__name__}: {e}")
        return 2

    vehicles = client.listVehicles()
    print("Vehicles:", vehicles)

    pose = client.simGetVehiclePose(vehicle_name=args.vehicle)
    print("Pose:", pose)

    if bool(args.rpc_only):
        print("OK (rpc_only)")
        return 0

    # Camera smoke test (expects camera "0" and "1" attached to the vehicle).
    requests = [
        airsim.ImageRequest("0", airsim.ImageType.Scene, pixels_as_float=False, compress=True),
        airsim.ImageRequest("1", airsim.ImageType.Scene, pixels_as_float=False, compress=True),
    ]
    responses = client.simGetImages(requests, vehicle_name=args.vehicle)
    if len(responses) != 2:
        raise RuntimeError(f"Expected 2 image responses, got {len(responses)}")

    if args.save_dir:
        out_dir = Path(args.save_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        airsim.write_file(str(out_dir / "cam0.png"), responses[0].image_data_uint8)
        airsim.write_file(str(out_dir / "cam1.png"), responses[1].image_data_uint8)
        print("Wrote:", out_dir / "cam0.png")
        print("Wrote:", out_dir / "cam1.png")

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
