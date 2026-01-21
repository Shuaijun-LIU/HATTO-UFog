from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge.import_airsim import import_airsim


def main() -> int:
    logging.getLogger("tornado.general").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(
        description="Validate the basic z-up (Framework) <-> NED (AirSim) position transform by a set/get pose roundtrip."
    )
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--timeout_s", type=int, default=10)
    parser.add_argument("--vehicle", default="Drone1")
    parser.add_argument("--x", type=float, default=3.0)
    parser.add_argument("--y", type=float, default=2.0)
    parser.add_argument("--z_up", type=float, default=5.0, help="Framework z-up meters (AirSim will use z_ned=-z_up).")
    args = parser.parse_args()

    airsim = import_airsim()
    client = airsim.VehicleClient(ip=args.ip, port=args.port, timeout_value=int(args.timeout_s))
    try:
        client.confirmConnection()
    except Exception as e:
        print("")
        print("ERROR: Failed to connect to AirSim RPC server. Start AirSim first, then re-run.")
        print("")
        print(f"Exception: {type(e).__name__}: {e}")
        return 2

    z_ned = -float(args.z_up)
    pose = client.simGetVehiclePose(vehicle_name=args.vehicle)
    pose.position = airsim.Vector3r(float(args.x), float(args.y), float(z_ned))
    pose.orientation = airsim.to_quaternion(0.0, 0.0, 0.0)
    client.simSetVehiclePose(pose, True, vehicle_name=args.vehicle)

    pose2 = client.simGetVehiclePose(vehicle_name=args.vehicle)
    dx = float(pose2.position.x_val) - float(args.x)
    dy = float(pose2.position.y_val) - float(args.y)
    dz = float(pose2.position.z_val) - float(z_ned)
    err = math.sqrt(dx * dx + dy * dy + dz * dz)

    print("Set  (NED):", (float(args.x), float(args.y), float(z_ned)))
    print("Get  (NED):", (float(pose2.position.x_val), float(pose2.position.y_val), float(pose2.position.z_val)))
    print("Error (m): ", err)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
