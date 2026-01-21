from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class FwToNed:
    """Match the mapping used by `scripts/replay_framework_mainline.py`.

    Framework coordinates:
      - x,y: planar meters
      - z: z-up meters

    AirSim coordinates:
      - NED, z-down meters
    """

    scale_xy: float = 1.0
    scale_z: float = 1.0
    x_offset_ned: float = 0.0
    y_offset_ned: float = 0.0
    z_offset_ned: float = 0.0
    z_up_m: Optional[float] = None  # if set, force constant altitude (ignore Framework z)

    def map_point(
        self,
        *,
        x_fw: float,
        y_fw: float,
        z_fw: float,
        x_first: float,
        y_first: float,
        z_first: float,
        x_anchor: float,
        y_anchor: float,
        z_anchor: float,
    ) -> Tuple[float, float, float]:
        x_ned = float(x_anchor) + (float(x_fw) - float(x_first)) * float(self.scale_xy) + float(self.x_offset_ned)
        y_ned = float(y_anchor) + (float(y_fw) - float(y_first)) * float(self.scale_xy) + float(self.y_offset_ned)
        if self.z_up_m is not None:
            z_ned = -float(self.z_up_m) + float(self.z_offset_ned)
        else:
            z_ned = float(z_anchor) - (float(z_fw) - float(z_first)) * float(self.scale_z) + float(self.z_offset_ned)
        return float(x_ned), float(y_ned), float(z_ned)


def yaw_rad_to_deg(yaw_rad: float) -> float:
    return float(yaw_rad) * 180.0 / math.pi


def wrap_deg(deg: float) -> float:
    d = float(deg)
    while d <= -180.0:
        d += 360.0
    while d > 180.0:
        d -= 360.0
    return d

