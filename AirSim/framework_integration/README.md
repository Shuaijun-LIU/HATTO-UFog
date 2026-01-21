# AirSim ↔ Framework integration

This folder provides **thin adapters and wrappers** to connect:

- **Framework outputs** (e.g., `Framework/runs/.../timeseries.parquet`)
- to **AirSim** for replay / tracking **with dual-view video capture**

Design goals:
- Keep `HATTO-UFog/AirSim/scripts/*` behavior unchanged (this folder wraps them when needed).
- Make the data contract explicit (columns/units/coordinate mapping).
- Prefer **Framework** as the source-of-truth for paper metrics (energy/delay/queue/comm/resource allocation), and use AirSim primarily for rendering + collision/visibility signals.

## What you get

### 1) Replay (pose-driven; deterministic video)

- Script: `scripts/replay_framework_mainline_plus.py`
- Mode: ExternalPhysicsEngine + `simSetVehiclePose`
- Output: runpack folder under `AirSim/runs_airsim/` with:
  - `artifacts/video.mp4` (FPV|Chase)
  - `timeseries.parquet` (AirSim NED pose + selected paper metrics, if present in Framework timeseries)
  - `summary.json`

The replay wrapper will **propagate key paper metrics** from the Framework row if available:
`S`, `E_total/E_mov/E_tr/E_comp`, `D_total/D_tr/D_comp/D_q/D_uavq`, and any `viol_*` flags.

### 2) Track (AirSim physics; feasibility check)

- Script: `scripts/track_framework_auxline.py`
- Mode: AirSim multirotor dynamics, velocity+altitude commands
- Output: runpack folder under `AirSim/runs_airsim/` with:
  - `artifacts/video.mp4` (FPV|Chase)
  - `timeseries.parquet` (desired vs actual + the same propagated metric fields from Framework timeseries)
  - `summary.json`

Note: the propagated metrics are the **reference metrics from Framework** (i.e., tied to Framework decisions/model).
If you need “metrics under AirSim dynamics”, you must recompute metrics using the actual tracked state and a clearly-defined network/queue/alloc contract.

## Static checks

- `scripts/validate_inputs.py` validates `timeseries.parquet` columns and basic sanity (no AirSim connection required).

