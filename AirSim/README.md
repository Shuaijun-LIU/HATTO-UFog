# AirSim experiments

This folder contains **[Microsoft AirSim](https://microsoft.github.io/AirSim/)** experiment scripts for HATTO-UFog runs, including flight planning, video generation, mainline/auxline experiment runners, and framework integration scripts.

![Showcase preview](example/AirSimNH/video.webp)
![Showcase preview](example/AbandonedPark/video.webp)
![Showcase preview](example/LandscapeMountains/video.webp)
![Showcase preview](example/Blocks/video.webp)

Example result videos (MP4, click to preview): [AirSimNH](example/AirSimNH/video__web.mp4) · [Blocks](example/Blocks/video__web.mp4) · [AbandonedPark](example/AbandonedPark/video__web.mp4) · [LandscapeMountains](example/LandscapeMountains/video__web.mp4)

Key constraints:
- AirSim Python client is vendored under `vendor/airsim/` (no dependency on local `UAV/ref/*` after open-source).
- We implement **two lines**:
  - **Mainline**: ExternalPhysicsEngine (pose driven via `simSetVehiclePose`)
  - **Auxline**: AirSim built-in multirotor dynamics (velocity commands)
- Each run outputs exactly **one MP4**: `artifacts/video.mp4` (split-screen `FPV | Chase`).

## Prerequisites

### System
- Linux (tested) or Windows (paths/launchers differ).
- Recommended on Linux:
  - `curl` + `unzip` (used by `scripts/download_env_*.sh`)
  - `lsof` (used by `scripts/run_showcase_pack.sh` to detect/clear `127.0.0.1:41451`)
  - `nvidia-smi` (optional; used to auto-pick an idle GPU for Unreal `-graphicsadapter=<idx>`)

Example (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y curl unzip lsof
```

### Python
- Use a **dedicated venv/conda env** (AirSim Python client depends on `msgpack-rpc-python` which pins older Tornado).
- Install deps from repo root (recommended; includes AirSim + Isaac postprocess deps):

```bash
python -m pip install -r requirements.txt
```

Minimal install (AirSim-only; if you don’t need the Framework package):

```bash
python -m pip install \
  msgpack-rpc-python==0.4.1 msgpack-python==0.5.6 backports.ssl_match_hostname==3.7.0.1 \
  numpy pandas pyarrow opencv-python imageio imageio-ffmpeg pyyaml
```

## Directory Structure

```
AirSim/
├── bridge/              # Core bridge modules (flight planning, video generation, runpack management)
├── configs/            # AirSim configuration files (mainline/auxline settings)
├── envs/               # Simulation environments (AirSimNH, Blocks, AbandonedPark, etc.)
├── example/            # Demo assets (videos/GIFs/screenshots)
├── framework_integration/  # Framework integration layer (trajectory replay, validation)
├── runs_airsim/        # Experiment run output directory
├── scripts/            # Experiment scripts (run mainline/auxline, env downloads, validation)
├── vendor/             # Third-party dependencies (AirSim Python client)
└── README.md
```

Key directories:
- `bridge/`: Core functionality for AirSim interaction (flight planning, video rendering, runpack management)
- `scripts/`: Experiment execution scripts and environment download scripts
- `framework_integration/`: Replay and validation of Framework outputs in AirSim
- `configs/`: AirSim settings files (mainline/auxline mode configurations)
- `runs_airsim/`: Storage directory for all experiment run results

## Quick start

0) Install Python deps (recommended: dedicated venv/conda env):

```bash
# from repo root
python -m pip install -r requirements.txt
```

1) Launch AirSim.

   Recommended:
   - **AirSimNH** for showcase-quality visuals.
   - **Blocks** for quick smoke tests.
   - See **Environments** section below for environment download scripts.

   Important (Linux):
   - In this environment, the GUI/windowed startup path got stuck before the map loads
     (log stops near `LogMoviePlayer: Initializing movie player`), and **RPC never opened**.
   - Launching with Unreal's **offscreen rendering** fixed it: `-RenderOffScreen`.

   Example (paths relative to project root `HATTO-UFog/`; yours may differ):
   - `HATTO-UFog/AirSim/envs/AirSimNH/AirSimNH/LinuxNoEditor/AirSimNH.sh -settings=$HOME/Documents/AirSim/settings.json -windowed -ResX=640 -ResY=480 -NoSound -RenderOffScreen`
   - `HATTO-UFog/AirSim/envs/Blocks/LinuxBlocks1.8.1/LinuxNoEditor/Blocks.sh -settings=$HOME/Documents/AirSim/settings.json -windowed -ResX=640 -ResY=480 -NoSound -RenderOffScreen`

   Sanity check:
   - AirSim RPC should listen on `127.0.0.1:41451`.
   - You may also see a listener on `:1985` (Unreal messaging/backchannel). That is **not** the AirSim API port.

2) Run connection check:
   - `python scripts/validate_airsim_connection.py --save_dir /tmp/airsim_check`
3) Run **mainline** (requires restarting AirSim with ExternalPhysicsEngine):
   - `python scripts/run_mainline.py`
   - For higher-res demo assets, use: `--settings_template configs/airsim_settings/settings_mainline_showcase.json`
4) Restart AirSim, then run **auxline**:
   - `python scripts/run_auxline.py`
   - For higher-res demo assets, use: `--settings_template configs/airsim_settings/settings_auxline_showcase.json`

Optional (Framework replay → AirSim mainline video):
- `python scripts/replay_framework_mainline.py --framework_timeseries ../Framework/runs/<...>/timeseries.parquet`

## Framework integration validation
This repo also includes a **validation integration layer** under `framework_integration/` that uses **AirSim as the environment** to validate Framework outputs (trajectory feasibility + optional heading/attitude visualization).

Design rules:
- Do not change behavior of existing `scripts/*.py` (use wrappers here).
- Keep artifacts reproducible via runpack metadata (`runs_airsim/*/meta.json`, etc.).
- Treat coordinate mapping (Framework z-up ↔ AirSim NED) as a first-class contract.

0) (Optional) Validate Framework artifact statically (no AirSim needed):
   - `python framework_integration/scripts/validate_inputs.py --framework_timeseries ../Framework/runs/<...>/timeseries.parquet --mode replay`
   - `python framework_integration/scripts/validate_inputs.py --framework_timeseries ../Framework/runs/<...>/timeseries.parquet --mode track`

1) Create a “plan pack” (generates `planned_*.sh` scripts):
   - `python framework_integration/scripts/plan_end2end.py --framework_timeseries ../Framework/runs/<...>/timeseries.parquet --plan_output_root plans_airsim --name fw2airsim_plan`
   - Output example: `plans_airsim/<run_id>__fw2airsim_plan/` with `planned_*.sh`.

2) Mainline replay with richer overlay (ExternalPhysicsEngine required; restart AirSim):
   - `python framework_integration/scripts/replay_framework_mainline_plus.py --framework_timeseries ../Framework/runs/<...>/timeseries.parquet --overlay --use_yaw`
   - Optional: also apply roll/pitch/yaw if present: add `--use_rpy`

3) Auxline tracking feasibility (AirSim physics; normal multirotor mode; restart AirSim):
   - `python framework_integration/scripts/track_framework_auxline.py --framework_timeseries ../Framework/runs/<...>/timeseries.parquet --overlay --speed_m_s 3.0 --yaw_mode face_path`
   - Dry-run (no RPC): `python framework_integration/scripts/track_framework_auxline.py --framework_timeseries ../Framework/runs/<...>/timeseries.parquet --dry_run`

One-click (CPU/GPU check + prefer highest-index idle GPU + showcase pack):
- `bash scripts/run_showcase_pack.sh`
  - Adds overlay + visible user marker by default (see `scripts/run_showcase_pack.sh`).

Outputs:
- `runs_airsim/<run_id>__<slug>/artifacts/video.mp4`
- `runs_airsim/<run_id>__<slug>/artifacts/video.gif` (preview)
- `runs_airsim/<run_id>__<slug>/timeseries.parquet`
- `runs_airsim/index.jsonl`

Notes:
- Scripts write `~/Documents/AirSim/settings.json` from `configs/airsim_settings/*.json`.
  AirSim reads settings at startup, so you must restart AirSim after switching between mainline/auxline.
- `framework_integration/scripts/*` propagate key paper metric columns (`E_*`, `D_*`, `S`, and optional `viol_*`) from the Framework timeseries when available.


## Environments

This project **does not** commit Unreal binary environments to git. Instead, download them locally into `AirSim/envs/`.

### Quick start

From `HATTO-UFog/AirSim`:

```bash
bash scripts/download_env_airsimnh.sh
bash scripts/download_env_blocks.sh
bash scripts/download_env_abandonedpark.sh
bash scripts/download_env_landscapemountains.sh
```

Then run the demo pack:

```bash
bash scripts/run_showcase_pack.sh --env airsimnh
```

### Included scenes

#### AirSimNH (Neighborhood)
- Download: `bash scripts/download_env_airsimnh.sh`
- Launcher: `envs/AirSimNH/AirSimNH/LinuxNoEditor/AirSimNH.sh`

#### Blocks
- Download: `bash scripts/download_env_blocks.sh`
- Launcher: `envs/Blocks/LinuxBlocks1.8.1/LinuxNoEditor/Blocks.sh`

#### AbandonedPark
- Download: `bash scripts/download_env_abandonedpark.sh`
- Launcher: `envs/AbandonedPark/LinuxNoEditor/AbandonedPark.sh`

#### LandscapeMountains
- Download: `bash scripts/download_env_landscapemountains.sh`
- Launcher: `envs/LandscapeMountains/LinuxNoEditor/LandscapeMountains.sh`
