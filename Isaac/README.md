# Isaac Sim experiments

This folder contains **[Isaac Sim](https://developer.nvidia.com/isaac-sim)** experiment scripts for HATTO-UFog runs, including high-fidelity dynamics simulation runners, framework integration scripts, and runpack management tools.

![Showcase preview](example/video.webp)

[Example result video (MP4) — short preview clip.](example/video__web.mp4)

## Prerequisites

### System / GPU
- NVIDIA GPU recommended (tested on multi-GPU Linux).
- Headless runs use Vulkan; if running on a remote server, ensure your driver/runtime supports headless Vulkan/EGL.
- Internet access recommended for showcase mode (we load official Omniverse USD stages via public HTTPS URLs; no assets are committed).

### Isaac Sim install
- Install Isaac Sim separately (we do not ship it in this repo).
- Set:

```bash
export ISAACSIM_ROOT=/path/to/isaacsim
```

### Python environments (important)
There are **two** Python contexts:
- **Isaac Sim Python** (required for simulation): run via `$ISAACSIM_ROOT/python.sh ...`
- **Normal Python** (recommended for postprocess): used for `postprocess_video.py` / `postprocess_timeseries.py`

Install postprocess deps once (from repo root):

```bash
python -m pip install -r requirements.txt
```

Minimal install (postprocess-only; if you don’t need the Framework package):

```bash
python -m pip install numpy opencv-python pandas pyarrow pyyaml
```

## Directory Structure

```
Isaac/
├── bridge/                 # runpack + video helpers
├── configs/                # Isaac Sim experiment configs
├── demo_assets/            # small in-repo demo USD assets (OK to commit)
├── example/                # curated media only (the only Isaac media committed)
├── scripts/                # runners / validators / postprocess
├── framework_integration/  # Framework → Isaac replay integration
├── external/               # external resources (gitignored)
├── runs_isaac/             # run outputs (gitignored)
└── README.md
```

Key directories:
- `bridge/`: run directory bookkeeping + reusable video helpers
- `scripts/`: experiment runners, validation, and postprocessing
- `framework_integration/`: replay Framework outputs in Isaac Sim (isolated integration layer)
- `configs/`: configs (baselines, showcase, disturbance sweeps)
- `external/`: external repos/assets (not committed; fetched locally)
- `runs_isaac/`: all run outputs (not committed)

## Quick start

1) Check GPU/CPU availability (single GPU, prefer higher index):

```bash
python scripts/check_resources.py --show-top
```

2) Validate the installation:

```bash
$ISAACSIM_ROOT/python.sh scripts/validate_isaac_install.py
```

3) Run a minimal hover smoke test (writes `timeseries.jsonl`):

```bash
$ISAACSIM_ROOT/python.sh scripts/run_isaac_experiment.py --config configs/isaac_base.yaml --output runs_isaac --name hover_smoke
```

Notes:
- A frozen baseline (pre-upgrade) runner is kept as: `scripts/run_isaac_experiment_v1_minimal.py`
- `scripts/run_isaac_experiment.py` supports `control.mode = open_loop_hover | pid_hover` and disturbance knobs (wind/latency/measurement noise).

4) Capture demo (best-effort in-sim frames; always can postprocess a fallback trajectory video):

```bash
$ISAACSIM_ROOT/python.sh scripts/run_isaac_experiment.py --config configs/isaac_capture_demo.yaml --output runs_isaac --name capture_demo
# run postprocess with normal python (not python.sh)
python -m pip install -r ../requirements.txt
python scripts/postprocess_video.py --run_dir runs_isaac/<run_id>__<slug>
```

5) Showcase-quality render capture (M1.5 target: UHD screenshot + short MP4; env overview + UAV in-frame):
Official Omniverse content stage (public HTTPS USD) + repo-shipped demo UAV asset (no local assets root required; no redistribution):

```bash
$ISAACSIM_ROOT/python.sh scripts/run_isaac_showcase.py --config configs/isaac_showcase_rivermark_outdoor_demo_debug.yaml --output runs_isaac --name showcase_rivermark_outdoor
# run postprocess with normal python (not python.sh)
python -m pip install -r ../requirements.txt
python scripts/postprocess_video.py --run_dir runs_isaac/<run_id>__<slug>
```

Best run on this machine (reference only; `runs_isaac/` is gitignored):
- `runs_isaac/rivermark_demo_uav_pbr_debug_v4_20260120_082602Z__rivermark_demo_uav_pbr_debug_v4`

Note:
- `configs/*showcase*.yaml` supports `scene.stage_usd`:
  - If empty: build a procedural debug scene (not recommended for showcase-quality materials).
  - If set: `open_stage()` that USD and skip procedural env creation.
    - `http(s)://...` and `omniverse://...` are supported (recommended for official content).
    - `/Isaac/...` requires a working `get_assets_root_path()` (local Nucleus/asset packs).

6) NVIDIA asset fallback (requires assets/Nucleus; does not rely on our controller):

```bash
$ISAACSIM_ROOT/python.sh scripts/validate_quadcopter_asset.py
$ISAACSIM_ROOT/python.sh scripts/run_nvidia_quadcopter_smoke.py --config configs/isaac_nvidia_quadcopter_smoke.yaml --output runs_isaac --name nvidia_smoke
```

7) (Optional) Convert JSONL → Parquet (run with normal Python, not `python.sh`):

```bash
python -m pip install -r ../requirements.txt
python scripts/postprocess_timeseries.py --input runs_isaac/<run_id>__<slug>/timeseries.jsonl --output runs_isaac/<run_id>__<slug>/timeseries.parquet
```

## Notes

- This repo does not ship Isaac Sim. Install Isaac Sim separately and run scripts via `$ISAACSIM_ROOT/python.sh`.
- Do not commit local run outputs or downloaded assets:
  - `Isaac/runs_isaac/`, `Isaac/assets/`, `Isaac/external/` are expected to be ignored by git.
- Media policy (for GitHub):
  - All `Isaac/**/*.png|mp4|gif|webp` are ignored **except** curated files under `Isaac/example/`.
- Optional helper: find official standalone examples in your local Isaac install:
  - `$ISAACSIM_ROOT/python.sh scripts/find_isaac_examples.py`

## Framework integration (planning + replay)

This repository includes an **isolated** integration layer under `framework_integration/` for replaying `HATTO-UFog/Framework` outputs in Isaac Sim.

Design rule:
- All integration logic lives under `HATTO-UFog/Isaac/framework_integration/` and does **not** change the default behavior of `HATTO-UFog/Isaac/scripts/*.py`.

0) Validate inputs (no Isaac needed):

```bash
python framework_integration/scripts/validate_inputs.py --framework_timeseries ../Framework/runs/<...>/timeseries.parquet
python framework_integration/scripts/validate_inputs.py --trajectory_json ../Framework/runs/<...>/trajectory.json --require_trajectory
```

1) Create a “plan pack” (generates `planned_*.sh` scripts; does not execute):

```bash
python framework_integration/scripts/plan_end2end.py --framework_timeseries ../Framework/runs/<...>/timeseries.parquet --name fw2isaac_replay
```

2) Replay a `trajectory.json` in Isaac (pose-driven visualization):

```bash
$ISAACSIM_ROOT/python.sh framework_integration/scripts/replay_framework_trajectory.py --trajectory_json <trajectory.json> --output runs_isaac --name fw_replay --headless
```

3) Build MP4/screenshot from captured frames (or fallback to trajectory video):

```bash
python -m pip install -r ../requirements.txt
python scripts/postprocess_video.py --run_dir runs_isaac/<run_id>__<slug>
```
