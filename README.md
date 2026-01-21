<style>
  .logo-light { display: inline-block; }
  .logo-dark { display: none; }
  @media (prefers-color-scheme: dark) {
    .logo-light { display: none; }
    .logo-dark { display: inline-block; }
  }
</style>

<div align="center">
  <h1 style="display: inline-flex; align-items: center; gap: 8px; margin: 0;">
    <img src="docs/static/images/logo.png" alt="HATTO-UFog logo" width="40" class="logo-light" style="vertical-align: middle;">
    <img src="docs/static/images/logo_dark.png" alt="HATTO-UFog logo" width="40" class="logo-dark" style="vertical-align: middle;">
    HATTO-UFog: Holistic Attitude–Trajectory–Task Optimization for UAV-Assisted Fog Computing
  </h1>
</div>

<a href="https://arxiv.org/abs/2407.14894"><img src="https://img.shields.io/badge/arXiv-2407.14894-b31b1b.svg" alt="arXiv"></a> <a href="https://shuaijun-liu.github.io/HATTO-UFog/"><img src="https://img.shields.io/badge/Website-HATTO--UFog-6366F1.svg" alt="Website"></a> <a href="https://shuaijun-liu.github.io/UAV-Assisted-Fog-Computing-Simulation-Demo"><img src="https://img.shields.io/badge/Demo-Web--based-FF6B35.svg" alt="Demo"></a> <a href="https://microsoft.github.io/AirSim/"><img src="https://img.shields.io/badge/AirSim-Microsoft-95E1D3.svg" alt="AirSim"></a> <a href="https://developer.nvidia.com/isaac"><img src="https://img.shields.io/badge/NVIDIA%20Isaac-Simulation-76B900.svg" alt="NVIDIA Isaac"></a> ![](https://img.shields.io/badge/PRs-Welcome-blue)


HATTO-UFog is a UAV-assisted fog computing system that **jointly optimizes**: (1). Attitude control (2). Trajectory planning (3). Resource allocation (4). Task assignment/offloading to reduce end-to-end **latency** and **energy consumption** in a terrain-aware 3D environment.

This folder contains the code for the paper "Energy-Aware Holistic Optimization in UAV-Assisted Fog Computing: Attitude, Trajectory, and Task Assignment".

Project website: https://shuaijun-liu.github.io/HATTO-UFog/.

Online demo: https://shuaijun-liu.github.io/UAV-Assisted-Fog-Computing-Simulation-Demo.

<small>

> **Note:** The demo is a lightweight, browser-based (Three.js) interactive visualization created for the paper to help reviewers quickly understand the simulation environment in a simplified setting.

</small>


## Table of Contents

- [Modules](#modules)
- [Showcases](#showcases)
- [Installation](#installation)
- [Repository structure](#repository-structure)
- [Quickstart](#quickstart)
- [Tested environment (reference)](#tested-environment-reference)
- [Online demo gallery (small)](#online-demo-gallery-small)
- [License](#license)

## Modules

- **Framework (Phase 1)**: paper-aligned Python simulation framework for trajectory planning + control + communication + resource allocation + task/offloading, with deterministic Parquet+JSON logging.
- **AirSim (Phase 2)**: AirSim-based validation and video capture for Framework outputs (mainline/auxline). Included Unreal environments: **AirSimNH**, **Blocks**, **AbandonedPark**, **LandscapeMountains**.
- **Isaac Sim (Phase 3)**: higher-fidelity dynamics and disturbance robustness experiments, plus Framework→Isaac replay integration and postprocess tooling.

## Showcases

### AirSim 

![](AirSim/example/AirSimNH/video.webp)
![](AirSim/example/AbandonedPark/video.webp)
![](AirSim/example/LandscapeMountains/video.webp)
![](AirSim/example/Blocks/video.webp)

Example result videos (MP4): [AirSimNH](AirSim/example/AirSimNH/video__web.mp4) · [Blocks](AirSim/example/Blocks/video__web.mp4) · [AbandonedPark](AirSim/example/AbandonedPark/video__web.mp4) · [LandscapeMountains](AirSim/example/LandscapeMountains/video__web.mp4)

### Isaac Sim

![](Isaac/example/video.webp)

[Example result video (MP4).](Isaac/example/video__web.mp4)

### Framework showcase assets (web-based)
![](Framework/example/media/video.webp)

[Example result video (MP4).](Framework/example/media/video__web.mp4)

## Installation

### Option A: pip

From `HATTO-UFog/`:

```bash
python -m pip install -r requirements.txt
```

If you use the showcase renderer (Playwright):

```bash
python -m playwright install chromium
```

### Option B: conda (YAML env)

From `HATTO-UFog/`:

```bash
conda env create -f environment.yaml
conda activate hatto-ufog
python -m pip install -r requirements.txt
```

Python 3.13 reference env (toolchain-aligned):

```bash
conda env create -f environment-python313.yaml
conda activate hatto-ufog-py313
python -m pip install -r requirements.txt
```

## Repository structure

- `Framework/`: Python experiment framework (Parquet + JSON outputs)
  - Trajectory planner adapters: `acs`, `acs_ds`, `cps_aco`, `ga_sca`, `td3`
  - Control/dynamics: PID / Fuzzy / FEAR-PID, kinematic / rigid-body
- `AirSim/`: AirSim-based experiments
- `Isaac/`: Isaac Sim experiments
- `docs/`: project-page assets (GitHub Pages)

Module docs:
- Framework: `Framework/README.md`
- AirSim: `AirSim/README.md`
- Isaac: `Isaac/README.md`

Environment boundary: Framework / AirSim / Isaac have different dependencies and launch modes; follow each module README instead of assuming a single end-to-end command.

## Quickstart

Framework (from `HATTO-UFog/Framework`):

```bash
python -m pip install -e .
ufog_network run --config configs/smoke.yaml --output runs
```

Framework (from `HATTO-UFog/`):

```bash
python -m pip install -r requirements.txt
ufog_network run --config Framework/configs/smoke.yaml --output Framework/runs
```

## Tested environment (reference)

These version pins are a reference baseline for reproducing the reported runs and the curated media artifacts.

- Isaac Sim
  - Isaac Sim version: `5.1.0-rc.19+release.26219.9c81211b.gl`
  - Isaac built-in Python: `3.11.13` (`$ISAACSIM_ROOT/python.sh`)
  - Official stage asset prefix: `Assets/Isaac/5.1/...`
- AirSim
  - Vendored AirSim PythonClient (`AirSim/vendor/airsim`) upstream commit: `13448700ec2b36d6aad7a4e0909bc9daf9d3d931` (`microsoft/AirSim`)
  - Binary environments: AirSim release `v1.8.1` (e.g., `AirSimNH.zip`, `Blocks.zip`)
- GPU / Driver
  - NVIDIA driver: `580.119.02`
  - GPU: `NVIDIA RTX 5880 Ada Generation`
- Python
  - Python packages: `numpy 2.4.1`, `pandas 2.3.3`, `pyarrow 22.0.0`, `opencv-python 4.11.0`, `PyYAML 6.0.3`, `imageio 2.37.2`, `imageio-ffmpeg 0.6.0`, `playwright 1.49.1`, `msgpack-rpc-python 0.4.1`


## Online demo gallery (small)

![](docs/static/images/path_side_by_side__small.gif)
![](docs/static/images/atc_side_by_side__small.gif)

## License

MIT License: see [`LICENSE`](LICENSE).

