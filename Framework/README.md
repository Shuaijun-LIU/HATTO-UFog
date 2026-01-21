# HATTO-UFog Framework

Python experiment framework for HATTO-UFog, including world generation, UAV motion models, communication models, resource allocation baselines, and deterministic logging.

![Showcase preview](example/media/video.webp)

[Example result video (MP4) — short clip.](example/media/video__web.mp4)

What this framework provides:
- World generation (terrain + city), MD placement, and waypoint graph routing (3D obstacle avoidance).
- UAV motion models (kinematic + rigid-body) with attitude/trajectory control (PID / fuzzy / FEAR-PID training entry).
- Communication + channel allocation models (LoS/NLoS, interference options) and delay/queueing models.
- Energy and end-to-end delay accounting (movement + transmission + computation) at timeslot resolution.
- Resource/offloading baselines as **adapters** (run the baseline decision logic inside this framework).
- Deterministic logging: Parquet time-series + JSON snapshots (config/world/summary).

## Directory Structure

```
Framework/
├── ufog_network/       # Core framework code
│   ├── alloc/          # Resource allocation algorithms (PSO, heuristic, random)
│   ├── baselines/      # Baseline methods (ACS, CPS-ACO, GASCA, TD3)
│   ├── control/        # Control algorithms (PID, fuzzy PID, FEAR-PID, DDQN)
│   ├── env/            # Environment generation (world, tasks, queue, wind)
│   ├── sim/            # Simulation core (dynamics, energy, control loop)
│   ├── logging/        # Logging (Parquet, runpack)
│   └── cli.py          # Command-line interface
├── configs/            # Configuration files (default and experiment configs)
│   └── parts/          # Modular configs (resource, comm, control, tasks, etc.)
├── scripts/            # Utility scripts (validation, analysis, training, rendering)
├── showcase/           # WebGL viewer interface
├── runs/               # Experiment run output directory
├── example/            # Example outputs and media assets
└── README.md
```

Key directories:
- `ufog_network/`: Core framework code containing all major functional modules
- `configs/`: YAML configuration files with modular config support (`parts/`)
- `scripts/`: Validation, analysis, and training scripts
- `showcase/`: WebGL visualization interface for previewing generated environments and flight trajectories
- `runs/`: Experiment run result storage (timeseries.parquet, config.json, summary.json)

## Installation

From `HATTO-UFog/Framework`:

```bash
python -m pip install -e .
```

If you prefer not to install the CLI, you can also run:

```bash
python -m ufog_network.cli --help
```

## Quick start

```bash
ufog_network generate-world --output world.json
ufog_network run --config configs/default_split.yaml --output runs
```

Train TD3 or FEAR-PID gain adjuster:
```bash
ufog_network train-td3 --config configs/default_split.yaml --output runs/td3_train
ufog_network train-fear-pid --output runs/fear_pid
```

Validation scripts (no auto-run):
```bash
python scripts/validate_install.py
python scripts/validate_world.py --config configs/default_split.yaml
python scripts/validate_run.py --config configs/default_split.yaml --output runs/validate
python scripts/analyze_md_waypoints.py --config configs/default_split.yaml
python scripts/analyze_md_reachability.py --config configs/default_split.yaml
```

Outputs:
- `runs/<run_id>/timeseries.parquet`
- `runs/<run_id>/config.json`
- `runs/<run_id>/summary.json`
- `runs/<run_id>/world.json`

Example output artifacts (small, committable) are in `example/manifest.json`.

Timeseries target fields:
- `md_target_*` = MD ground position (used for communication/tasks).
- `md_service_*` = MD service/hover point (used for navigation/arrival checks).

Parameter storage:
- `configs/*.yaml` (editable defaults and experiment configs)
- `runs/<run_id>/config.json` (resolved config snapshot for each run)
- `configs/default_split.yaml` + `configs/parts/*.yaml` (split configs; loaded via include list)
  - Resource allocation parameters live in `configs/parts/resource.yaml`.
  - MD placement constraints live in `configs/parts/tasks.yaml`.
  - Service/target scheduling parameters live in `configs/parts/sim.yaml`.

To enable rigid-body dynamics with attitude control, set in config:
```yaml
dynamics:
  mode: rigid_body
control:
  mode: pid
```

To enable PSO resource allocation (task/offload/power/frequency), set:
```yaml
resource:
  mode: pso
```

Random task assignment + uniform power (ablation) can be enabled via:
```yaml
resource:
  mode: random
```

Offload granularity can be configured:
```yaml
resource:
  offload_granularity: md   # or "task"
```

Queueing model (delay) can be switched between analytic M/M/1 and backlog simulation:
```yaml
delay:
  queue_model: mm1   # or "backlog"
```

Aggregated interference model can be enabled via:
```yaml
comm:
  interference_mode: aggregate
  interference_scale: 1.0
```

Wind disturbance can be enabled via:
```yaml
wind:
  enabled: true
  model: steady   # steady | gust | noise
  speed_mean_m_s: 2.0
  direction_deg: 45.0
  accel_gain: 0.2
  apply_to: both  # kinematic | rigid_body | both
  spatial_scale_m: 300.0
  spatial_variation: 0.25
```

Service (MD coverage) and target scheduling controls:
```yaml
sim:
  service_radius_m: 120.0      # horizontal service radius (0 uses reach_threshold)
  service_require_los: false   # optional LoS requirement
  target_policy: shuffle       # shuffle | nearest_unserved
  target_max_steps: 0          # 0 disables forced target switch
  target_skip_steps: 0         # cooldown steps for skipped targets
tasks:
  service_hover_offset_m: 5.0
  md_require_service_waypoint: false
  md_service_waypoint_max_dist_m: 120.0
  md_service_waypoint_require_los: true
```

FEAR-PID DDQN training can use attitude-style dynamics:
```yaml
fear_ddqn:
  env_mode: attitude
  axes: 3
```


