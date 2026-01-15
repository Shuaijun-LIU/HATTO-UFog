# HATTO-UFog Framework

Python framework for UAV-assisted fog computing experiments with Parquet + JSON outputs.

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
```

Outputs:
- `runs/<run_id>/timeseries.parquet`
- `runs/<run_id>/config.json`
- `runs/<run_id>/summary.json`
- `runs/<run_id>/world.json`

Parameter storage:
- `configs/*.yaml` (editable defaults and experiment configs)
- `runs/<run_id>/config.json` (resolved config snapshot for each run)
- `configs/default_split.yaml` + `configs/parts/*.yaml` (split configs; loaded via include list)
  - Resource allocation parameters live in `configs/parts/resource.yaml`.
  - MD placement constraints live in `configs/parts/tasks.yaml`.

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

FEAR-PID DDQN training can use attitude-style dynamics:
```yaml
fear_ddqn:
  env_mode: attitude
  axes: 3
```
