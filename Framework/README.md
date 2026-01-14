# HATTO-UFog Framework

Python framework for UAV-assisted fog computing experiments with Parquet + JSON outputs.

## Quick start

```bash
ufog generate-world --output world.json
ufog run --config configs/default_split.yaml --output runs
```

Train TD3 or FEAR-PID gain adjuster:
```bash
ufog train-td3 --config configs/default_split.yaml --output runs/td3_train
ufog train-fear-pid --output runs/fear_pid
```

Validation scripts (no auto-run):
```bash
python scripts/validate_install.py
python scripts/validate_world.py --config configs/default_split.yaml
python scripts/validate_run.py --config configs/default_split.yaml --output runs/validate
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

FEAR-PID DDQN training can use attitude-style dynamics:
```yaml
fear_ddqn:
  env_mode: attitude
  axes: 3
```
