"""Configuration models and load utilities."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class TerrainConfig:
    seed: int = 0
    size_m: float = 1000.0
    max_height_m: float = 220.0
    base_height_m: float = 0.0
    octaves: int = 4
    lacunarity: float = 2.0
    gain: float = 0.5
    ridge_strength: float = 0.6
    valley_strength: float = 0.4
    radial_mountain: bool = True
    radial_scale: float = 1.0
    radial_base: float = 0.3
    radial_gain: float = 0.7
    clearance_m: float = 3.0


@dataclass
class CityConfig:
    enabled: bool = True
    radius_m: float = 350.0
    road_grid_m: float = 60.0
    road_width_m: float = 12.0
    park_radius_m: float = 120.0
    building_density: float = 0.35
    min_height_m: float = 8.0
    max_height_m: float = 90.0
    footprint_min_m: float = 8.0
    footprint_max_m: float = 40.0
    max_buildings: int = 800
    max_attempts_factor: int = 10


@dataclass
class LakeConfig:
    enabled: bool = True
    count: int = 3
    min_radius_m: float = 80.0
    max_radius_m: float = 200.0
    buffer_m: float = 10.0
    placement_scale: float = 0.8


@dataclass
class UncertainObstacleConfig:
    enabled: bool = True
    count: int = 6
    min_radius_m: float = 20.0
    max_radius_m: float = 60.0
    placement_scale: float = 0.9
    min_altitude_offset_m: float = 30.0
    max_altitude_offset_m: float = 120.0


@dataclass
class WorldConfig:
    seed: int = 1234
    size_m: float = 1000.0
    height_m: float = 360.0
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    city: CityConfig = field(default_factory=CityConfig)
    lakes: LakeConfig = field(default_factory=LakeConfig)
    obstacles: UncertainObstacleConfig = field(default_factory=UncertainObstacleConfig)
    waypoint_count: int = 600
    waypoint_altitude_min_m: float = 20.0
    waypoint_altitude_max_m: float = 220.0
    connect_radius_m: float = 120.0
    connect_step_m: float = 5.0
    waypoint_attempts_factor: int = 20
    # Waypoint graph connectivity enforcement (recommended for graph-based baselines).
    waypoint_enforce_connected: bool = True
    waypoint_bridge_k: int = 6
    waypoint_bridge_max_pairs: int = 600
    heightmap_step_m: float = 0.0  # 0 disables heightmap export
    heightmap_extent_m: float = 0.0  # 0 uses full size_m
    segment_cache_enabled: bool = True
    segment_cache_resolution_m: float = 0.1
    segment_cache_max_items: int = 200000


@dataclass
class TaskConfig:
    md_count: int = 50
    md_distribution: str = "fixed"  # fixed | poisson
    md_density: float = 5e-5
    md_min_count: int = 1
    md_require_waypoint: bool = False
    md_waypoint_max_dist_m: float = 80.0
    md_waypoint_require_los: bool = True
    service_hover_offset_m: float = 8.0
    md_require_service_waypoint: bool = True
    md_service_waypoint_max_dist_m: float = 180.0
    md_service_waypoint_require_los: bool = True
    arrival_rate: float = 0.6
    arrival_rate_mode: str = "fixed"  # fixed | sampled
    task_size_dist: str = "exponential"  # exponential | normal
    task_size_mean: float = 5e6
    task_size_std: float = 2e6
    task_size_min_bits: float = 1.0
    task_size_max_bits: float = 1e9
    cycles_dist: str = "normal"  # normal | exponential
    cycles_mean: float = 800.0
    cycles_std: float = 200.0
    cycles_min: float = 1.0
    cycles_max: float = 1e6
    md_height_offset_m: float = 0.1
    position_attempts_factor: int = 50


@dataclass
class CommConfig:
    channel_count: int = 40
    channel_mode: str = "gamma"  # gamma | uniform
    channel_rounding: str = "fractional"  # fractional | floor | round
    channel_min: int = 0
    gamma_alpha: float = 2.0
    gamma_beta: float = 2.0
    bandwidth_hz: float = 1e6
    backhaul_bandwidth_hz: float = 5e6
    carrier_hz: float = 2.4e9
    noise_power: float = 1e-10
    backhaul_noise_power: float = 1e-10
    p_min_w: float = 0.05
    p_max_w: float = 1.0
    power_mode: str = "max"  # max | fixed
    p_fixed_w: float = 0.5
    backhaul_power_w: float = 2.0
    round_trip_factor: float = 2.0
    enable_los: bool = True
    los_model: str = "paper"  # paper | logistic
    path_loss_model: str = "paper"  # paper | log_distance
    los_zeta_clear: float = 1.0
    los_zeta_blocked: float = 5.0
    los_a: float = 9.61
    los_b: float = 0.16
    los_c: float = 0.0
    los_loss_db: float = 1.0
    nlos_loss_db: float = 20.0
    ref_loss_db: float = 30.0
    path_loss_exp: float = 2.2
    los_blocked_factor: float = 0.1
    interference_power: float = 0.0
    interference_mode: str = "fixed"  # fixed | aggregate
    interference_scale: float = 1.0
    light_speed: float = 3.0e8
    # Collision/occlusion check step for LoS/NLoS classification (coarse is ok: LoS is modeled probabilistically).
    los_check_step_m: float = 20.0


@dataclass
class EnergyConfig:
    battery_wh: float = 5000.0
    motor_voltage_v: float = 22.2
    motor_resistance_ohm: float = 0.08
    motor_kv: float = 920.0
    omega_min_rpm: float = 200.0
    omega_max_rpm: float = 5000.0
    delta_u: float = 1.2e-28
    delta_j: float = 3.0e-28
    uav_cpu_hz: float = 2.0e9
    md_cpu_hz: float = 1.0e9
    dc_cpu_hz: float = 5.0e9
    ignore_dc_energy: bool = True


@dataclass
class DelayConfig:
    queue_capacity: float = 5e7
    tau_s: float = 1.0
    queue_eps: float = 1.0
    ignore_dc_compute: bool = True
    queue_model: str = "backlog"  # mm1 | backlog


@dataclass
class SimConfig:
    decision_dt_s: float = 0.1
    control_dt_s: float = 0.01
    steps: int = 600
    stop_when_all_md_visited: bool = False
    service_require_los: bool = False
    epsilon: float = 0.2
    seed: int = 1234
    max_speed_m_s: float = 20.0
    max_accel_m_s2: float = 6.0
    reach_threshold_m: float = 10.0
    hover_power_w: float = 180.0
    move_power_coeff: float = 12.0
    initial_altitude_margin_m: float = 20.0
    recovery_altitude_margin_m: float = 20.0
    reroute_altitude_margin_m: float = 15.0
    flyover_altitude_margin_m: float = 20.0
    # Horizontal service reach radius (coverage-style). This avoids false "unvisited" when altitude is clamped by safety logic.
    service_radius_m: float = 80.0
    target_policy: str = "nearest_unserved"  # shuffle | nearest_unserved
    target_max_steps: int = 0
    target_skip_steps: int = 0
    rl_reward_scale: float = 1.0
    rl_collision_penalty: float = 50.0
    init_pos_attempts: int = 1000


@dataclass
class WindConfig:
    enabled: bool = False
    model: str = "steady"  # steady | gust | noise
    speed_mean_m_s: float = 2.0
    speed_std_m_s: float = 0.5
    direction_deg: float = 0.0
    direction_std_deg: float = 10.0
    gust_period_s: float = 4.0
    gust_scale: float = 0.5
    vertical_scale: float = 0.1
    accel_gain: float = 0.2
    spatial_scale_m: float = 300.0
    spatial_variation: float = 0.2
    direction_variation_deg: float = 8.0
    vertical_shear_strength: float = 0.15
    vertical_shear_ref_m: float = 120.0
    max_speed_m_s: float = 6.0
    seed: int = 1234
    apply_to: str = "both"  # kinematic | rigid_body | both


@dataclass
class OffloadConfig:
    # Note: "heuristic" is a framework policy (not a paper claim) for stable smoke/long runs.
    mode: str = "heuristic"  # uav | md | dc | mixed | heuristic
    mixed_ratio_uav: float = 1.0
    mixed_ratio_md: float = 0.0
    mixed_ratio_dc: float = 0.0
    heuristic_distance_m: float = 450.0
    heuristic_size_bits: float = 1.0e7
    heuristic_dc_size_bits: float = 2.0e7
    heuristic_blocked_to_md: bool = False


@dataclass
class CloudConfig:
    enabled: bool = False
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 500.0


@dataclass
class DynamicsConfig:
    mode: str = "kinematic"  # kinematic | rigid_body
    mass_kg: float = 1.8
    gravity_m_s2: float = 9.81
    arm_length_m: float = 0.25
    kf: float = 1e-5
    km: float = 1e-6
    omega_unit: str = "rpm"  # rpm | rad_s
    omega_min: float = 200.0
    omega_max: float = 5000.0
    max_omega_rad_s: float = 8.0
    inertia_xx: float = 0.03
    inertia_yy: float = 0.03
    inertia_zz: float = 0.06
    linear_drag: float = 0.1
    angular_drag: float = 0.02


@dataclass
class ControlConfig:
    mode: str = "pid"  # pid | fuzzy | fear
    pos_kp: float = 1.2
    pos_ki: float = 0.0
    pos_kd: float = 0.8
    att_kp: float = 6.0
    att_ki: float = 0.0
    att_kd: float = 1.5
    yaw_kp: float = 4.0
    yaw_ki: float = 0.0
    yaw_kd: float = 1.0
    max_tilt_deg: float = 35.0
    vel_damping: float = 0.3
    fear_model_path: str = ""
    fear_scales: List[float] = field(default_factory=lambda: [0.7, 0.9, 1.0, 1.1, 1.3])
    fear_kp_min: float = 0.0
    fear_kp_max: float = 10.0
    fear_ki_min: float = 0.0
    fear_ki_max: float = 5.0
    fear_kd_min: float = 0.0
    fear_kd_max: float = 5.0
    integral_limit: float | None = 5.0
    fear_action_grid_kp: List[float] = field(default_factory=lambda: [-0.2, 0.0, 0.2])
    fear_action_grid_ki: List[float] = field(default_factory=lambda: [-0.05, 0.0, 0.05])
    fear_action_grid_kd: List[float] = field(default_factory=lambda: [-0.1, 0.0, 0.1])
    fear_action_deltas: List[List[float]] = field(default_factory=list)
    fuzzy_labels: List[str] = field(default_factory=lambda: ["NB", "NM", "NS", "ZO", "PS", "PM", "PB"])
    fuzzy_centers: List[float] = field(default_factory=lambda: [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0])
    fuzzy_label_values_kp: List[float] = field(default_factory=lambda: [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    fuzzy_label_values_ki: List[float] = field(default_factory=lambda: [-1.5, -1.0, -1.0, 0.0, 1.0, 1.0, 1.5])
    fuzzy_label_values_kd: List[float] = field(default_factory=lambda: [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    fuzzy_rule_table: List[List[str]] = field(default_factory=list)


@dataclass
class FearDDQNConfig:
    episodes: int = 200
    steps_per_episode: int = 200
    gamma: float = 0.98
    lr: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    target_update: int = 10
    batch_size: int = 64
    buffer_size: int = 20000
    hidden: int = 64
    seed: int = 1234
    device: str = "cpu"
    scales: List[float] = field(default_factory=lambda: [0.7, 0.9, 1.0, 1.1, 1.3])
    action_grid_kp: List[float] = field(default_factory=lambda: [-0.2, 0.0, 0.2])
    action_grid_ki: List[float] = field(default_factory=lambda: [-0.05, 0.0, 0.05])
    action_grid_kd: List[float] = field(default_factory=lambda: [-0.1, 0.0, 0.1])
    action_deltas: List[List[float]] = field(default_factory=list)
    base_kp: float = 1.2
    base_ki: float = 0.0
    base_kd: float = 0.8
    kp_min: float = 0.0
    kp_max: float = 10.0
    ki_min: float = 0.0
    ki_max: float = 5.0
    kd_min: float = 0.0
    kd_max: float = 5.0
    integral_limit: float = 5.0
    u_clip: float = 50.0
    mass: float = 1.0
    error_limit: float = 5.0
    reward_w_e: float = 1.0
    reward_w_de: float = 0.4
    reward_w_du: float = 0.02
    reward_w_stable: float = 1.0
    axes: int = 3
    env_mode: str = "attitude"  # attitude | point
    target_mode: str = "fixed"  # fixed | random
    target_min: float = -1.0
    target_max: float = 1.0
    target_change_interval: int = 50
    disturbance_std: float = 0.0
    measurement_noise_std: float = 0.0
    inertia: List[float] = field(default_factory=lambda: [0.02, 0.02, 0.04])
    angular_damping: float = 0.1
    angle_limit_rad: float = 0.7
    omega_limit: float = 4.0
    dt: float = 0.05
    damping: float = 0.2
    kp: float = 1.0
    kd: float = 0.2
    target: float = 1.0


@dataclass
class PSOConfig:
    particles: int = 30
    iterations: int = 60
    inertia: float = 0.65  # F_I in paper
    c1: float = 2.0  # F_A1
    c2: float = 2.0  # F_A2
    vel_clip: float = 0.2
    tol: float = 1e-4
    no_improve_iters: int = 20
    seed: int = 1234


@dataclass
class ResourceConfig:
    mode: str = "heuristic"  # heuristic | pso | random
    optimize_offload: bool = True
    optimize_power: bool = True
    optimize_freq: bool = True
    optimize_channels: bool = False
    offload_strategy: str = "fractional"  # fractional | hard
    # Heuristic allocator knobs (used when mode=="heuristic")
    heuristic_max_served_mds: int = 1
    heuristic_max_distance_m: float = 600.0
    heuristic_force_nearest: bool = True
    heuristic_uav_fraction: float = 1.0
    random_seed: int = 1234
    random_power_mode: str = "fixed"  # fixed | mid | max
    random_freq_mode: str = "fixed"  # fixed | mid
    offload_granularity: str = "md"  # md | task
    uav_freq_min_hz: float = 1.0e9
    uav_freq_max_hz: float = 3.0e9
    md_freq_min_hz: float = 5.0e8
    md_freq_max_hz: float = 2.0e9
    power_min_w: float = 0.05
    power_max_w: float = 1.0
    pso: PSOConfig = field(default_factory=PSOConfig)


@dataclass
class BaselineConfig:
    name: str = "acs"
    params: Dict[str, Any] = field(default_factory=dict)
    presets: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class Config:
    world: WorldConfig = field(default_factory=WorldConfig)
    tasks: TaskConfig = field(default_factory=TaskConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    delay: DelayConfig = field(default_factory=DelayConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    wind: WindConfig = field(default_factory=WindConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    fear_ddqn: FearDDQNConfig = field(default_factory=FearDDQNConfig)
    resource: ResourceConfig = field(default_factory=ResourceConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Config":
        def _merge(cls, payload):
            if payload is None:
                return cls()
            return cls(**payload)

        def _merge_world(payload):
            if payload is None:
                return WorldConfig()
            wc = WorldConfig()
            for key in [
                "seed",
                "size_m",
                "height_m",
                "waypoint_count",
                "waypoint_altitude_min_m",
                "waypoint_altitude_max_m",
                "connect_radius_m",
                "connect_step_m",
                "waypoint_attempts_factor",
                "waypoint_enforce_connected",
                "waypoint_bridge_k",
                "waypoint_bridge_max_pairs",
                "heightmap_step_m",
                "heightmap_extent_m",
                "segment_cache_enabled",
                "segment_cache_resolution_m",
                "segment_cache_max_items",
            ]:
                if key in payload:
                    setattr(wc, key, payload[key])
            if "terrain" in payload:
                wc.terrain = TerrainConfig(**payload["terrain"])
            if "city" in payload:
                wc.city = CityConfig(**payload["city"])
            if "lakes" in payload:
                wc.lakes = LakeConfig(**payload["lakes"])
            if "obstacles" in payload:
                wc.obstacles = UncertainObstacleConfig(**payload["obstacles"])
            return wc

        def _merge_resource(payload):
            if payload is None:
                return ResourceConfig()
            rc = ResourceConfig()
            for key in [
                "mode",
                "optimize_offload",
                "optimize_power",
                "optimize_freq",
                "optimize_channels",
                "offload_strategy",
                "heuristic_max_served_mds",
                "heuristic_max_distance_m",
                "heuristic_force_nearest",
                "heuristic_uav_fraction",
                "random_seed",
                "random_power_mode",
                "random_freq_mode",
                "offload_granularity",
                "uav_freq_min_hz",
                "uav_freq_max_hz",
                "md_freq_min_hz",
                "md_freq_max_hz",
                "power_min_w",
                "power_max_w",
            ]:
                if key in payload:
                    setattr(rc, key, payload[key])
            if "pso" in payload:
                rc.pso = PSOConfig(**payload["pso"])
            return rc

        cfg = Config(
            world=_merge_world(data.get("world")),
            tasks=_merge(TaskConfig, data.get("tasks")),
            comm=_merge(CommConfig, data.get("comm")),
            energy=_merge(EnergyConfig, data.get("energy")),
            delay=_merge(DelayConfig, data.get("delay")),
            sim=_merge(SimConfig, data.get("sim")),
            wind=_merge(WindConfig, data.get("wind")),
            offload=_merge(OffloadConfig, data.get("offload")),
            cloud=_merge(CloudConfig, data.get("cloud")),
            dynamics=_merge(DynamicsConfig, data.get("dynamics")),
            control=_merge(ControlConfig, data.get("control")),
            fear_ddqn=_merge(FearDDQNConfig, data.get("fear_ddqn")),
            resource=_merge_resource(data.get("resource")),
            baseline=_merge(BaselineConfig, data.get("baseline")),
        )
        return cfg


def _load_config_dict(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if p.suffix in {".yml", ".yaml"}:
        data = yaml.safe_load(p.read_text())
    elif p.suffix == ".json":
        data = json.loads(p.read_text())
    else:
        raise ValueError("Config file must be .json or .yaml")
    if data is None:
        data = {}
    if isinstance(data, dict) and "include" in data:
        include_paths = data.get("include") or []
        if not isinstance(include_paths, list):
            raise ValueError("include must be a list of file paths")
        merged: Dict[str, Any] = {}
        for inc in include_paths:
            inc_path = Path(inc)
            if not inc_path.is_absolute():
                inc_path = p.parent / inc_path
            inc_data = _load_config_dict(str(inc_path))
            merged = _deep_merge(merged, inc_data)
        # Overlay current file (excluding include)
        data = {k: v for k, v in data.items() if k != "include"}
        merged = _deep_merge(merged, data)
        data = merged
    return data


def _coerce_numbers(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _coerce_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numbers(v) for v in obj]
    if isinstance(obj, str):
        try:
            val = float(obj)
            return val
        except ValueError:
            return obj
    return obj


def load_config(path: Optional[str]) -> Config:
    if not path:
        return Config()
    data = _load_config_dict(path)
    data = _coerce_numbers(data)
    return Config.from_dict(data)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def save_config(cfg: Config, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg.to_dict(), indent=2))
