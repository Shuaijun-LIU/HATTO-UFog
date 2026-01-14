"""Energy and delay metrics based on the paper's model (simplified)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import math
import numpy as np

from framework.config import CommConfig, DelayConfig, EnergyConfig, SimConfig, OffloadConfig, CloudConfig
from framework.alloc.base import AllocationDecision
from framework.env.world import World
from framework.env.queue import QueueState
from framework.env.tasks import Task


@dataclass
class Metrics:
    E_total: float
    E_mov: float
    E_tr: float
    E_comp: float
    D_total: float
    D_tr: float
    D_comp: float
    D_q: float
    D_uavq: float


def gamma_channel_allocation(task_sizes: Dict[int, float], cfg: CommConfig) -> Dict[int, float]:
    # P_j(t) = (beta^alpha/(alpha-1)!) s^{alpha-1} exp(-beta s)
    alpha = cfg.gamma_alpha
    beta = cfg.gamma_beta
    factorial = math.factorial(int(alpha - 1)) if alpha >= 1 else 1
    probs = {}
    for j, s in task_sizes.items():
        p = (beta ** alpha) / factorial * (s ** (alpha - 1)) * math.exp(-beta * s)
        probs[j] = max(p, 1e-12)
    total = sum(probs.values())
    if total <= 0:
        if not task_sizes:
            return {}
        uniform = cfg.channel_count / max(1, len(task_sizes))
        return {j: uniform for j in task_sizes}
    # Use effective (fractional) channel allocation by default
    alloc = {j: cfg.channel_count * probs[j] / total for j in task_sizes}
    return alloc


def uniform_channel_allocation(task_sizes: Dict[int, float], cfg: CommConfig) -> Dict[int, float]:
    if not task_sizes:
        return {}
    per = cfg.channel_count / max(1, len(task_sizes))
    return {j: per for j in task_sizes}


def round_channels(raw: Dict[int, float], cfg: CommConfig) -> Dict[int, float]:
    if cfg.channel_rounding == "fractional":
        return raw
    if not raw:
        return {}
    # Round or floor with sum constraint
    items = sorted(raw.items(), key=lambda x: x[1], reverse=True)
    alloc = {}
    for j, val in items:
        if cfg.channel_rounding == "round":
            alloc[j] = int(round(val))
        else:
            alloc[j] = int(math.floor(val))
        if alloc[j] < cfg.channel_min:
            alloc[j] = cfg.channel_min
    total = sum(alloc.values())
    # Adjust to fit channel_count
    if total > cfg.channel_count:
        # Remove from largest allocations first
        for j, _val in items:
            if total <= cfg.channel_count:
                break
            if alloc[j] > cfg.channel_min:
                alloc[j] -= 1
                total -= 1
    elif total < cfg.channel_count:
        # Add to largest fractional remainders
        remainders = sorted(((j, raw[j] - math.floor(raw[j])) for j in raw), key=lambda x: x[1], reverse=True)
        idx = 0
        while total < cfg.channel_count and remainders:
            j, _rem = remainders[idx % len(remainders)]
            alloc[j] += 1
            total += 1
            idx += 1
    return {j: float(v) for j, v in alloc.items()}


def los_probability(theta_deg: float, blocked: bool, cfg: CommConfig) -> float:
    if cfg.los_model == "paper":
        zeta = cfg.los_zeta_blocked if blocked else cfg.los_zeta_clear
        p = 1.0 / (1.0 + zeta * math.exp(-(theta_deg - zeta)))
        return max(0.0, min(1.0, p))
    # Logistic LoS probability model (urban macro inspired)
    p = 1.0 / (1.0 + cfg.los_a * math.exp(-cfg.los_b * (theta_deg - cfg.los_c)))
    if blocked:
        p *= cfg.los_blocked_factor
    return max(0.0, min(1.0, p))


def path_loss_linear(distance: float, p_los: float, cfg: CommConfig) -> float:
    d = max(1.0, distance)
    if cfg.path_loss_model == "paper":
        # Paper form: L = P_LoS * ((4π f_c / c) * d)
        factor = (4.0 * math.pi * cfg.carrier_hz / cfg.light_speed) * d
        return max(1e-12, p_los * factor)
    # Log-distance loss in dB with excess losses
    loss_db = cfg.ref_loss_db + 10.0 * cfg.path_loss_exp * math.log10(d)
    excess_db = p_los * cfg.los_loss_db + (1.0 - p_los) * cfg.nlos_loss_db
    total_db = loss_db + excess_db
    return 10.0 ** (total_db / 10.0)


def transmission_rate(
    distance: float,
    theta_deg: float,
    blocked: bool,
    cfg: CommConfig,
    bandwidth_hz: float,
    power_w: float,
    noise_power: float,
    interference_power: float | None = None,
) -> float:
    if distance <= 0:
        distance = 1.0
    if cfg.enable_los:
        p_los = los_probability(theta_deg, blocked, cfg)
    else:
        p_los = 1.0
    path_loss = path_loss_linear(distance, p_los, cfg)
    interference = cfg.interference_power if interference_power is None else interference_power
    sinr = power_w / (path_loss * (interference + noise_power + 1e-12))
    return bandwidth_hz * math.log2(1.0 + sinr)


def _power_for_md(cfg: CommConfig) -> float:
    if cfg.power_mode == "fixed":
        return cfg.p_fixed_w
    return cfg.p_max_w


def _offload_fractions(cfg: OffloadConfig) -> Tuple[float, float, float]:
    if cfg.mode == "md":
        return 0.0, 1.0, 0.0
    if cfg.mode == "dc":
        return 0.0, 0.0, 1.0
    if cfg.mode == "mixed":
        total = max(1e-9, cfg.mixed_ratio_uav + cfg.mixed_ratio_md + cfg.mixed_ratio_dc)
        return cfg.mixed_ratio_uav / total, cfg.mixed_ratio_md / total, cfg.mixed_ratio_dc / total
    return 1.0, 0.0, 0.0


def _offload_heuristic(
    total_size: float,
    distance: float,
    blocked: bool,
    cfg: OffloadConfig,
    cloud_cfg: CloudConfig | None,
) -> Tuple[float, float, float]:
    # Heuristic policy: large tasks or long distance favor MD/DC; blocked link favors MD
    if blocked and cfg.heuristic_blocked_to_md:
        return 0.0, 1.0, 0.0
    if cloud_cfg is not None and cloud_cfg.enabled and total_size >= cfg.heuristic_dc_size_bits:
        return 0.0, 0.0, 1.0
    if distance >= cfg.heuristic_distance_m or total_size >= cfg.heuristic_size_bits:
        return 0.0, 1.0, 0.0
    return 1.0, 0.0, 0.0


def compute_metrics(
    tasks: List[Task],
    md_positions: List[Tuple[float, float, float]],
    uav_pos: Tuple[float, float, float],
    energy_cfg: EnergyConfig,
    comm_cfg: CommConfig,
    delay_cfg: DelayConfig,
    sim_cfg: SimConfig,
    E_mov: float,
    world: World | None = None,
    offload_cfg: OffloadConfig | None = None,
    cloud_cfg: CloudConfig | None = None,
    arrival_rate: float | Dict[int, float] | None = None,
    alloc: AllocationDecision | None = None,
    queue_state: QueueState | None = None,
    offload_granularity: str = "md",
) -> Metrics:
    if not tasks:
        return Metrics(E_total=E_mov, E_mov=E_mov, E_tr=0.0, E_comp=0.0, D_total=0.0, D_tr=0.0, D_comp=0.0, D_q=0.0, D_uavq=0.0)

    # Aggregate tasks per MD
    size_sum: Dict[int, float] = {}
    cycles_sum: Dict[int, float] = {}
    count_sum: Dict[int, int] = {}
    tasks_by_md: Dict[int, List[Task]] = {}
    for t in tasks:
        size_sum[t.md_id] = size_sum.get(t.md_id, 0.0) + t.size_bits
        cycles_sum[t.md_id] = cycles_sum.get(t.md_id, 0.0) + t.size_bits * t.cycles
        count_sum[t.md_id] = count_sum.get(t.md_id, 0) + 1
        tasks_by_md.setdefault(t.md_id, []).append(t)

    if alloc is not None and alloc.channel_alloc:
        channels = alloc.channel_alloc
    else:
        if comm_cfg.channel_mode == "uniform":
            channels_raw = uniform_channel_allocation(size_sum, comm_cfg)
        else:
            channels_raw = gamma_channel_allocation(size_sum, comm_cfg)
        channels = round_channels(channels_raw, comm_cfg)
    off_cfg = offload_cfg or OffloadConfig()

    E_tr = 0.0
    D_tr = 0.0
    D_comp = 0.0
    D_q = 0.0
    D_uavq = 0.0
    E_comp = 0.0
    use_backlog = delay_cfg.queue_model == "backlog" and queue_state is not None
    L = sim_cfg.decision_dt_s

    if arrival_rate is None:
        arrival_rate = 0.0
    # Pre-compute UAV queue terms
    lam_uav_sum = 0.0
    work_uav_sum = 0.0
    total_uav_tasks = 0.0

    total_uav_cycles = 0.0
    link_meta: Dict[int, Dict[str, float]] = {}
    for j, total_size in size_sum.items():
        md_x, md_y, md_z = md_positions[j]
        dx = uav_pos[0] - md_x
        dy = uav_pos[1] - md_y
        dz = uav_pos[2] - md_z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        horiz = math.sqrt(dx * dx + dy * dy)
        theta = math.degrees(math.atan2(abs(dz), max(1e-6, horiz)))
        blocked = False
        if world is not None and comm_cfg.enable_los:
            blocked = not world.segment_is_free(uav_pos, (md_x, md_y, md_z), step=world.cfg.connect_step_m)
        if comm_cfg.enable_los:
            p_los = los_probability(theta, blocked, comm_cfg)
        else:
            p_los = 1.0
        path_loss = path_loss_linear(dist, p_los, comm_cfg)
        link_meta[j] = {
            "dist": dist,
            "theta": theta,
            "blocked": float(blocked),
            "path_loss": path_loss,
        }

    def _task_split(tasks_j: List[Task], frac_uav: float, frac_md: float, frac_dc: float) -> Tuple[float, float, float, float, float, float, int, int, int]:
        total = sum(t.size_bits for t in tasks_j)
        targets = {
            "uav": total * frac_uav,
            "md": total * frac_md,
            "dc": total * frac_dc,
        }
        sizes = {"uav": 0.0, "md": 0.0, "dc": 0.0}
        cycles = {"uav": 0.0, "md": 0.0, "dc": 0.0}
        counts = {"uav": 0, "md": 0, "dc": 0}
        # Deterministic: sort by size descending
        for task in sorted(tasks_j, key=lambda t: t.size_bits, reverse=True):
            remaining = {k: targets[k] - sizes[k] for k in targets}
            dest = max(remaining, key=lambda k: remaining[k])
            if remaining[dest] <= 0:
                dest = max(targets, key=lambda k: targets[k])
            sizes[dest] += task.size_bits
            cycles[dest] += task.size_bits * task.cycles
            counts[dest] += 1
        return (
            sizes["uav"],
            sizes["md"],
            sizes["dc"],
            cycles["uav"],
            cycles["md"],
            cycles["dc"],
            counts["uav"],
            counts["md"],
            counts["dc"],
        )

    for j, total_size in size_sum.items():
        total_cycles = cycles_sum.get(j, 0.0)
        avg_size = total_size / max(1, count_sum.get(j, 1))
        avg_cycles_per_bit = total_cycles / max(1e-9, total_size)
        meta = link_meta[j]
        dist = meta["dist"]
        theta = meta["theta"]
        blocked = bool(meta["blocked"])

        cj = max(1e-6, float(channels.get(j, 1e-6)))
        power_md = _power_for_md(comm_cfg)
        if alloc is not None and j in alloc.power_w:
            power_md = float(alloc.power_w[j])
        power_md = max(comm_cfg.p_min_w, min(comm_cfg.p_max_w, power_md))
        interference_power = None
        if comm_cfg.interference_mode == "aggregate":
            total_channels = max(1e-6, float(comm_cfg.channel_count))
            inter = 0.0
            for k, meta_k in link_meta.items():
                if k == j:
                    continue
                ck = max(1e-6, float(channels.get(k, 1e-6)))
                overlap = min(cj, ck) / total_channels
                p_k = _power_for_md(comm_cfg)
                if alloc is not None and k in alloc.power_w:
                    p_k = float(alloc.power_w[k])
                p_k = max(comm_cfg.p_min_w, min(comm_cfg.p_max_w, p_k))
                path_loss_k = meta_k["path_loss"]
                inter += overlap * (p_k / max(1e-9, path_loss_k))
            interference_power = comm_cfg.interference_scale * inter
        rate = transmission_rate(
            dist,
            theta,
            blocked,
            comm_cfg,
            comm_cfg.bandwidth_hz,
            power_md,
            comm_cfg.noise_power,
            interference_power=interference_power,
        )
        rt = comm_cfg.round_trip_factor

        if alloc is not None and j in alloc.offload_uav:
            frac_uav = float(alloc.offload_uav.get(j, 0.0))
            frac_md = float(alloc.offload_md.get(j, 0.0))
            frac_dc = float(alloc.offload_dc.get(j, 0.0))
            total_frac = frac_uav + frac_md + frac_dc
            if total_frac > 0:
                frac_uav /= total_frac
                frac_md /= total_frac
                frac_dc /= total_frac
        else:
            if off_cfg.mode == "heuristic":
                frac_uav, frac_md, frac_dc = _offload_heuristic(total_size, dist, blocked, off_cfg, cloud_cfg)
            else:
                frac_uav, frac_md, frac_dc = _offload_fractions(off_cfg)
        if cloud_cfg is None or not cloud_cfg.enabled:
            if frac_dc > 0:
                total_frac = frac_uav + frac_md
                if total_frac > 0:
                    frac_uav /= total_frac
                    frac_md /= total_frac
                else:
                    frac_uav = 1.0
                    frac_md = 0.0
                frac_dc = 0.0

        size_uav = total_size * frac_uav
        size_md = total_size * frac_md
        size_dc = total_size * frac_dc
        cycles_uav = total_cycles * frac_uav
        cycles_md = total_cycles * frac_md
        cycles_dc = total_cycles * frac_dc
        count_uav = int(round(count_sum.get(j, 0) * frac_uav))
        count_md = int(round(count_sum.get(j, 0) * frac_md))
        count_dc = max(0, count_sum.get(j, 0) - count_uav - count_md)
        if offload_granularity == "task":
            tasks_j = tasks_by_md.get(j, [])
            if tasks_j:
                (
                    size_uav,
                    size_md,
                    size_dc,
                    cycles_uav,
                    cycles_md,
                    cycles_dc,
                    count_uav,
                    count_md,
                    count_dc,
                ) = _task_split(tasks_j, frac_uav, frac_md, frac_dc)
        total_uav_cycles += cycles_uav

        # Transmission MD <-> UAV for UAV/DC execution
        size_tx = size_uav + size_dc
        if size_tx > 0:
            D_tr += rt * size_tx / max(1e-9, (cj * rate))
            E_tr += rt * power_md * size_tx / max(1e-9, (cj * rate))

        # Transmission UAV <-> DC for DC execution
        if size_dc > 0 and cloud_cfg is not None and cloud_cfg.enabled:
            dx_c = uav_pos[0] - cloud_cfg.x_m
            dy_c = uav_pos[1] - cloud_cfg.y_m
            dz_c = uav_pos[2] - cloud_cfg.z_m
            dist_c = math.sqrt(dx_c * dx_c + dy_c * dy_c + dz_c * dz_c)
            horiz_c = math.sqrt(dx_c * dx_c + dy_c * dy_c)
            theta_c = math.degrees(math.atan2(abs(dz_c), max(1e-6, horiz_c)))
            rate_c = transmission_rate(dist_c, theta_c, False, comm_cfg, comm_cfg.backhaul_bandwidth_hz, comm_cfg.backhaul_power_w, comm_cfg.backhaul_noise_power)
            D_tr += rt * size_dc / max(1e-9, rate_c)
            E_tr += rt * comm_cfg.backhaul_power_w * size_dc / max(1e-9, rate_c)

        # Computation delay and energy at UAV/MD/DC
        if size_uav > 0:
            f_uav = energy_cfg.uav_cpu_hz
            if alloc is not None and alloc.freq_uav_hz is not None:
                f_uav = float(alloc.freq_uav_hz)
            D_comp += cycles_uav / max(1.0, f_uav)
            E_comp += energy_cfg.delta_u * cycles_uav * (f_uav ** 2)
            lam_j = arrival_rate[j] if isinstance(arrival_rate, dict) else (arrival_rate or 0.0)
            lam_uav_sum += lam_j * frac_uav
            work_uav_sum += lam_j * frac_uav * avg_cycles_per_bit * avg_size
            total_uav_tasks += count_uav
        if size_md > 0:
            f_md = energy_cfg.md_cpu_hz
            if alloc is not None and j in alloc.freq_md_hz:
                f_md = float(alloc.freq_md_hz[j])
            D_comp += cycles_md / max(1.0, f_md)
            E_comp += energy_cfg.delta_j * cycles_md * (f_md ** 2)
        if size_dc > 0 and not delay_cfg.ignore_dc_compute:
            f_dc = energy_cfg.dc_cpu_hz
            D_comp += cycles_dc / max(1.0, f_dc)
        # Queueing
        trans_frac = frac_uav + frac_dc
        if use_backlog:
            backlog = queue_state.md_tx_bits.get(j, 0.0)
            if size_tx > 0:
                D_q += (backlog / max(1e-9, (cj * rate))) * max(1, (count_uav + count_dc))
            service_bits = cj * rate * L
            queue_state.md_tx_bits[j] = max(0.0, backlog + size_tx - service_bits)
        else:
            # Transmission queue delay (M/M/1 approximation)
            lam_j = arrival_rate[j] if isinstance(arrival_rate, dict) else (arrival_rate or 0.0)
            lam_eff = lam_j * trans_frac
            if lam_eff > 0 and avg_size > 0:
                denom = (L * cj * rate - lam_eff * avg_size)
                if denom > 0:
                    d_q = (lam_eff * (avg_size ** 2)) / (L * cj * rate * denom)
                    D_q += d_q * max(1, (count_uav + count_dc))

    # UAV compute queue delay (paper formula or backlog model)
    f_uav = energy_cfg.uav_cpu_hz
    if alloc is not None and alloc.freq_uav_hz is not None:
        f_uav = float(alloc.freq_uav_hz)
    if use_backlog:
        backlog_cyc = queue_state.uav_comp_cycles
        if total_uav_tasks > 0:
            D_uavq += (backlog_cyc / max(1e-9, f_uav)) * total_uav_tasks
        service_cycles = f_uav * L
        queue_state.uav_comp_cycles = max(0.0, backlog_cyc + total_uav_cycles - service_cycles)
    else:
        if lam_uav_sum > 0:
            d_uavq = delay_cfg.queue_capacity / lam_uav_sum - (work_uav_sum) / (delay_cfg.tau_s * f_uav * lam_uav_sum)
            d_uavq = max(0.0, d_uavq)
            D_uavq += d_uavq * total_uav_tasks

    E_total = E_mov + E_tr + E_comp
    D_total = D_tr + D_comp + D_q + D_uavq
    return Metrics(
        E_total=E_total,
        E_mov=E_mov,
        E_tr=E_tr,
        E_comp=E_comp,
        D_total=D_total,
        D_tr=D_tr,
        D_comp=D_comp,
        D_q=D_q,
        D_uavq=D_uavq,
    )
