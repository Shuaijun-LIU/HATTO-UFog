from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


def _run_lines(cmd: List[str], *, timeout_s: float = 2.0) -> List[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=float(timeout_s))
        text = out.decode("utf-8", errors="replace")
        return [ln.strip() for ln in text.splitlines() if ln.strip()]
    except Exception:
        return []


def _parse_gpu_table(lines: List[str]) -> List[Dict[str, int]]:
    rows: List[Dict[str, int]] = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 5:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "util_gpu": int(parts[1]),
                    "util_mem": int(parts[2]),
                    "mem_used": int(parts[3]),
                    "mem_total": int(parts[4]),
                }
            )
        except Exception:
            continue
    return rows


def _gpu_uuid_map() -> Dict[str, int]:
    lines = _run_lines(["nvidia-smi", "--query-gpu=index,gpu_uuid", "--format=csv,noheader,nounits"])
    mapping: Dict[str, int] = {}
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
            uuid = parts[1]
        except Exception:
            continue
        mapping[uuid] = idx
    return mapping


def _busy_gpus_from_apps() -> List[int]:
    uuid_map = _gpu_uuid_map()
    lines = _run_lines(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    busy: List[int] = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if not parts:
            continue
        uuid = parts[0]
        if uuid in uuid_map:
            busy.append(uuid_map[uuid])
    return sorted(set(busy))


def _recommend_gpu(rows: List[Dict[str, int]], *, prefer_high: bool = True) -> Optional[int]:
    if not rows:
        return None
    busy = set(_busy_gpus_from_apps())
    candidates = [
        r
        for r in rows
        if r["util_gpu"] <= 5 and r["mem_used"] <= 1024 and r["index"] not in busy
    ]
    if candidates:
        return max(c["index"] for c in candidates) if prefer_high else min(c["index"] for c in candidates)
    # Fallback: choose least loaded GPU (util + mem), still prefer high index on ties.
    rows_sorted = sorted(
        rows,
        key=lambda r: (r["util_gpu"] + r["util_mem"] + int(r["mem_used"] / 256), r["index"]),
        reverse=prefer_high,
    )
    return rows_sorted[0]["index"] if rows_sorted else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check GPU/CPU availability before running Isaac Sim.")
    parser.add_argument("--prefer-high", action="store_true", default=True, help="Prefer higher-index GPU (default).")
    parser.add_argument("--prefer-low", action="store_true", help="Prefer lower-index GPU instead.")
    parser.add_argument("--show-top", action="store_true", help="Show top CPU processes.")
    args = parser.parse_args()
    if args.prefer_low:
        prefer_high = False
    else:
        prefer_high = bool(args.prefer_high)

    print("[cpu] loadavg:", ", ".join(f"{v:.2f}" for v in os.getloadavg()))
    print("[cpu] cores:", os.cpu_count())
    if args.show_top:
        lines = _run_lines(["ps", "-eo", "pid,pcpu,pmem,comm", "--sort=-pcpu"])
        print("[cpu] top processes:")
        for ln in lines[:10]:
            print("  ", ln)

    if not shutil.which("nvidia-smi"):
        print("[gpu] nvidia-smi not found; cannot query GPUs.")
        return 0

    gpu_lines = _run_lines(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = _parse_gpu_table(gpu_lines)
    if not rows:
        print("[gpu] failed to parse GPU table.")
        return 0

    busy = _busy_gpus_from_apps()
    print("[gpu] summary (index, util_gpu%, util_mem%, mem_used_MB/total_MB):")
    for r in sorted(rows, key=lambda x: x["index"]):
        busy_mark = " *busy" if r["index"] in busy else ""
        print(
            f"  {r['index']}: {r['util_gpu']}%, {r['util_mem']}%, {r['mem_used']}/{r['mem_total']}MB{busy_mark}"
        )

    rec = _recommend_gpu(rows, prefer_high=prefer_high)
    if rec is not None:
        print(f"[gpu] recommended single GPU index: {rec}")
    else:
        print("[gpu] no recommendation available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
