#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run mainline-only (ExternalPhysicsEngine) demo videos for selected environments, then
copy updated artifacts into `AirSim/example/<Env>/` with filenames containing "update".

Default runs (re-generate):
- AbandonedPark
- LandscapeMountains (reduced snow for visibility)

Does NOT run replay/auxline.

Examples:
  bash scripts/run_mainline_updates.sh
  bash scripts/run_mainline_updates.sh --env abandonedpark --steps 220
  bash scripts/run_mainline_updates.sh --env landscapemountains --weather_snow 0.15 --weather_road_snow 0.10
  bash scripts/run_mainline_updates.sh --gpu 7
EOF
}

RESX=1920
RESY=1080
STEPS=200
DT=0.05
FPS=20.0
GPU_OVERRIDE=""
ENV_LIST=("abandonedpark" "landscapemountains")
TIME_OF_DAY="2020-01-01 12:00:00"

WEATHER_SNOW_DEFAULT=""
WEATHER_ROAD_SNOW_DEFAULT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --res)
      RESX="${2:?}"; RESY="${3:?}"; shift 3 ;;
    --steps)
      STEPS="${2:?}"; shift 2 ;;
    --dt)
      DT="${2:?}"; shift 2 ;;
    --fps)
      FPS="${2:?}"; shift 2 ;;
    --gpu)
      GPU_OVERRIDE="${2:?}"; shift 2 ;;
    --env)
      ENV_LIST=("${2:?}"); shift 2 ;;
    --time_of_day)
      TIME_OF_DAY="${2:?}"; shift 2 ;;
    --weather_snow)
      WEATHER_SNOW_DEFAULT="${2:?}"; shift 2 ;;
    --weather_road_snow)
      WEATHER_ROAD_SNOW_DEFAULT="${2:?}"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

resource_snapshot() {
  echo "=== Resource snapshot ==="
  echo "UTC: $(date -u '+%Y-%m-%d %H:%M:%S')"
  ps -eo pid,user,pcpu,pmem,comm --sort=-pcpu | head -n 12 || true
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits || true
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
  fi
}

select_gpu() {
  if [[ -n "$GPU_OVERRIDE" ]]; then
    echo "$GPU_OVERRIDE"
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 0
    return 0
  fi
  local gpu_info busy_uuids
  gpu_info="$(nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader,nounits || true)"
  busy_uuids="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits | awk 'NF{print $1}' | sort -u || true)"
  python - <<'PY' "$gpu_info" "$busy_uuids"
import sys

gpu_info = sys.argv[1].strip().splitlines()
busy = {l.strip() for l in sys.argv[2].strip().splitlines() if l.strip()}

idle = []
idle_not_busy = []
for line in gpu_info:
    idx_s, uuid, util_s, mem_s = [x.strip() for x in line.split(",")]
    idx = int(idx_s)
    util = int(util_s)
    mem = int(mem_s)
    if util >= 10:
        continue
    if mem >= 500:
        continue
    idle.append(idx)
    if uuid not in busy:
        idle_not_busy.append(idx)

if idle_not_busy:
    print(max(idle_not_busy))
elif idle:
    print(max(idle))
else:
    print(0)
PY
}

ensure_rpc_free() {
  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi
  local pids
  pids="$(lsof -nP -iTCP:41451 -sTCP:LISTEN -t 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi
  echo "RPC port 41451 already in use; attempting to stop existing env(s) from this repo..."
  for pid in $pids; do
    local cmd
    cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    if [[ "$cmd" == *"$BASE_DIR/envs/"* ]]; then
      echo "Killing PID $pid ($cmd)"
      kill "$pid" 2>/dev/null || true
    else
      echo "ERROR: 127.0.0.1:41451 is used by a non-repo process; refusing to kill it." >&2
      echo "PID $pid: $cmd" >&2
      exit 2
    fi
  done
  for _i in $(seq 1 30); do
    if ! lsof -nP -iTCP:41451 -sTCP:LISTEN >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
}

resolve_env_bin() {
  local env_name="${1:?}"
  local env_bin=""
  case "$env_name" in
    airsimnh) env_bin="$BASE_DIR/envs/AirSimNH/AirSimNH/LinuxNoEditor/AirSimNH.sh" ;;
    blocks) env_bin="$BASE_DIR/envs/Blocks/LinuxBlocks1.8.1/LinuxNoEditor/Blocks.sh" ;;
    abandonedpark) env_bin="$BASE_DIR/envs/AbandonedPark/LinuxNoEditor/AbandonedPark.sh" ;;
    landscapemountains) env_bin="$BASE_DIR/envs/LandscapeMountains/LinuxNoEditor/LandscapeMountains.sh" ;;
    *)
      echo "ERROR: Unknown env '$env_name'" >&2
      exit 2 ;;
  esac

  if [[ "$env_name" == "abandonedpark" && ! -x "$env_bin" ]]; then
    bash "$BASE_DIR/scripts/download_env_abandonedpark.sh"
  fi
  if [[ "$env_name" == "airsimnh" && ! -x "$env_bin" ]]; then
    bash "$BASE_DIR/scripts/download_env_airsimnh.sh"
  fi
  if [[ "$env_name" == "blocks" && ! -x "$env_bin" ]]; then
    bash "$BASE_DIR/scripts/download_env_blocks.sh"
  fi
  if [[ "$env_name" == "landscapemountains" && ! -x "$env_bin" ]]; then
    bash "$BASE_DIR/scripts/download_env_landscapemountains.sh"
  fi

  if [[ ! -x "$env_bin" ]]; then
    echo "ERROR: AirSim env not found/executable: $env_bin" >&2
    exit 2
  fi
  echo "$env_bin"
}

wait_for_rpc() {
  local settings_path="${1:?}"
  for _i in $(seq 1 120); do
    if AIRSIM_SETTINGS_PATH="$settings_path" python "$BASE_DIR/scripts/validate_airsim_connection.py" --timeout_s 1 --rpc_only >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

latest_mainline_run_dir() {
  ls -1dt "$BASE_DIR/runs_airsim"/airsim_mainline_*__airsim_mainline 2>/dev/null | head -n 1
}

copy_update_artifacts() {
  local env_name="${1:?}"
  local run_dir="${2:?}"
  local folder_name="$env_name"
  case "$env_name" in
    abandonedpark) folder_name="AbandonedPark" ;;
    landscapemountains) folder_name="LandscapeMountains" ;;
  esac
  local dst_dir="$BASE_DIR/example/$folder_name"
  mkdir -p "$dst_dir"

  local run_id
  run_id="$(basename "$run_dir")"
  run_id="${run_id%%__*}"

  local src_video="$run_dir/artifacts/video.mp4"
  local src_gif="$run_dir/artifacts/video.gif"

  if [[ ! -f "$src_video" ]]; then
    echo "ERROR: Missing video at $src_video" >&2
    exit 2
  fi

  local out_video="$dst_dir/mainline_update_${run_id}.mp4"
  local out_gif="$dst_dir/mainline_update_${run_id}.gif"
  cp -f "$src_video" "$out_video"
  if [[ -f "$src_gif" ]]; then
    cp -f "$src_gif" "$out_gif"
  fi

  # Copy a few representative frames (start/mid/end), both FPV and chase.
  local n_frames
  n_frames="$(find "$run_dir/artifacts/frames_chase" -maxdepth 1 -type f -name 'frame_*.png' 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$n_frames" -gt 0 ]]; then
    local mid=$((n_frames / 2))
    local last=$((n_frames - 1))
    for idx in 0 "$mid" "$last"; do
      local f_fpv="$run_dir/artifacts/frames_fpv/frame_$(printf '%06d' "$idx").png"
      local f_chase="$run_dir/artifacts/frames_chase/frame_$(printf '%06d' "$idx").png"
      if [[ -f "$f_fpv" ]]; then
        cp -f "$f_fpv" "$dst_dir/mainline_update_${run_id}_fpv_$(printf '%06d' "$idx").png"
      fi
      if [[ -f "$f_chase" ]]; then
        cp -f "$f_chase" "$dst_dir/mainline_update_${run_id}_chase_$(printf '%06d' "$idx").png"
      fi
    done
  fi
}

run_one_env() {
  local env_name="${1:?}"
  local gpu="${2:?}"

  local env_bin
  env_bin="$(resolve_env_bin "$env_name")"

  local settings_template="$BASE_DIR/configs/airsim_settings/settings_mainline_showcase_1080p.json"
  local settings_path="$BASE_DIR/_runtime_settings/settings.json"
  mkdir -p "$(dirname "$settings_path")"
  cp "$settings_template" "$settings_path"

  ensure_rpc_free
  resource_snapshot

  echo "=== Starting env: $env_name (gpu=$gpu) ==="
  local log_dir="$BASE_DIR/_runtime_logs"
  mkdir -p "$log_dir"
  local env_log="$log_dir/${env_name}_mainline_$(date -u '+%Y%m%d_%H%M%SZ').log"
  set +e
  "$env_bin" \
    -settings="$settings_path" -windowed -ResX="$RESX" -ResY="$RESY" -NoSound -RenderOffScreen -graphicsadapter="$gpu" \
    >"$env_log" 2>&1 &
  local env_pid=$!
  set -e

  if ! wait_for_rpc "$settings_path"; then
    echo "ERROR: AirSim RPC did not become ready for env '$env_name'." >&2
    echo "Log: $env_log" >&2
    kill "$env_pid" 2>/dev/null || true
    exit 2
  fi

  local z_up="40"
  local weather_snow=""
  local weather_road_snow=""
  case "$env_name" in
    abandonedpark) z_up="65" ;;
    landscapemountains)
      z_up="110"
      weather_snow="${WEATHER_SNOW_DEFAULT:-0.15}"
      weather_road_snow="${WEATHER_ROAD_SNOW_DEFAULT:-0.10}"
      ;;
  esac

  local settings_basename
  settings_basename="$(basename "$settings_template")"

  local args=(
    "$BASE_DIR/scripts/run_mainline.py"
    --settings_template "configs/airsim_settings/$settings_basename"
    --steps "$STEPS"
    --dt "$DT"
    --fps "$FPS"
    --speed_m_s 2.0
    --z_up_m "$z_up"
    --ignore_collision
    --plot_user_point
    --overlay
    --scene_profile "$env_name"
    --time_of_day "$TIME_OF_DAY"
  )
  if [[ -n "$weather_snow" ]]; then
    args+=(--weather_snow "$weather_snow")
  fi
  if [[ -n "$weather_road_snow" ]]; then
    args+=(--weather_road_snow "$weather_road_snow")
  fi

  echo "=== Running mainline: $env_name ==="
  AIRSIM_SETTINGS_PATH="$settings_path" python "${args[@]}"

  local run_dir
  run_dir="$(latest_mainline_run_dir)"
  if [[ -z "$run_dir" ]]; then
    echo "ERROR: Failed to locate latest mainline run dir." >&2
    kill "$env_pid" 2>/dev/null || true
    exit 2
  fi

  echo "=== Making GIF: $env_name ==="
  python "$BASE_DIR/scripts/make_gif.py" --input "$run_dir/artifacts/video.mp4" --output "$run_dir/artifacts/video.gif" --width 960 --fps 12 --max_s 10.0

  echo "=== Copying update artifacts into example/: $env_name ==="
  copy_update_artifacts "$env_name" "$run_dir"

  echo "=== Stopping env: $env_name (pid=$env_pid) ==="
  kill "$env_pid" 2>/dev/null || true
  sleep 1
}

main() {
  local gpu
  gpu="$(select_gpu)"
  echo "Selected GPU: $gpu"
  for env_name in "${ENV_LIST[@]}"; do
    run_one_env "$env_name" "$gpu"
  done
  echo "Done."
}

main "$@"
