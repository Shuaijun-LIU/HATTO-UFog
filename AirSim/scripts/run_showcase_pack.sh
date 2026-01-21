#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run an AirSimNH showcase pack with CPU/GPU checks and "prefer highest-index idle GPU".

Default produces:
- mainline demo video
- Framework→AirSim mainline replay video (+ screenshots)
- auxline demo video
Plus:
- one GIF preview per video (no system ffmpeg required)

Examples:
  ./scripts/run_showcase_pack.sh
  ./scripts/run_showcase_pack.sh --framework_timeseries ../Framework/runs/<...>/timeseries.parquet
  ./scripts/run_showcase_pack.sh --gpu 7
  ./scripts/run_showcase_pack.sh --env landscapemountains --weather_snow 0.75 --weather_road_snow 0.70
  ./scripts/run_showcase_pack.sh --res 1920 1080 --steps 200

Notes:
- Uses AirSimNH environment by default.
- Forces GPU selection via Unreal/Vulkan `-graphicsadapter=<idx>` (check log for "Using Device <idx>").
  - Supported envs: airsimnh | blocks | abandonedpark | landscapemountains (auto-download if missing).
EOF
}

RESX=1280
RESY=720
STEPS=200
DT=0.05
FPS=20.0
SPEED=2.0
GPU_OVERRIDE=""
ENV_NAME="airsimnh"
KEEP_FRAMES=0
FRAMEWORK_TIMESERIES_DEFAULT="../Framework/runs/wind_long_complete/acs_20260115_230039/timeseries.parquet"
FRAMEWORK_TIMESERIES="$FRAMEWORK_TIMESERIES_DEFAULT"
WEATHER_FOG=""
WEATHER_RAIN=""
WEATHER_SNOW=""
WEATHER_ROAD_SNOW=""
TIME_OF_DAY=""
Z_UP_M=""

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
    --speed)
      SPEED="${2:?}"; shift 2 ;;
    --z_up_m)
      Z_UP_M="${2:?}"; shift 2 ;;
    --gpu)
      GPU_OVERRIDE="${2:?}"; shift 2 ;;
    --env)
      ENV_NAME="${2:?}"; shift 2 ;;
    --keep_frames)
      KEEP_FRAMES=1; shift 1 ;;
    --framework_timeseries)
      FRAMEWORK_TIMESERIES="${2:?}"; shift 2 ;;
    --time_of_day)
      TIME_OF_DAY="${2:?}"; shift 2 ;;
    --weather_fog)
      WEATHER_FOG="${2:?}"; shift 2 ;;
    --weather_rain)
      WEATHER_RAIN="${2:?}"; shift 2 ;;
    --weather_snow)
      WEATHER_SNOW="${2:?}"; shift 2 ;;
    --weather_road_snow)
      WEATHER_ROAD_SNOW="${2:?}"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$ENV_NAME" == "airsimnh" ]]; then
  ENV_BIN="$BASE_DIR/envs/AirSimNH/AirSimNH/LinuxNoEditor/AirSimNH.sh"
elif [[ "$ENV_NAME" == "blocks" ]]; then
  ENV_BIN="$BASE_DIR/envs/Blocks/LinuxBlocks1.8.1/LinuxNoEditor/Blocks.sh"
elif [[ "$ENV_NAME" == "abandonedpark" ]]; then
  ENV_BIN="$BASE_DIR/envs/AbandonedPark/LinuxNoEditor/AbandonedPark.sh"
elif [[ "$ENV_NAME" == "landscapemountains" ]]; then
  ENV_BIN="$BASE_DIR/envs/LandscapeMountains/LinuxNoEditor/LandscapeMountains.sh"
else
  echo "ERROR: Unknown --env '$ENV_NAME' (expected: airsimnh|blocks|abandonedpark|landscapemountains)" >&2
  exit 2
fi
if [[ "$ENV_NAME" == "abandonedpark" && ! -x "$ENV_BIN" ]]; then
  echo "AbandonedPark missing; downloading..."
  bash "$BASE_DIR/scripts/download_env_abandonedpark.sh"
fi
if [[ "$ENV_NAME" == "airsimnh" && ! -x "$ENV_BIN" ]]; then
  echo "AirSimNH missing; downloading..."
  bash "$BASE_DIR/scripts/download_env_airsimnh.sh"
fi
if [[ "$ENV_NAME" == "blocks" && ! -x "$ENV_BIN" ]]; then
  echo "Blocks missing; downloading..."
  bash "$BASE_DIR/scripts/download_env_blocks.sh"
fi
if [[ "$ENV_NAME" == "landscapemountains" && ! -x "$ENV_BIN" ]]; then
  echo "LandscapeMountains missing; downloading..."
  bash "$BASE_DIR/scripts/download_env_landscapemountains.sh"
fi
if [[ ! -x "$ENV_BIN" ]]; then
  echo "ERROR: AirSim env not found/executable: $ENV_BIN" >&2
  exit 2
fi

resource_snapshot() {
  echo "=== Resource snapshot ==="
  echo "UTC: $(date -u '+%Y-%m-%d %H:%M:%S')"
  uptime || true
  nproc || true
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
  # Pick highest-index GPU with:
  # - util < 10%
  # - memory used < 500 MiB (ignore Xorg's tiny allocation)
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
    # Fallback: if compute-apps reporting is overly broad, still pick the "least likely busy" idle GPU.
    print(max(idle))
else:
    print(0)
PY
}

AIRSIM_PID=""
AIRSIM_LOG=""
AIRSIM_SETTINGS_PATH=""
ENV_EXE_NAME="$(basename "$ENV_BIN" .sh)"

ensure_rpc_free() {
  # AirSim RPC uses 127.0.0.1:41451; multiple envs cannot share it.
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
      echo "Tip: stop that process or run on a different host/session." >&2
      exit 2
    fi
  done

  for i in $(seq 1 30); do
    if ! lsof -nP -iTCP:41451 -sTCP:LISTEN >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
}

launch_env() {
  local settings_template="$1"
  local gpu="$2"

  ensure_rpc_free

  AIRSIM_SETTINGS_PATH="$BASE_DIR/_runtime_settings/settings.json"
  mkdir -p "$(dirname "$AIRSIM_SETTINGS_PATH")"
  cp "$BASE_DIR/$settings_template" "$AIRSIM_SETTINGS_PATH"

  AIRSIM_LOG="$BASE_DIR/_runtime_settings/${ENV_NAME}_gpu${gpu}_$(date -u +%Y%m%d_%H%M%SZ).log"
  echo "=== Launch ${ENV_NAME} ==="
  echo "GPU: $gpu"
  echo "Settings: $AIRSIM_SETTINGS_PATH"
  echo "Log: $AIRSIM_LOG"

  # Force GPU selection for Unreal/Vulkan via -graphicsadapter.
  nohup "$ENV_BIN" \
    -settings="$AIRSIM_SETTINGS_PATH" -windowed -ResX="$RESX" -ResY="$RESY" -NoSound -RenderOffScreen -graphicsadapter="$gpu" \
    >"$AIRSIM_LOG" 2>&1 &
  AIRSIM_PID="$!"

  echo "$AIRSIM_PID" >"$BASE_DIR/_runtime_settings/${ENV_NAME}.pid"

  echo "Waiting for RPC 127.0.0.1:41451 ..."
  local ok=0
  for i in $(seq 1 120); do
    if AIRSIM_SETTINGS_PATH="$AIRSIM_SETTINGS_PATH" python "$BASE_DIR/scripts/validate_airsim_connection.py" --timeout_s 1 --rpc_only >/dev/null 2>&1; then
      ok=1
      break
    fi
    sleep 1
  done
  if [[ "$ok" -ne 1 ]]; then
    echo "ERROR: AirSim RPC not ready." >&2
    tail -n 80 "$AIRSIM_LOG" || true
    exit 2
  fi

  if command -v rg >/dev/null 2>&1; then
    rg -n "Using Device" "$AIRSIM_LOG" | tail -n 2 || true
  else
    grep -n "Using Device" "$AIRSIM_LOG" | tail -n 2 || true
  fi

  # Give the world a brief warmup before the first simGetImages calls (avoids rare Unreal crashes during level init).
  sleep 2
}

stop_env() {
  echo "=== Stop ${ENV_NAME} ==="
  if [[ -n "${AIRSIM_PID:-}" ]]; then
    kill "$AIRSIM_PID" 2>/dev/null || true
  fi
  sleep 1
  pkill -f "/Binaries/Linux/${ENV_EXE_NAME}" 2>/dev/null || true
  AIRSIM_PID=""
}

extract_run_dir() {
  # expects a log file containing "Done: <run_dir>"
  awk '/^Done:/{print $2; exit}' "$1"
}

prune_frames_keep_screenshots() {
  local run_dir="$1"
  local steps="$2"

  local artifacts="$run_dir/artifacts"
  if [[ ! -d "$artifacts" ]]; then
    return 0
  fi
  mkdir -p "$artifacts/screenshots"

  local mid_idx=$((steps / 2))
  local last_idx=$((steps - 1))
  local mid=$(printf "%06d" "$mid_idx")
  local last=$(printf "%06d" "$last_idx")

  for idx in 000000 "$mid" "$last"; do
    for cam in chase fpv; do
      local src="$artifacts/frames_${cam}/frame_${idx}.png"
      if [[ -f "$src" ]]; then
        cp -f "$src" "$artifacts/screenshots/${cam}_${idx}.png"
      fi
    done
  done

  if [[ "$KEEP_FRAMES" -eq 0 ]]; then
    rm -rf "$artifacts/frames_fpv" "$artifacts/frames_chase"
  fi
}

make_gif() {
  local run_dir="$1"
  local mp4="$run_dir/artifacts/video.mp4"
  local gif="$run_dir/artifacts/video.gif"
  if [[ ! -f "$mp4" ]]; then
    return 0
  fi
  if [[ -f "$gif" ]]; then
    return 0
  fi
  python "$BASE_DIR/scripts/make_gif.py" --input "$mp4" --output "$gif" --width 960 --fps 12 --max_s 10 || true
}

main() {
  cd "$BASE_DIR"

  resource_snapshot
  GPU="$(select_gpu)"
  echo "Selected GPU: $GPU"

  # Environment-specific defaults (can be overridden by CLI args).
  if [[ -z "$Z_UP_M" ]]; then
    if [[ "$ENV_NAME" == "abandonedpark" ]]; then
      Z_UP_M="65.0"
    elif [[ "$ENV_NAME" == "landscapemountains" ]]; then
      Z_UP_M="110.0"
    else
      Z_UP_M="30.0"
    fi
  fi

  # For a snowy mountain feel, default snow if not provided.
  if [[ "$ENV_NAME" == "landscapemountains" ]]; then
    if [[ -z "$WEATHER_SNOW" ]]; then WEATHER_SNOW="0.75"; fi
    if [[ -z "$WEATHER_ROAD_SNOW" ]]; then WEATHER_ROAD_SNOW="0.70"; fi
  fi

  # Mainline (ExternalPhysicsEngine): demo + replay.
  launch_env "configs/airsim_settings/settings_mainline_showcase.json" "$GPU"
  EXTRA_ARGS=()
  if [[ -n "$TIME_OF_DAY" ]]; then EXTRA_ARGS+=(--time_of_day "$TIME_OF_DAY"); fi
  if [[ -n "$WEATHER_FOG" ]]; then EXTRA_ARGS+=(--weather_fog "$WEATHER_FOG"); fi
  if [[ -n "$WEATHER_RAIN" ]]; then EXTRA_ARGS+=(--weather_rain "$WEATHER_RAIN"); fi
  if [[ -n "$WEATHER_SNOW" ]]; then EXTRA_ARGS+=(--weather_snow "$WEATHER_SNOW"); fi
  if [[ -n "$WEATHER_ROAD_SNOW" ]]; then EXTRA_ARGS+=(--weather_road_snow "$WEATHER_ROAD_SNOW"); fi

  RUN_LOG="$BASE_DIR/_runtime_settings/run_mainline_$(date -u +%Y%m%d_%H%M%SZ).log"
  AIRSIM_SETTINGS_PATH="$AIRSIM_SETTINGS_PATH" python scripts/run_mainline.py \
    --settings_template configs/airsim_settings/settings_mainline_showcase.json \
    --steps "$STEPS" --dt "$DT" --fps "$FPS" --speed_m_s "$SPEED" --z_up_m "$Z_UP_M" --ignore_collision \
    --plot_user_point --overlay --scene_profile "$ENV_NAME" "${EXTRA_ARGS[@]}" 2>&1 | tee "$RUN_LOG"
  MAINLINE_DIR="$(extract_run_dir "$RUN_LOG")"
  prune_frames_keep_screenshots "$MAINLINE_DIR" "$STEPS"
  make_gif "$MAINLINE_DIR"

  RUN_LOG="$BASE_DIR/_runtime_settings/replay_mainline_$(date -u +%Y%m%d_%H%M%SZ).log"
  AIRSIM_SETTINGS_PATH="$AIRSIM_SETTINGS_PATH" python scripts/replay_framework_mainline.py \
    --framework_timeseries "$FRAMEWORK_TIMESERIES" \
    --settings_template configs/airsim_settings/settings_mainline_showcase.json \
    --max_steps "$STEPS" --dt "$DT" --fps "$FPS" --scale_xy 0.1 --z_up_m "$Z_UP_M" --screenshot_stride 20 --ignore_collision --timeout_s 120 \
    --plot_user_point --overlay --auto_offset "${EXTRA_ARGS[@]}" 2>&1 | tee "$RUN_LOG"
  REPLAY_DIR="$(extract_run_dir "$RUN_LOG")"
  prune_frames_keep_screenshots "$REPLAY_DIR" "$STEPS"
  make_gif "$REPLAY_DIR"
  stop_env

  # Auxline (AirSim dynamics): demo.
  launch_env "configs/airsim_settings/settings_auxline_showcase.json" "$GPU"
  RUN_LOG="$BASE_DIR/_runtime_settings/run_auxline_$(date -u +%Y%m%d_%H%M%SZ).log"
  AIRSIM_SETTINGS_PATH="$AIRSIM_SETTINGS_PATH" python scripts/run_auxline.py \
    --settings_template configs/airsim_settings/settings_auxline_showcase.json \
    --steps "$STEPS" --dt "$DT" --fps "$FPS" --speed_m_s "$SPEED" \
    --plot_user_point --overlay --scene_profile "$ENV_NAME" "${EXTRA_ARGS[@]}" 2>&1 | tee "$RUN_LOG"
  AUXLINE_DIR="$(extract_run_dir "$RUN_LOG")"
  prune_frames_keep_screenshots "$AUXLINE_DIR" "$STEPS"
  make_gif "$AUXLINE_DIR"
  stop_env

  echo "OK: showcase runs are under: $BASE_DIR/runs_airsim"
}

trap stop_env EXIT
main
