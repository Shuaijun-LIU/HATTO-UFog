# AirSim environments (binary downloads)

This project **does not** commit Unreal binary environments to git. Instead, download them locally into `AirSim/envs/`.

## Quick start (recommended)

From `HATTO-UFog/AirSim`:

```bash
bash scripts/download_env_airsimnh.sh
bash scripts/download_env_blocks.sh
bash scripts/download_env_abandonedpark.sh
bash scripts/download_env_landscapemountains.sh
```

Then run the demo pack:

```bash
bash scripts/run_showcase_pack.sh --env airsimnh
```

## Included scenes in this repo (without binaries)

### AirSimNH (Neighborhood)
- Download: `bash scripts/download_env_airsimnh.sh`
- Launcher: `envs/AirSimNH/AirSimNH/LinuxNoEditor/AirSimNH.sh`

### Blocks
- Download: `bash scripts/download_env_blocks.sh`
- Launcher: `envs/Blocks/LinuxBlocks1.8.1/LinuxNoEditor/Blocks.sh`

### AbandonedPark
- Download: `bash scripts/download_env_abandonedpark.sh`
- Launcher: `envs/AbandonedPark/LinuxNoEditor/AbandonedPark.sh`

### LandscapeMountains
- Download: `bash scripts/download_env_landscapemountains.sh`
- Launcher: `envs/LandscapeMountains/LinuxNoEditor/LandscapeMountains.sh`
