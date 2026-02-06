# RealSense Frame

A utility to capture synchronized RGB-D and high-frequency IMU data from Intel RealSense cameras, with built-in depth alignment (using extrinsics), dataloader, visualizer, and point cloud export.

## Features
- **Synchronized RGB-D**: Captures color, depth, and infrared frames.
- **High-Frequency IMU**: Records accelerometer and gyroscope data at native frequencies (200Hz+).
- **Depth Compression**: Saves depth as zstandard (`.zst`) compressed files.
- **Depth-to-Color Alignment**: Uses [`realsense-align`](https://github.com/mlouielu/realsense-align) with full extrinsic support for metric-accurate projection.
- **Smart Auto-Capture**: Automatically triggers capture when the camera is stable.
- **Point Cloud Export**: Export aligned colored PLY point clouds for 3D reconstruction.
- **Dataloader API**: Python API for session processing, frame access, and alignment.
- **Configurable Depth Sensor**: Control laser power, visual preset, exposure via `config.toml`.

## Requirements
- Python 3.12+
- Intel RealSense Camera (D400 series recommended)
- `uv` for dependency management

## Installation
```bash
uv sync
```

### Shell Completion (optional)
For zsh, add to `~/.zshrc`:
```bash
eval "$(_REALSENSE_FRAME_COMPLETE=zsh_source realsense-frame)"
```
For bash, add to `~/.bashrc`:
```bash
eval "$(_REALSENSE_FRAME_COMPLETE=bash_source realsense-frame)"
```

## CLI

All commands are under a single `realsense-frame` entry point:

```
realsense-frame capture        # Start a capture session
realsense-frame visualize      # Browse captured frames
realsense-frame export-ply     # Export aligned point clouds
realsense-frame list-streams   # List available camera streams
```

### Capture
```bash
uv run realsense-frame capture --output captures --config config.toml
```
| Key | Action |
|-----|--------|
| `c` | Capture frame manually |
| `a` | Toggle auto-capture on stability |
| `v` | Switch view (all/color/depth/infra1/infra2) |
| `l` | Toggle laser emitter |
| `q` | Quit |

### Visualize
```bash
uv run realsense-frame visualize captures/session_YYYYMMDD_HHMMSS
```
| Key | Action |
|-----|--------|
| `n` / Right | Next frame |
| `p` / Left  | Previous frame |
| `q`         | Quit |

### Export Point Clouds
```bash
# Export all frames
uv run realsense-frame export-ply captures/session_YYYYMMDD_HHMMSS

# Export to a specific directory
uv run realsense-frame export-ply captures/session_YYYYMMDD_HHMMSS -o output/

# Export a range of frames
uv run realsense-frame export-ply captures/session_YYYYMMDD_HHMMSS -f 0-5
```

### List Streams
```bash
uv run realsense-frame list-streams
```

## Dataloader API

```python
from realsense_frame.loader import SessionLoader

loader = SessionLoader("captures/session_YYYYMMDD_HHMMSS")
print(f"Frames: {len(loader)}, Align target: {loader.align_target}")

frame = loader.get_frame(0)

frame.color    # numpy BGR image
frame.depth    # numpy uint16 depth map
frame.infra1   # numpy infrared image
frame.infra2   # numpy infrared image
frame.d2c()    # aligned depth (using intrinsics + extrinsics)

frame.show()           # display all streams
frame.color.show()     # display single stream
frame.d2c().show()     # display aligned depth
```

Intrinsics and extrinsics are loaded automatically from the session's `config.json`.

## Session Directory Structure
```
captures/session_YYYYMMDD_HHMMSS/
├── config.json                    # Intrinsics, extrinsics, device info, depth sensor options
├── imu.jsonl                      # Full session high-frequency IMU log
├── ply/                           # Point cloud exports (created by export-ply)
│   ├── frame_00000.ply
│   └── ...
└── frame_00000_YYYYMMDD_HHMMSS/   # Individual frame
    ├── color.png                  # RGB image
    ├── depth.zst                  # Zstandard compressed uint16 depth
    ├── infra_1.png                # Infrared left
    ├── infra_2.png                # Infrared right
    ├── metadata.json              # Timestamps, depth scale, IMU snapshot
    └── imu.jsonl                  # High-freq IMU window
```

### config.json
Contains camera intrinsics, extrinsics, and device parameters:
```json
{
  "color_intrinsics": { "width": 640, "height": 480, "fx": ..., "fy": ..., "ppx": ..., "ppy": ... },
  "depth_intrinsics": { ... },
  "depth_to_color_extrinsics": {
    "rotation": [r00, r10, r20, r01, r11, r21, r02, r12, r22],
    "translation": [tx, ty, tz]
  }
}
```

## Configuration (`config.toml`)

```toml
[color]
width = 640
height = 480
fps = 30
format = "bgr8"

[stereo]
width = 640
height = 480
fps = 30
enable_depth = true
enable_infra = true
depth_format = "z16"
infra_format = "y8"

[accel]
fps = 0  # 0 = max supported frequency

[gyro]
fps = 0

[depth_sensor]
laser_power = 150
visual_preset = "High Accuracy"
enable_auto_exposure = true
exposure = 8500  # only used if auto_exposure is false
```

Use `realsense-frame list-streams` to see supported stream profiles and presets.
