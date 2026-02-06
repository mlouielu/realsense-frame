# RealSense Frame Capture

A professional utility to capture synchronized RGB-D and high-frequency IMU data from Intel RealSense cameras, featuring a built-in dataloader and visualizer.

## Features
- **Synchronized RGB-D**: Captures color and depth frames.
- **High-Frequency IMU**: Records Accelerometer and Gyroscope data at native frequencies (e.g., 200Hz+) using callback-driven architecture.
- **Depth Compression**: Saves depth data as `zstandard` (`.zst`) compressed files.
- **Alignment Support**: Built-in integration with `realsense-align` for metric-accurate depth-to-RGB projection.
- **Smart Auto-Capture**: Automatically triggers a capture when the camera is detected as stable.
- **API/Dataloader**: Python API for easy session processing and frame access.
- **CLI for Device Configuration**: Easily list available camera streams and formats.
- **Point Cloud Export**: Export captured frames as colored PLY point clouds for 3D reconstruction.

## Requirements
- Python 3.12+ (required for `realsense-align` compatibility)
- Intel RealSense Camera (D400 series recommended)
- `uv` for dependency management.

## Installation
```bash
uv sync
```

## Usage

### 1. Configure Your RealSense Camera
Before capturing, you can list all available camera stream profiles (resolutions, formats, FPS) to help configure your `config.toml`:
```bash
uv run realsense-frame-capture list-streams
```
This command will output detailed information for all connected RealSense devices.

### 2. Capture Data
Run the capture tool to start a new session. The live preview window will now also display a **stability score** to guide your capture:
```bash
uv run realsense-frame-capture capture --output captures --config config.toml
```
- `c`: Capture frame manually.
- `a`: Toggle auto-capture on stability.
- `q`: Quit.

### 3. Visualize Session
Browse captured frames with real-time alignment. You can also export individual frames as colored PLY point clouds:
```bash
# Basic visualization
uv run realsense-frame-visualizer captures/session_YYYYMMDD_HHMMSS

# Visualize and export point clouds to a directory
uv run realsense-frame-visualizer captures/session_YYYYMMDD_HHMMSS --export-ply path/to/output_ply_folder
```
- `n` / `Right Arrow`: Next Frame.
- `p` / `Left Arrow`: Previous Frame.
- `q`: Quit.

### 4. Developer API (Dataloader)
Access session data programmatically:
```python
from realsense_frame.loader import SessionLoader

loader = SessionLoader("captures/session_20260206_120000")
frame = loader.get_frame(0)

print(f"Timestamp: {frame.timestamp}")
# frame.color: numpy BGR image
# frame.depth: numpy uint16 depth map (decompressed)
# frame.imu_samples: list of high-freq IMU readings
```

## Session Directory Structure
```text
captures/session_TIMESTAMP/
├── config.json          # Global camera intrinsics (color and depth)
├── imu.jsonl            # Full session high-frequency IMU log
└── frame_00000_TS/      # Individual frame directory
    ├── color.png        # RGB Image
    ├── depth.zst        # Zstandard compressed depth
    ├── metadata.json    # Frame timestamps, depth scale, and IMU snapshot
    └── imu.jsonl        # High-freq IMU window (latest 100 samples)
```

## Configuration (`config.toml`)
This file specifies the desired stream configurations for your RealSense camera. Use `realsense-frame-capture list-streams` to see supported values.

```toml
[color]
width = 640
height = 480
fps = 30
format = "bgr8" # e.g., bgr8, rgb8

[depth]
width = 640
height = 480
fps = 30
format = "z16" # e.g., z16

[accel]
fps = 0 # 0 for max supported frequency (e.g., 200, 400)

[gyro]
fps = 0 # 0 for max supported frequency (e.g., 200, 400)
```
