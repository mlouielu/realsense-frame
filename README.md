# RealSense Frame Capture

A professional utility to capture synchronized RGB-D and high-frequency IMU data from Intel RealSense cameras.

## Features
- **Synchronized RGB-D**: Captures color and depth frames.
- **High-Frequency IMU**: Records Accelerometer and Gyroscope data at native frequencies (e.g., 200Hz+) using callback-driven architecture.
- **Depth Compression**: Saves depth data as `zstandard` (`.zst`) compressed files to save space.
- **Smart Auto-Capture**: Automatically triggers a capture when the camera is detected as stable (not moving).
- **Session-based Organization**:
    - `config.json`: Camera intrinsics and stream configuration.
    - `imu.jsonl`: Continuous high-frequency IMU log for the entire session.
    - `frame_NNNNN_TIMESTAMP/`: Individual directories per frame containing raw data, frame-specific metadata, and a window of high-frequency IMU samples.
- **Configurable**: Define resolution, FPS, and formats via `config.toml`.

## Requirements
- Python 3.13+
- Intel RealSense Camera (D400 series recommended)
- `uv` for dependency management.

## Installation
```bash
uv sync
```

## Usage
Run the capture tool:
```bash
uv run realsense-frame-capture --output my_captures --config config.toml
```

### Controls:
- `c`: Capture frame manually.
- `a`: Toggle auto-capture on stability.
- `q`: Quit.

### Development:
Run unit tests for stability detection:
```bash
uv run pytest
```

## Configuration (`config.toml`)
```toml
[color]
width = 640
height = 480
fps = 30
format = "bgr8"

[depth]
width = 640
height = 480
fps = 30
format = "z16"

[accel]
fps = 0 # 0 for default/max

[gyro]
fps = 0
```