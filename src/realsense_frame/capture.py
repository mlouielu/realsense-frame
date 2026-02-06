import click
import sys
import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os
import time
import zstandard as zstd
import toml
import queue
from datetime import datetime
from realsense_frame.stability import StabilityDetector

class RealSenseCapture:
    def __init__(self, output_dir="captures", config_path=None):
        self.settings = self.load_config(config_path)

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(output_dir, f"session_{session_id}")
        if not os.path.exists(self.session_dir):
            os.makedirs(self.session_dir)

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.ctx = rs.context()
        devices = self.ctx.query_devices()
        if not devices:
            raise click.ClickException("No RealSense devices found!")

        device = devices[0]
        print(f"Using device: {device.get_info(rs.camera_info.name)}")

        self.available_streams = []
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                self.available_streams.append(profile.stream_type())

        fmt_map = {"bgr8": rs.format.bgr8, "rgb8": rs.format.rgb8, "z16": rs.format.z16, "yuyv": rs.format.yuyv}

        self.has_color = rs.stream.color in self.available_streams
        if self.has_color:
            c = self.settings.get("color", {})
            self.config.enable_stream(rs.stream.color, c.get("width", 640), c.get("height", 480),
                                     fmt_map.get(c.get("format", "bgr8"), rs.format.bgr8), c.get("fps", 30))

        self.has_depth = rs.stream.depth in self.available_streams
        if self.has_depth:
            d = self.settings.get("depth", {})
            self.config.enable_stream(rs.stream.depth, d.get("width", 640), d.get("height", 480),
                                     fmt_map.get(d.get("format", "z16"), rs.format.z16), d.get("fps", 30))

        self.has_accel = rs.stream.accel in self.available_streams
        self.has_gyro = rs.stream.gyro in self.available_streams
        if self.has_accel:
            self.config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, self.settings.get("accel", {}).get("fps", 0))
        if self.has_gyro:
            self.config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, self.settings.get("gyro", {}).get("fps", 0))

        self.imu_file = None
        if self.has_accel or self.has_gyro:
            self.imu_file = open(os.path.join(self.session_dir, "imu.jsonl"), "w")

        self.detector = StabilityDetector(
            history_size=100,
            threshold=0.05,
            has_accel=self.has_accel,
            has_gyro=self.has_gyro
        )

        self.latest_accel = None
        self.latest_gyro = None
        self.frames_queue = queue.Queue(maxsize=2)
        self.cctx = zstd.ZstdCompressor(level=3)
        self.frame_count = 0

        self.profile = self.pipeline.start(self.config, self.frame_callback)
        self.print_summary()
        self.save_session_config()

    def load_config(self, path):
        default = {"color": {"width": 640, "height": 480, "fps": 30, "format": "bgr8"},
                   "depth": {"width": 640, "height": 480, "fps": 30, "format": "z16"},
                   "accel": {"fps": 0}, "gyro": {"fps": 0}}
        if path and os.path.exists(path):
            try:
                user = toml.load(path)
                for s in default:
                    if s in user: default[s].update(user[s])
            except Exception as e: print(f"Config error: {e}")
        return default

    def frame_callback(self, frame):
        if frame.is_motion_frame():
            data = frame.as_motion_frame().get_motion_data()
            ts = frame.get_timestamp()
            st = frame.get_profile().stream_type()
            entry = {"ts": ts, "type": "accel" if st == rs.stream.accel else "gyro", "x": data.x, "y": data.y, "z": data.z}
            if self.imu_file:
                self.imu_file.write(json.dumps(entry) + "\n")
            if st == rs.stream.accel:
                self.latest_accel = data
                self.detector.add_accel(entry)
            else:
                self.latest_gyro = data
                self.detector.add_gyro(entry)
        elif frame.is_frameset():
            if not self.frames_queue.full():
                self.frames_queue.put(frame.as_frameset())

    def print_summary(self):
        print("\n" + "="*40)
        print("CAPTURE SESSION STARTED")
        print(f"Output: {self.session_dir}")
        print("-" * 40)
        print("Streams:")
        for s in self.profile.get_streams():
            if s.is_video_stream_profile():
                p = s.as_video_stream_profile()
                print(f"  - {s.stream_name()}: {p.width()}x{p.height()} @ {p.fps()}fps ({p.format()})")
            else:
                print(f"  - {s.stream_name()}: {s.fps()}Hz")
        print("="*40 + "\n")

    def save_session_config(self):
        cfg = {}
        if self.has_color: cfg["color_intrinsics"] = self.get_intrinsics(self.profile.get_stream(rs.stream.color))
        if self.has_depth: cfg["depth_intrinsics"] = self.get_intrinsics(self.profile.get_stream(rs.stream.depth))
        with open(os.path.join(self.session_dir, "config.json"), "w") as f: json.dump(cfg, f, indent=4)

    def get_intrinsics(self, p):
        try:
            i = p.as_video_stream_profile().get_intrinsics()
            return {"width": i.width, "height": i.height, "ppx": i.ppx, "ppy": i.ppy, "fx": i.fx, "fy": i.fy, "model": str(i.model), "coeffs": i.coeffs}
        except: return None

    def save_frame(self, fs):
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fdir = os.path.join(self.session_dir, f"frame_{self.frame_count:05d}_{ts_str}")
        os.makedirs(fdir)
        meta = {"index": self.frame_count, "ts_iso": datetime.now().isoformat()}
        if self.has_color:
            cf = fs.get_color_frame()
            if cf:
                cv2.imwrite(os.path.join(fdir, "color.png"), np.asanyarray(cf.get_data()))
                meta.update({"color_ts": cf.get_timestamp(), "color_fn": cf.get_frame_number()})
        if self.has_depth:
            df = fs.get_depth_frame()
            if df:
                with open(os.path.join(fdir, "depth.zst"), "wb") as f: f.write(self.cctx.compress(np.asanyarray(df.get_data()).tobytes()))
                meta.update({"depth_ts": df.get_timestamp(), "depth_fn": df.get_frame_number(), "depth_units": df.get_units()})

        if self.has_accel or self.has_gyro:
            with open(os.path.join(fdir, "imu.jsonl"), "w") as f:
                combined = sorted(self.detector.accel_history + self.detector.gyro_history, key=lambda x: x['ts'])
                for e in combined:
                    f.write(json.dumps(e) + "\n")

        if self.latest_accel: meta["accel"] = {"x": self.latest_accel.x, "y": self.latest_accel.y, "z": self.latest_accel.z}
        if self.latest_gyro: meta["gyro"] = {"x": self.latest_gyro.x, "y": self.latest_gyro.y, "z": self.latest_gyro.z}
        with open(os.path.join(fdir, "metadata.json"), "w") as f: json.dump(meta, f, indent=4)
        print(f"Frame saved: {os.path.basename(fdir)}")
        self.frame_count += 1

    def run(self):
        print("Commands: [c] Capture, [a] Toggle Auto, [q] Quit")
        auto, last_auto = False, 0
        try:
            while True:
                try: fs = self.frames_queue.get(timeout=0.1)
                except queue.Empty: fs = None
                if fs:
                    cf = fs.get_color_frame()
                    if cf: img = np.asanyarray(cf.get_data())
                    else:
                        df = fs.get_depth_frame()
                        img = cv2.applyColorMap(cv2.convertScaleAbs(np.asanyarray(df.get_data()), alpha=0.03), cv2.COLORMAP_JET)
                    # Overlay stability score
                    stability_score = self.detector.get_stability_score()
                    score_text = f"Stability: {stability_score:.2f}"
                    text_color = (0, 255, 0) if stability_score < self.detector.threshold else (0, 0, 255) # Green if stable, Red if unstable
                    cv2.putText(img, score_text, (10, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
                    cv2.imshow("RealSense", img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('c') and fs: self.save_frame(fs)
                elif key == ord('a'):
                    auto = not auto
                    print(f"Auto-capture: {'ON' if auto else 'OFF'}")
                if auto and fs and self.detector.is_stable():
                    if time.time() - last_auto > 2:
                        self.save_frame(fs)
                        last_auto = time.time()
        finally:
            self.pipeline.stop()
            if self.imu_file: self.imu_file.close()
            cv2.destroyAllWindows()

@click.group()
def main():
    """RealSense Capture utilities."""
    pass

@main.command(name="capture")
@click.option("--output", default="captures", help="Output directory for captures.")
@click.option("--config", default="config.toml", help="Path to the configuration file.")
def capture_command(output, config):
    """Capture a RealSense session."""
    try:
        RealSenseCapture(output, config).run()
    except click.ClickException as e:
        click.echo(f"Error: {e.message}")
        sys.exit(1)

@main.command(name="list-streams")
def list_streams_command():
    """List available RealSense camera streams and formats."""
    ctx = rs.context()
    devices = ctx.query_devices()
    if not devices:
        click.echo("No RealSense devices found!")
        sys.exit(1)

    click.echo("Available RealSense Devices and Stream Profiles:")
    click.echo("="*60)

    for i, device in enumerate(devices):
        click.echo(f"Device {i}: {device.get_info(rs.camera_info.name)} ({device.get_info(rs.camera_info.serial_number)})")
        click.echo("-" * 60)
        
        sensors = device.query_sensors()
        for sensor in sensors:
            click.echo(f"  Sensor: {sensor.get_info(rs.camera_info.name)}")
            stream_profiles = sensor.get_stream_profiles()
            
            # Group profiles by stream type for better readability
            grouped_profiles = {}
            for profile in stream_profiles:
                stream_type = profile.stream_type()
                if stream_type not in grouped_profiles:
                    grouped_profiles[stream_type] = []
                grouped_profiles[stream_type].append(profile)

            for stream_type, profiles in grouped_profiles.items():
                click.echo(f"    Stream Type: {stream_type}")
                for profile in profiles:
                    if profile.is_video_stream_profile():
                        v_profile = profile.as_video_stream_profile()
                        click.echo(f"      - {v_profile.width()}x{v_profile.height()} @ {v_profile.fps()}fps, Format: {v_profile.format()}")
                    elif profile.is_motion_stream_profile():
                        m_profile = profile.as_motion_stream_profile()
                        click.echo(f"      - {m_profile.fps()}fps, Format: {m_profile.format()}")
                    else:
                        click.echo(f"      - Type: {stream_type}, Format: {profile.format()}")
        click.echo("="*60)
    sys.exit(0)

if __name__ == "__main__":
    main()