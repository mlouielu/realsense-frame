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
import shutil
from datetime import datetime
from loguru import logger
from realsense_frame.stability import StabilityDetector
from realsense_frame.utils import create_colorbar # Import create_colorbar

class RealSenseCapture:
    def __init__(self, output_dir="captures", config_path=None):
        self.settings = self.load_config(config_path)

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(output_dir, f"session_{session_id}")

        if not os.path.exists(self.session_dir):
            os.makedirs(self.session_dir)

        # Setup loguru to log to both console and session file
        logger.remove()  # Remove default handler
        logger.add(
            sys.stderr,
            format="<green>[{time:HH:mm:ss}]</green> <level>{message}</level>",
            level="INFO",
        )
        logger.add(
            os.path.join(self.session_dir, "session.log"),
            level="DEBUG",
            rotation="10 MB",
        )

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.ctx = rs.context()
        devices = self.ctx.query_devices()
        if not devices:
            logger.error("No RealSense devices found!")
            raise click.ClickException("No RealSense devices found!")

        device = devices[0]
        logger.info(f"Using device: {device.get_info(rs.camera_info.name)}")

        self.available_streams = []
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                self.available_streams.append(
                    (profile.stream_type(), profile.stream_index())
                )

        fmt_map = {
            "bgr8": rs.format.bgr8,
            "rgb8": rs.format.rgb8,
            "z16": rs.format.z16,
            "yuyv": rs.format.yuyv,
            "y8": rs.format.y8,
        }

        self.has_color = any(s[0] == rs.stream.color for s in self.available_streams)
        if self.has_color:
            c = self.settings.get("color", {})
            self.config.enable_stream(
                rs.stream.color,
                c.get("width", 640),
                c.get("height", 480),
                fmt_map.get(c.get("format", "bgr8"), rs.format.bgr8),
                c.get("fps", 30),
            )

        s = self.settings.get("stereo", {})
        sw, sh, sfps = s.get("width", 640), s.get("height", 480), s.get("fps", 30)

        self.has_depth = s.get("enable_depth", True) and any(
            st[0] == rs.stream.depth for st in self.available_streams
        )
        if self.has_depth:
            self.config.enable_stream(
                rs.stream.depth,
                sw,
                sh,
                fmt_map.get(s.get("depth_format", "z16"), rs.format.z16),
                sfps,
            )

        want_infra = s.get("enable_infra", True)
        self.has_infra1 = want_infra and any(
            st[0] == rs.stream.infrared and st[1] == 1 for st in self.available_streams
        )
        self.has_infra2 = want_infra and any(
            st[0] == rs.stream.infrared and st[1] == 2 for st in self.available_streams
        )

        if self.has_infra1:
            self.config.enable_stream(
                rs.stream.infrared,
                1,
                sw,
                sh,
                fmt_map.get(s.get("infra_format", "y8"), rs.format.y8),
                sfps,
            )
        if self.has_infra2:
            self.config.enable_stream(
                rs.stream.infrared,
                2,
                sw,
                sh,
                fmt_map.get(s.get("infra_format", "y8"), rs.format.y8),
                sfps,
            )

        self.has_accel = any(s[0] == rs.stream.accel for s in self.available_streams)
        self.has_gyro = any(s[0] == rs.stream.gyro for s in self.available_streams)
        if self.has_accel:
            self.config.enable_stream(
                rs.stream.accel,
                rs.format.motion_xyz32f,
                self.settings.get("accel", {}).get("fps", 0),
            )
        if self.has_gyro:
            self.config.enable_stream(
                rs.stream.gyro,
                rs.format.motion_xyz32f,
                self.settings.get("gyro", {}).get("fps", 0),
            )

        self.imu_file = None
        if self.has_accel or self.has_gyro:
            self.imu_file = open(os.path.join(self.session_dir, "imu.jsonl"), "w")

        self.detector = StabilityDetector(
            history_size=100,
            threshold=0.5,
            has_accel=self.has_accel,
            has_gyro=self.has_gyro,
        )

        self.latest_accel = None
        self.latest_gyro = None
        self.frames_queue = queue.Queue(maxsize=2)
        self.cctx = zstd.ZstdCompressor(level=3)
        self.frame_count = 0
        self.dropped_frames = 0
        self.start_time = time.time()
        self.last_fps_time = time.time()
        self.fps_counters = {}
        self.stream_fps = {}

        try:
            self.profile = self.pipeline.start(self.config, self.frame_callback)
        except RuntimeError as e:
            if "Couldn't resolve requests" in str(e):
                logger.error("!!! RealSense Error: Couldn't resolve requests !!!")
                logger.error(
                    "This often happens if you requested a resolution/FPS combination not supported by the Stereo Module."
                )
                logger.error("Check 'list-streams' for supported combinations.")
            raise click.ClickException(str(e))

        # Get Depth Sensor for Emitter Control
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.emitter_enabled = True
        if self.depth_sensor.supports(rs.option.emitter_enabled):
            self.emitter_enabled = bool(
                self.depth_sensor.get_option(rs.option.emitter_enabled)
            )

        self.print_summary()
        self.save_session_config()

    def load_config(self, path):
        default = {
            "color": {"width": 640, "height": 480, "fps": 30, "format": "bgr8"},
            "stereo": {
                "width": 640,
                "height": 480,
                "fps": 30,
                "enable_depth": True,
                "enable_infra": True,
                "depth_format": "z16",
                "infra_format": "y8",
            },
            "accel": {"fps": 0},
            "gyro": {"fps": 0},
        }
        if path and os.path.exists(path):
            try:
                user = toml.load(path)
                for section in default:
                    if section in user:
                        default[section].update(user[section])
            except Exception as e:
                logger.error(f"Config error: {e}")
        return default

    def frame_callback(self, frame):
        now = time.time()

        # Track FPS for this specific frame's stream(s)
        if frame.is_frameset():
            for f in frame.as_frameset():
                s_name = f.get_profile().stream_name()
                self.fps_counters[s_name] = self.fps_counters.get(s_name, 0) + 1
        else:
            s_name = frame.get_profile().stream_name()
            self.fps_counters[s_name] = self.fps_counters.get(s_name, 0) + 1

        if now - self.last_fps_time >= 1.0:
            dt = now - self.last_fps_time
            for s_name, count in self.fps_counters.items():
                self.stream_fps[s_name] = count / dt
            self.fps_counters = {k: 0 for k in self.fps_counters}  # Reset counts
            self.last_fps_time = now

        if frame.is_motion_frame():
            data = frame.as_motion_frame().get_motion_data()
            ts = frame.get_timestamp()
            st = frame.get_profile().stream_type()
            entry = {
                "ts": ts,
                "type": "accel" if st == rs.stream.accel else "gyro",
                "x": data.x,
                "y": data.y,
                "z": data.z,
            }
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
            else:
                self.dropped_frames += 1

    def print_summary(self):
        logger.info("\n" + "=" * 40)
        logger.info("CAPTURE SESSION STARTED")
        logger.info(f"Output: {self.session_dir}")
        logger.info("-" * 40)
        logger.info("Streams:")
        for s in self.profile.get_streams():
            s_name = s.stream_name()
            if s.is_video_stream_profile():
                p = s.as_video_stream_profile()
                fmt = str(p.format()).split(".")[-1].upper()
                logger.info(
                    f"  - {s_name}: {p.width()}x{p.height()} @ {p.fps()}fps ({fmt})"
                )
            else:
                logger.info(f"  - {s_name}: {s.fps()}Hz")
        logger.info("=" * 40 + "\n")

    def save_session_config(self):
        device = self.profile.get_device()
        cfg = {
            "device": {
                "name": device.get_info(rs.camera_info.name),
                "serial_number": device.get_info(rs.camera_info.serial_number),
                "firmware_version": device.get_info(rs.camera_info.firmware_version),
                "usb_type": (
                    device.get_info(rs.camera_info.usb_type_descriptor)
                    if device.supports(rs.camera_info.usb_type_descriptor)
                    else "unknown"
                ),
            }
        }
        if self.has_color:
            cfg["color_intrinsics"] = self.get_intrinsics(
                self.profile.get_stream(rs.stream.color)
            )
        if self.has_depth:
            cfg["depth_intrinsics"] = self.get_intrinsics(
                self.profile.get_stream(rs.stream.depth)
            )
        if self.has_infra1:
            cfg["infra1_intrinsics"] = self.get_intrinsics(
                self.profile.get_stream(rs.stream.infrared, 1)
            )
        if self.has_infra2:
            cfg["infra2_intrinsics"] = self.get_intrinsics(
                self.profile.get_stream(rs.stream.infrared, 2)
            )
        with open(os.path.join(self.session_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=4)

    def get_intrinsics(self, p):
        try:
            i = p.as_video_stream_profile().get_intrinsics()
            return {
                "width": i.width,
                "height": i.height,
                "ppx": i.ppx,
                "ppy": i.ppy,
                "fx": i.fx,
                "fy": i.fy,
                "model": str(i.model),
                "coeffs": i.coeffs,
            }
        except:
            return None

    def save_frame(self, fs, trigger="manual"):
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fdir = os.path.join(self.session_dir, f"frame_{self.frame_count:05d}_{ts_str}")
        os.makedirs(fdir)
        meta = {
            "index": self.frame_count,
            "ts_iso": datetime.now().isoformat(),
            "trigger": trigger,
        }
        if self.has_color:
            cf = fs.get_color_frame()
            if cf:
                cv2.imwrite(
                    os.path.join(fdir, "color.png"), np.asanyarray(cf.get_data())
                )
                meta.update(
                    {
                        "color_ts": cf.get_timestamp(),
                        "color_ts_domain": str(cf.get_frame_timestamp_domain()),
                        "color_fn": cf.get_frame_number(),
                    }
                )
        if self.has_depth:
            df = fs.get_depth_frame()
            if df:
                with open(os.path.join(fdir, "depth.zst"), "wb") as f:
                    f.write(self.cctx.compress(np.asanyarray(df.get_data()).tobytes()))
                meta.update(
                    {
                        "depth_ts": df.get_timestamp(),
                        "depth_ts_domain": str(df.get_frame_timestamp_domain()),
                        "depth_fn": df.get_frame_number(),
                        "depth_units": df.get_units(),
                    }
                )

        if self.has_infra1:
            if1 = fs.get_infrared_frame(1)
            if if1:
                cv2.imwrite(
                    os.path.join(fdir, "infra_1.png"), np.asanyarray(if1.get_data())
                )
                meta.update(
                    {
                        "infra1_ts": if1.get_timestamp(),
                        "infra1_ts_domain": str(if1.get_frame_timestamp_domain()),
                        "infra1_fn": if1.get_frame_number(),
                    }
                )
        if self.has_infra2:
            if2 = fs.get_infrared_frame(2)
            if if2:
                cv2.imwrite(
                    os.path.join(fdir, "infra_2.png"), np.asanyarray(if2.get_data())
                )
                meta.update(
                    {
                        "infra2_ts": if2.get_timestamp(),
                        "infra2_ts_domain": str(if2.get_frame_timestamp_domain()),
                        "infra2_fn": if2.get_frame_number(),
                    }
                )

        if self.has_accel or self.has_gyro:
            with open(os.path.join(fdir, "imu.jsonl"), "w") as f:
                combined = sorted(
                    self.detector.accel_history + self.detector.gyro_history,
                    key=lambda x: x["ts"],
                )
                for e in combined:
                    f.write(json.dumps(e) + "\n")

        if self.latest_accel:
            meta["accel"] = {
                "x": self.latest_accel.x,
                "y": self.latest_accel.y,
                "z": self.latest_accel.z,
            }
        if self.latest_gyro:
            meta["gyro"] = {
                "x": self.latest_gyro.x,
                "y": self.latest_gyro.y,
                "z": self.latest_gyro.z,
            }
        with open(os.path.join(fdir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=4)
        logger.debug(f"Frame saved: {os.path.basename(fdir)}")
        self.frame_count += 1
        self.last_capture_ts = time.time()

    def _get_display_image(self, fs, mode="all"):
        images = []

        # Helper to process image
        def process(img, is_depth=False):
            if img is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            data = np.asanyarray(img.get_data())
            if is_depth:
                return cv2.applyColorMap(
                    cv2.convertScaleAbs(data, alpha=0.03), cv2.COLORMAP_JET
                )
            if len(data.shape) == 2:
                return cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
            return data

        # Collect available frames
        c = process(fs.get_color_frame()) if self.has_color else None
        d = process(fs.get_depth_frame(), is_depth=True) if self.has_depth else None
        if d is not None:
            colorbar = create_colorbar(d.shape[0])
            d = np.hstack((d, colorbar))
        i1 = process(fs.get_infrared_frame(1)) if self.has_infra1 else None
        i2 = process(fs.get_infrared_frame(2)) if self.has_infra2 else None

        if mode == "color":
            return c if c is not None else np.zeros((480, 640, 3), np.uint8)
        if mode == "depth":
            return d if d is not None else np.zeros((480, 640, 3), np.uint8)
        if mode == "infra1":
            return i1 if i1 is not None else np.zeros((480, 640, 3), np.uint8)
        if mode == "infra2":
            return i2 if i2 is not None else np.zeros((480, 640, 3), np.uint8)

        # Mode "all" - Tile images
        valid_imgs = [x for x in [c, d, i1, i2] if x is not None]
        if not valid_imgs:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # Simple grid layout
        count = len(valid_imgs)
        if count == 1:
            return valid_imgs[0]

        # Resize to match first image height for stacking
        h, w = valid_imgs[0].shape[:2]
        resized = []
        for img in valid_imgs:
            if img.shape[:2] != (h, w):
                resized.append(cv2.resize(img, (w, h)))
            else:
                resized.append(img)

        if count == 2:
            return np.hstack(resized)
        if count <= 4:
            top = np.hstack(resized[:2])
            bottom = np.hstack(
                resized[2:] + [np.zeros_like(resized[0])] * (2 - len(resized[2:]))
            )
            return np.vstack([top, bottom])

        return valid_imgs[0]  # Fallback

    def run(self):
        print(
            "Commands: [c] Capture, [a] Toggle Auto, [v] Switch View, [l] Toggle Laser, [q] Quit"
        )
        auto, last_auto = False, 0
        auto_period = 2.0  # Minimum period for auto-capture
        self.last_capture_ts = 0
        latest_fs = None

        display_modes = ["all"]
        if self.has_color:
            display_modes.append("color")
        if self.has_depth:
            display_modes.append("depth")
        if self.has_infra1:
            display_modes.append("infra1")
        if self.has_infra2:
            display_modes.append("infra2")
        current_mode_idx = 0

        # Disk usage check throttle
        last_disk_check = 0
        free_space_gb = 0.0

        try:
            while True:
                try:
                    fs = self.frames_queue.get(timeout=0.1)
                except queue.Empty:
                    fs = None

                # Check warm-up
                is_warmup = (time.time() - self.start_time) < 2.0

                if fs:
                    latest_fs = fs
                    img = self._get_display_image(fs, display_modes[current_mode_idx])

                    # Update Disk Usage (every 2s)
                    if time.time() - last_disk_check > 2.0:
                        try:
                            total, used, free = shutil.disk_usage(self.session_dir)
                            free_space_gb = free / (1024**3)
                        except:
                            pass
                        last_disk_check = time.time()

                    # Stability Check
                    is_stable = False
                    stability_score = 0.0
                    if self.has_accel or self.has_gyro:
                        stability_score = self.detector.get_stability_score()
                        is_stable = self.detector.is_stable()
                    else:
                        is_stable = True  # Assume stable if no IMU

                    # Visual Stability Border
                    border_color = (0, 255, 0) if is_stable else (0, 0, 255)
                    img = cv2.copyMakeBorder(
                        img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=border_color
                    )

                    # Overlay Info
                    infos = []

                    if is_warmup:
                        infos.append(("WARMUP...", (0, 165, 255)))
                    else:
                        # Stability / IMU Status
                        if self.has_accel or self.has_gyro:
                            infos.append(
                                (f"Stability: {stability_score:.2f}", border_color)
                            )
                        else:
                            infos.append(("No IMU", (200, 200, 200)))

                        infos.append(
                            (
                                f"Auto: {'ON' if auto else 'OFF'}",
                                (0, 255, 255) if auto else (200, 200, 200),
                            )
                        )
                        infos.append(
                            (
                                f"View: {display_modes[current_mode_idx].upper()}",
                                (255, 255, 0),
                            )
                        )
                        infos.append(
                            (
                                f"Emitter: {'ON' if self.emitter_enabled else 'OFF'}",
                                (0, 255, 0) if self.emitter_enabled else (0, 0, 255),
                            )
                        )

                        # Stats
                        fps_list = []
                        # Priority order for video streams
                        for priority in ["Color", "Depth", "Infrared 1", "Infrared 2"]:
                            if priority in self.stream_fps:
                                fps_list.append(
                                    f"{priority[0]}:{self.stream_fps[priority]:.0f}"
                                )

                        if fps_list:
                            infos.append(
                                ("FPS: " + " ".join(fps_list), (255, 255, 255))
                            )

                        infos.append(
                            (
                                f"Drops: {self.dropped_frames}",
                                (
                                    (0, 0, 255)
                                    if self.dropped_frames > 0
                                    else (255, 255, 255)
                                ),
                            )
                        )
                        infos.append(
                            (
                                f"Disk: {free_space_gb:.1f} GB",
                                (255, 255, 255) if free_space_gb > 10 else (0, 0, 255),
                            )
                        )

                    y0 = img.shape[0] - 30
                    for i, (text, color) in enumerate(infos):
                        cv2.putText(
                            img,
                            text,
                            (20, y0 - (i * 25)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2,
                            cv2.LINE_AA,
                        )

                    # Status Indicator (Hollow circle = working, Filled = captured)
                    indicator_pos = (img.shape[1] - 50, 50)
                    is_capturing = time.time() - self.last_capture_ts < 0.3

                    if is_capturing:
                        cv2.circle(img, indicator_pos, 20, (0, 255, 0), -1)
                        cv2.putText(
                            img,
                            "CAPTURED",
                            (img.shape[1] - 170, 55),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )
                    else:
                        cv2.circle(
                            img, indicator_pos, 20, (0, 255, 0), 2
                        )  # Hollow circle

                    # Add colorbar if displaying depth
                    if display_modes[current_mode_idx] == "depth":
                        colorbar = create_colorbar(img.shape[0])
                        img = np.hstack((img, colorbar))

                    cv2.imshow("RealSense", img)

                    # Auto Capture Logic
                    if not is_warmup:
                        should_auto = False
                        if auto and (time.time() - last_auto > auto_period):
                            if is_stable:
                                should_auto = True

                        if should_auto:
                            self.save_frame(fs, trigger="auto")
                            logger.info(
                                f"[*] Auto-capture saved: frame_{self.frame_count-1:05d}"
                            )
                            last_auto = time.time()

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif not is_warmup:
                    if key == ord("c") and latest_fs:
                        self.save_frame(latest_fs, trigger="manual")
                        logger.info(
                            f"[*] Manual capture saved: frame_{self.frame_count-1:05d}"
                        )
                    elif key == ord("a"):
                        auto = not auto
                        state = "ENABLED" if auto else "DISABLED"
                        logger.info(f"[!] Auto-capture: {state}")
                    elif key == ord("v"):
                        current_mode_idx = (current_mode_idx + 1) % len(display_modes)
                        logger.info(
                            f"[!] View mode: {display_modes[current_mode_idx].upper()}"
                        )
                    elif key == ord("l"):
                        if self.depth_sensor.supports(rs.option.emitter_enabled):
                            self.emitter_enabled = not self.emitter_enabled
                            self.depth_sensor.set_option(
                                rs.option.emitter_enabled,
                                1.0 if self.emitter_enabled else 0.0,
                            )
                            state = "ON" if self.emitter_enabled else "OFF"
                            logger.info(f"[!] Laser Emitter: {state}")

        finally:
            self.pipeline.stop()
            if self.imu_file:
                self.imu_file.close()
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
        click.echo("Error: No RealSense devices found!")
        sys.exit(1)

    click.echo("\nAvailable RealSense Devices and Stream Profiles:")
    click.echo("=" * 60)

    for i, device in enumerate(devices):
        click.echo(
            f"Device {i}: {device.get_info(rs.camera_info.name)} ({device.get_info(rs.camera_info.serial_number)})"
        )
        click.echo("-" * 60)

        sensors = device.query_sensors()
        for sensor in sensors:
            click.echo(f"  Sensor: {sensor.get_info(rs.camera_info.name)}")
            stream_profiles = sensor.get_stream_profiles()

            # Group profiles by stream type for better readability
            grouped_profiles = {}
            for profile in stream_profiles:
                stream_type = str(profile.stream_type()).split(".")[-1]
                if stream_type not in grouped_profiles:
                    grouped_profiles[stream_type] = []
                grouped_profiles[stream_type].append(profile)

            for stream_type, profiles in grouped_profiles.items():
                click.echo(f"    Stream Type: {stream_type}")
                for profile in profiles:
                    fmt = str(profile.format()).split(".")[-1]
                    if profile.is_video_stream_profile():
                        v_profile = profile.as_video_stream_profile()
                        click.echo(
                            f"      - {v_profile.width()}x{v_profile.height()} @ {v_profile.fps()}fps, Format: {fmt}"
                        )
                    elif profile.is_motion_stream_profile():
                        m_profile = profile.as_motion_stream_profile()
                        click.echo(f"      - {m_profile.fps()}fps, Format: {fmt}")
                    else:
                        click.echo(f"      - Type: {stream_type}, Format: {fmt}")
        click.echo("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
