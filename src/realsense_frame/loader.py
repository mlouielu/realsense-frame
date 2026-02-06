import os
import json
import cv2
import numpy as np
import zstandard as zstd
import glob
from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class FrameData:
    index: int
    timestamp: str
    color: Optional["VisualArray"]
    depth: Optional["VisualArray"]
    infra1: Optional["VisualArray"] = None
    infra2: Optional["VisualArray"] = None
    metadata: Dict = None
    imu_samples: List[Dict] = None

    def show(self, wait=True):
        """Display all available streams in this frame."""
        imgs = []
        if self.color is not None:
            imgs.append(self.color)
        if self.infra1 is not None:
            imgs.append(cv2.cvtColor(self.infra1, cv2.COLOR_GRAY2BGR))
        if self.infra2 is not None:
            imgs.append(cv2.cvtColor(self.infra2, cv2.COLOR_GRAY2BGR))
        if self.depth is not None:
            imgs.append(
                cv2.applyColorMap(
                    cv2.convertScaleAbs(self.depth, alpha=0.03), cv2.COLORMAP_JET
                )
            )

        if not imgs:
            print("No streams available to show.")
            return

        # Simple tiling
        h, w = imgs[0].shape[:2]
        resized = [cv2.resize(img, (w, h)) for img in imgs]
        count = len(resized)

        if count == 1:
            display = resized[0]
        elif count == 2:
            display = np.hstack(resized)
        else:
            top = np.hstack(resized[:2])
            bottom = np.hstack(
                resized[2:] + [np.zeros_like(resized[0])] * (2 - len(resized[2:]))
            )
            display = np.vstack([top, bottom])

        title = f"Frame {self.index} - {self.timestamp}"
        cv2.imshow(title, display)
        if wait:
            cv2.waitKey(0)
            cv2.destroyWindow(title)


class VisualArray(np.ndarray):
    """Numpy array subclass that adds a .show() method for convenience."""

    def __new__(cls, input_array, title="Stream"):
        if input_array is None:
            return None
        obj = np.asanyarray(input_array).view(cls)
        obj._title = title
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._title = getattr(obj, "_title", "Stream")

    def show(self, title=None, wait=True):
        t = title or self._title
        display_img = np.asanyarray(self)
        if self.dtype == np.uint16:
            display_img = cv2.applyColorMap(
                cv2.convertScaleAbs(display_img, alpha=0.03), cv2.COLORMAP_JET
            )
        elif len(display_img.shape) == 2:
            display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

        cv2.imshow(t, display_img)
        if wait:
            cv2.waitKey(0)
            cv2.destroyWindow(t)


class SessionLoader:
    def __init__(self, session_path: str):
        self.session_path = session_path
        if not os.path.exists(session_path):
            raise ValueError(f"Session path does not exist: {session_path}")

        self.config_path = os.path.join(session_path, "config.json")
        self.config = self._load_json(self.config_path)

        # Sort frames by the frame index in the folder name
        self.frame_dirs = sorted(glob.glob(os.path.join(session_path, "frame_*")))

        self.dctx = zstd.ZstdDecompressor()

    def _load_json(self, path):
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}

    def __len__(self):
        return len(self.frame_dirs)

    def get_intrinsics(self):
        return {
            "color": self.config.get("color_intrinsics"),
            "depth": self.config.get("depth_intrinsics"),
            "infra1": self.config.get("infra1_intrinsics"),
            "infra2": self.config.get("infra2_intrinsics"),
        }

    def get_frame(self, idx: int) -> FrameData:
        if idx < 0 or idx >= len(self.frame_dirs):
            raise IndexError("Frame index out of range")

        frame_dir = self.frame_dirs[idx]

        # Load Metadata
        meta = self._load_json(os.path.join(frame_dir, "metadata.json"))

        # Load Color
        color_path = os.path.join(frame_dir, "color.png")
        color = cv2.imread(color_path) if os.path.exists(color_path) else None

        # Load Infrared
        infra1_path = os.path.join(frame_dir, "infra_1.png")
        infra1 = (
            cv2.imread(infra1_path, cv2.IMREAD_UNCHANGED)
            if os.path.exists(infra1_path)
            else None
        )

        infra2_path = os.path.join(frame_dir, "infra_2.png")
        infra2 = (
            cv2.imread(infra2_path, cv2.IMREAD_UNCHANGED)
            if os.path.exists(infra2_path)
            else None
        )

        # Load Depth (ZST Compressed)
        depth = None
        depth_path = os.path.join(frame_dir, "depth.zst")
        if os.path.exists(depth_path):
            with open(depth_path, "rb") as f:
                compressed_data = f.read()
                decompressed = self.dctx.decompress(compressed_data)
                # Assume Z16 (uint16)
                depth = np.frombuffer(decompressed, dtype=np.uint16)
                # We need to reshape. Use intrinsics or metadata to find shape
                d_intr = self.config.get("depth_intrinsics")
                if d_intr:
                    depth = depth.reshape((d_intr["height"], d_intr["width"]))
                elif color is not None:
                    depth = depth.reshape((color.shape[0], color.shape[1]))
                elif infra1 is not None:
                    depth = depth.reshape((infra1.shape[0], infra1.shape[1]))

        # Load Frame-Specific IMU
        imu_samples = []
        imu_path = os.path.join(frame_dir, "imu.jsonl")
        if os.path.exists(imu_path):
            with open(imu_path, "r") as f:
                for line in f:
                    imu_samples.append(json.loads(line))

        return FrameData(
            index=idx,
            timestamp=meta.get("ts_iso", ""),
            color=VisualArray(color, "Color"),
            depth=VisualArray(depth, "Depth"),
            infra1=VisualArray(infra1, "Infrared 1"),
            infra2=VisualArray(infra2, "Infrared 2"),
            metadata=meta,
            imu_samples=imu_samples,
        )
