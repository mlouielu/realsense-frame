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
    color: Optional[np.ndarray]
    depth: Optional[np.ndarray]
    metadata: Dict
    imu_samples: List[Dict]

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
        return self.config.get("color_intrinsics"), self.config.get("depth_intrinsics")

    def get_frame(self, idx: int) -> FrameData:
        if idx < 0 or idx >= len(self.frame_dirs):
            raise IndexError("Frame index out of range")

        frame_dir = self.frame_dirs[idx]
        
        # Load Metadata
        meta = self._load_json(os.path.join(frame_dir, "metadata.json"))
        
        # Load Color
        color_path = os.path.join(frame_dir, "color.png")
        color = cv2.imread(color_path) if os.path.exists(color_path) else None
        
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
                # Fallback: metadata might not store W/H, check config
                d_intr = self.config.get("depth_intrinsics")
                if d_intr:
                    depth = depth.reshape((d_intr["height"], d_intr["width"]))
                elif color is not None:
                     # Fallback assumption if same res
                    depth = depth.reshape((color.shape[0], color.shape[1]))

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
            color=color,
            depth=depth,
            metadata=meta,
            imu_samples=imu_samples
        )
