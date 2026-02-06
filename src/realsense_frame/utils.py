import cv2
import numpy as np
import os
import realsense_align as ra


def depth_to_pointcloud(depth_image, color_image, depth_intrinsics, depth_scale=0.001):
    """Converts a depth image and color image to a colored 3D point cloud."""
    # Ensure depth_image is float32 for calculations
    depth_image = depth_image.astype(np.float32) * depth_scale

    # Get intrinsics
    fx = depth_intrinsics["fx"]
    fy = depth_intrinsics["fy"]
    ppx = depth_intrinsics["ppx"]
    ppy = depth_intrinsics["ppy"]

    h, w = depth_image.shape

    # Create a grid of pixel coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Calculate 3D points
    z = depth_image
    x = (u - ppx) * z / fx
    y = (v - ppy) * z / fy

    points = np.stack((x, y, z), axis=-1)

    # Reshape for PLY
    points = points.reshape(-1, 3)
    colors = color_image.reshape(-1, 3)  # Assuming BGR, will convert to RGB for PLY

    # Filter out invalid points (where depth is 0)
    valid_mask = z.reshape(-1) > 0
    points = points[valid_mask]
    colors = colors[valid_mask]

    return points, colors


def write_pointcloud_to_ply(filename, points, colors=None):
    """Writes a point cloud to a PLY file."""
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(len(points)):
            if colors is not None:
                # OpenCV uses BGR, PLY usually expects RGB
                f.write(
                    f"{points[i, 0]} {points[i, 1]} {points[i, 2]} {colors[i, 2]} {colors[i, 1]} {colors[i, 0]}\n"
                )
            else:
                f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]}\n")


def create_colorbar(
    height, width=20, cmap=cv2.COLORMAP_JET, max_depth_meters=8.5, alpha=0.03
):
    """Creates a vertical colorbar for depth visualization."""
    colorbar = np.zeros((height, width, 3), dtype=np.uint8)

    # Generate gradient for colorbar
    for i in range(height):
        # Scale 0-height to 0-255 for colormap
        val = int(255 * (height - 1 - i) / (height - 1))
        color = cv2.applyColorMap(np.array([[[val]]], dtype=np.uint8), cmap)[0, 0, :]
        colorbar[i, :] = color

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (255, 255, 255)  # White text

    # Max depth label
    max_label = f"{max_depth_meters:.1f}m"
    cv2.putText(
        colorbar,
        max_label,
        (5, 15),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )

    # Min depth label (0m)
    min_label = "0m"
    text_size = cv2.getTextSize(min_label, font, font_scale, font_thickness)[0]
    cv2.putText(
        colorbar,
        min_label,
        (5, height - text_size[1] - 5),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )

    return colorbar


class RealSenseAligner:
    def __init__(self, target_intr, depth_intr, extrinsics=None):
        # ra.Intrinsics(model, width, height, fx, fy, ppx, ppy)
        self.d_int = ra.Intrinsics(
            str(depth_intr.get("model", "distortion.brown_conrady")),
            depth_intr["width"],
            depth_intr["height"],
            depth_intr["fx"],
            depth_intr["fy"],
            depth_intr["ppx"],
            depth_intr["ppy"],
        )
        self.c_int = ra.Intrinsics(
            str(target_intr.get("model", "distortion.brown_conrady")),
            target_intr["width"],
            target_intr["height"],
            target_intr["fx"],
            target_intr["fy"],
            target_intr["ppx"],
            target_intr["ppy"],
        )
        self.extrinsics = None
        if extrinsics:
            self.extrinsics = ra.Extrinsics(
                rotation=extrinsics["rotation"],
                translation=extrinsics["translation"],
            )

    def align(self, depth_image, target_image, depth_scale=0.001):
        if depth_image is None or target_image is None:
            return None
        if self.extrinsics:
            return ra.align_z_to_other(
                depth_image, target_image, self.d_int, self.c_int, depth_scale,
                depth_to_other=self.extrinsics,
            )
        return ra.align_z_to_other(depth_image, target_image, self.d_int, self.c_int, depth_scale)
