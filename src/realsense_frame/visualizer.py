import click
import cv2
import numpy as np
from realsense_frame.loader import SessionLoader, RealSenseAligner
from loguru import logger
from realsense_frame.utils import (
    depth_to_pointcloud,
    write_pointcloud_to_ply,
    create_colorbar,
)
import os  # Import os for path manipulation


@click.command(name="visualize")
@click.argument(
    "session_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option(
    "--export-ply",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Export point clouds to PLY files in the specified directory.",
)
def visualize_command(session_path, export_ply):
    """Visualize a RealSense capture session."""
    if export_ply and not os.path.exists(export_ply):
        os.makedirs(export_ply)

    try:
        loader = SessionLoader(session_path)
    except Exception as e:
        raise click.ClickException(f"Failed to load session: {e}")

    aligner = loader.aligner
    intrinsics = loader.get_intrinsics()
    c_intr = intrinsics["color"] or intrinsics["infra1"]
    if aligner:
        logger.info("Intrinsics found. Using realsense-align.")
    else:
        logger.warning(
            "Intrinsics missing or failed to initialize. Alignment disabled."
        )

    logger.info(f"Loaded {len(loader)} frames. Controls:")
    logger.info("  [n/Right]: Next Frame")
    logger.info("  [p/Left]:  Previous Frame")
    logger.info("  [q]:       Quit")
    if export_ply:
        logger.info(f"  PLY export enabled to: {export_ply}")

    idx = 0
    while True:
        frame = loader.get_frame(idx)

        # Prepare Images
        valid_imgs = []
        if frame.color is not None:
            valid_imgs.append(frame.color)

        if frame.infra1 is not None:
            valid_imgs.append(
                cv2.cvtColor(frame.infra1, cv2.COLOR_GRAY2BGR)
                if len(frame.infra1.shape) == 2
                else frame.infra1
            )

        if frame.infra2 is not None:
            valid_imgs.append(
                cv2.cvtColor(frame.infra2, cv2.COLOR_GRAY2BGR)
                if len(frame.infra2.shape) == 2
                else frame.infra2
            )

        aligned_depth = None
        if frame.depth is not None:
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(frame.depth, alpha=0.03), cv2.COLORMAP_JET
            )

            # Generate and add colorbar for depth
            colorbar = create_colorbar(depth_vis.shape[0])
            depth_with_colorbar = np.hstack((depth_vis, colorbar))
            valid_imgs.append(depth_with_colorbar)

            if aligner and frame.color is not None:
                depth_scale = frame.metadata.get("depth_units", 0.001)
                aligned_depth = aligner.align(frame.depth, frame.color, depth_scale)
                # We could add aligned_depth to valid_imgs if desired,
                # but for now let's keep it consistent with capture

        # Tile images in a grid (same logic as capture)
        if valid_imgs:
            count = len(valid_imgs)
            h, w = valid_imgs[0].shape[:2]
            resized = [
                cv2.resize(img, (w, h)) if img.shape[:2] != (h, w) else img
                for img in valid_imgs
            ]

            if count == 1:
                display = resized[0]
            elif count == 2:
                display = np.hstack(resized)
            elif count <= 4:
                top = np.hstack(resized[:2])
                bottom = np.hstack(
                    resized[2:] + [np.zeros_like(resized[0])] * (2 - len(resized[2:]))
                )
                display = np.vstack([top, bottom])
            else:
                display = resized[0]
        else:
            display = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                display,
                "No Data",
                (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        # PLY Export
        if (
            export_ply
            and aligned_depth is not None
            and frame.color is not None
            and c_intr is not None
        ):
            export_filename = os.path.join(export_ply, f"frame_{idx:05d}.ply")
            points, colors = depth_to_pointcloud(
                aligned_depth,
                frame.color,
                c_intr,
                frame.metadata.get("depth_units", 0.001),
            )
            write_pointcloud_to_ply(export_filename, points, colors)
            logger.debug(f"Exported {export_filename}")

        # Overlay Info
        info_txt = [
            f"Frame: {idx+1}/{len(loader)}",
            f"TS: {frame.timestamp}",
            f"IMU Samples: {len(frame.imu_samples)}",
        ]

        if frame.imu_samples:
            last_imu = frame.imu_samples[-1]
            info_txt.append(f"Last IMU ({last_imu['type']}):")
            info_txt.append(
                f" x={last_imu['x']:.2f}, y={last_imu['y']:.2f}, z={last_imu['z']:.2f}"
            )

        y0, dy = 30, 25
        for i, line in enumerate(info_txt):
            cv2.putText(
                display,
                line,
                (10, y0 + i * dy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Session Visualizer", display)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n") or key == 83:  # Right arrow
            idx = min(idx + 1, len(loader) - 1)
        elif key == ord("p") or key == 81:  # Left arrow
            idx = max(idx - 1, 0)

    cv2.destroyAllWindows()


def main():
    visualize_command()


if __name__ == "__main__":
    main()
