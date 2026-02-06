import click
import cv2
import numpy as np
import realsense_align as ra
from realsense_frame.loader import SessionLoader
import os # Import os for path manipulation

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
    colors = color_image.reshape(-1, 3) # Assuming BGR, will convert to RGB for PLY
    
    # Filter out invalid points (where depth is 0)
    valid_mask = (z.reshape(-1) > 0)
    points = points[valid_mask]
    colors = colors[valid_mask]

    return points, colors

def write_pointcloud_to_ply(filename, points, colors=None):
    """Writes a point cloud to a PLY file."""
    with open(filename, 'w') as f:
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
                f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]} {colors[i, 2]} {colors[i, 1]} {colors[i, 0]}\n")
            else:
                f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]}\n")

class RealSenseAligner:
    def __init__(self, color_intr, depth_intr):
        # Initialize Intrinsics using realsense-align API
        # Model 0 usually corresponds to None/distortion-free/Brown-Conrady in bindings
        # ra.Intrinsics(model, width, height, fx, fy, ppx, ppy)
        self.d_int = ra.Intrinsics(0, 
                                   depth_intr["width"], depth_intr["height"],
                                   depth_intr["fx"], depth_intr["fy"],
                                   depth_intr["ppx"], depth_intr["ppy"])
        
        self.c_int = ra.Intrinsics(0, 
                                   color_intr["width"], color_intr["height"],
                                   color_intr["fx"], color_intr["fy"],
                                   color_intr["ppx"], c_intr["ppy"])
        
        # Identity Extrinsics (Rotation 3x3 flat, Translation 1x3)
        self.extrin = ra.Extrinsics([1.0, 0.0, 0.0, 
                                     0.0, 1.0, 0.0, 
                                     0.0, 0.0, 1.0], 
                                    [0.0, 0.0, 0.0])

    def align(self, depth_image, color_image, depth_scale=0.001):
        if depth_image is None or color_image is None: return None
        
        # ra.align_z_to_other(depth_frame, color_frame, depth_intrin, color_intrin, extrin, scale)
        return ra.align_z_to_other(depth_image, color_image, self.d_int, self.c_int, self.extrin, depth_scale)

@click.command()
@click.argument("session_path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--export-ply', type=click.Path(file_okay=False, dir_okay=True), help='Export point clouds to PLY files in the specified directory.')
def main(session_path, export_ply):
    """Visualize a RealSense capture session."""
    if export_ply and not os.path.exists(export_ply):
        os.makedirs(export_ply)

    try:
        loader = SessionLoader(session_path)
    except Exception as e:
        raise click.ClickException(f"Failed to load session: {e}")

    c_intr, d_intr = loader.get_intrinsics()
    aligner = None
    if c_intr and d_intr:
        click.echo("Intrinsics found. Using realsense-align.")
        try:
            aligner = RealSenseAligner(c_intr, d_intr)
        except Exception as e:
            click.echo(f"Warning: Failed to initialize aligner: {e}")
    else:
        click.echo("Warning: Intrinsics missing. Alignment disabled.")

    click.echo(f"Loaded {len(loader)} frames. Controls:")
    click.echo("  [n/Right]: Next Frame")
    click.echo("  [p/Left]:  Previous Frame")
    click.echo("  [q]:       Quit")
    if export_ply:
        click.echo(f"  PLY export enabled to: {export_ply}")

    idx = 0
    while True:
        frame = loader.get_frame(idx)
        
        # Prepare Images
        images = []
        if frame.color is not None:
            images.append(frame.color)
        
        aligned_d_vis = None
        aligned_depth = None # Initialize aligned_depth for PLY export
        if frame.depth is not None:
            # Colorize Raw Depth
            depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(frame.depth, alpha=0.03), cv2.COLORMAP_JET)
            
            if aligner and frame.color is not None:
                # Perform Alignment
                depth_scale = frame.metadata.get("depth_units", 0.001)
                
                # Align depth to color resolution
                aligned_depth = aligner.align(frame.depth, frame.color, depth_scale)
                
                # Visualize Aligned Depth
                aligned_d_vis = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth, alpha=0.03), cv2.COLORMAP_MAGMA)
            
            # Ensure raw depth visualization matches color height for display
            if frame.color is not None and depth_vis.shape[:2] != frame.color.shape[:2]:
                depth_vis = cv2.resize(depth_vis, (frame.color.shape[1], frame.color.shape[0]))
            
            images.append(depth_vis)
            
        if aligned_d_vis is not None:
            images.append(aligned_d_vis)
            
        # PLY Export
        if export_ply and aligned_depth is not None and frame.color is not None and c_intr is not None:
            export_filename = os.path.join(export_ply, f"frame_{idx:05d}.ply")
            points, colors = depth_to_pointcloud(aligned_depth, frame.color, c_intr, frame.metadata.get("depth_units", 0.001))
            write_pointcloud_to_ply(export_filename, points, colors)
            click.echo(f"Exported {export_filename}")

        # Concatenate horizontally
        if images:
            h_max = max(img.shape[0] for img in images)
            padded_images = []
            for img in images:
                if img.shape[0] < h_max:
                    pad = np.zeros((h_max - img.shape[0], img.shape[1], 3), dtype=np.uint8)
                    padded_images.append(np.vstack([img, pad]))
                else:
                    padded_images.append(img)
            
            display = np.hstack(padded_images)
        else:
            display = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(display, "No Data", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # Overlay Info
        info_txt = [
            f"Frame: {idx+1}/{len(loader)}",
            f"TS: {frame.timestamp}",
            f"IMU Samples: {len(frame.imu_samples)}"
        ]
        
        if frame.imu_samples:
            last_imu = frame.imu_samples[-1]
            info_txt.append(f"Last IMU ({last_imu['type']}):")
            info_txt.append(f" x={last_imu['x']:.2f}, y={last_imu['y']:.2f}, z={last_imu['z']:.2f}")

        y0, dy = 30, 25
        for i, line in enumerate(info_txt):
            cv2.putText(display, line, (10, y0 + i*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        cv2.imshow("Session Visualizer", display)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n') or key == 83: # Right arrow
            idx = min(idx + 1, len(loader) - 1)
        elif key == ord('p') or key == 81: # Left arrow
            idx = max(idx - 1, 0)
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
