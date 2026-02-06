import cv2
import numpy as np
import argparse
import realsense_align as ra
from realsense_frame.loader import SessionLoader

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
                                   color_intr["ppx"], color_intr["ppy"])
        
        # Identity Extrinsics (Rotation 3x3 flat, Translation 1x3)
        self.extrin = ra.Extrinsics([1.0, 0.0, 0.0, 
                                     0.0, 1.0, 0.0, 
                                     0.0, 0.0, 1.0], 
                                    [0.0, 0.0, 0.0])

    def align(self, depth_image, color_image, depth_scale=0.001):
        if depth_image is None or color_image is None: return None
        
        # ra.align_z_to_other(depth_frame, color_frame, depth_intrin, color_intrin, extrin, scale)
        return ra.align_z_to_other(depth_image, color_image, self.d_int, self.c_int, self.extrin, depth_scale)

def main():
    parser = argparse.ArgumentParser(description="Visualize RealSense Capture Session")
    parser.add_argument("session_path", help="Path to the session directory")
    args = parser.parse_args()

    try:
        loader = SessionLoader(args.session_path)
    except Exception as e:
        print(f"Failed to load session: {e}")
        return

    c_intr, d_intr = loader.get_intrinsics()
    aligner = None
    if c_intr and d_intr:
        print("Intrinsics found. Using realsense-align.")
        try:
            aligner = RealSenseAligner(c_intr, d_intr)
        except Exception as e:
            print(f"Failed to initialize aligner: {e}")
    else:
        print("Warning: Intrinsics missing. Alignment disabled.")

    print(f"Loaded {len(loader)} frames. Controls:")
    print("  [n/Right]: Next Frame")
    print("  [p/Left]:  Previous Frame")
    print("  [q]:       Quit")

    idx = 0
    while True:
        frame = loader.get_frame(idx)
        
        # Prepare Images
        images = []
        if frame.color is not None:
            images.append(frame.color)
        
        aligned_d_vis = None
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
