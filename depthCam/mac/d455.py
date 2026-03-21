#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 15:04:22 2025

@author: navidhasanzadeh
"""

import os
import sys
import ctypes
import time

import numpy as np
import cv2

# ---------------------------------------------------------------------
# 1) Point this to your compiled librealsense2 dylib on macOS
# ---------------------------------------------------------------------
LIBREALSENSE_PATH = "/Users/navidhasanzadeh/Documents/realsense/librealsense2.2.56.5.dylib"  # <-- change if needed

# Optionally, make sure the folder is on DYLD_LIBRARY_PATH for the current process
lib_dir = os.path.dirname(LIBREALSENSE_PATH)
if lib_dir and lib_dir not in os.environ.get("DYLD_LIBRARY_PATH", ""):
    os.environ["DYLD_LIBRARY_PATH"] = lib_dir + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")

# Load the shared library explicitly (this should succeed without error)
try:
    rs_lib = ctypes.CDLL(LIBREALSENSE_PATH)
    print(f"Loaded librealsense library from: {LIBREALSENSE_PATH}")
except OSError as e:
    print(f"Failed to load librealsense library at {LIBREALSENSE_PATH}: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------
# 2) Import pyrealsense2 (must be built/installed against your dylib)
# ---------------------------------------------------------------------
try:
    import pyrealsense2 as rs
except ImportError as e:
    print("Could not import pyrealsense2. Make sure you built/install the Python bindings "
          "with librealsense and that they are on PYTHONPATH.")
    print(f"ImportError: {e}")
    sys.exit(1)

def main():
    # -----------------------------------------------------------------
    # 3) Configure and start the RealSense pipeline for D455
    # -----------------------------------------------------------------
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable depth and color streams; adjust resolution/frame rate if desired
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        profile = pipeline.start(config)
        print("D455 streaming started.")
    except Exception as e:
        print(f"Failed to start RealSense pipeline: {e}")
        return

    # Colorizer for depth visualization
    colorizer = rs.colorizer()

    try:
        while True:
            # ---------------------------------------------------------
            # 4) Wait for frames from the camera
            # ---------------------------------------------------------
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # ---------------------------------------------------------
            # 5) Convert to numpy arrays
            # ---------------------------------------------------------
            depth_color_frame = colorizer.colorize(depth_frame)
            depth_image = np.asanyarray(depth_color_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Optional: resize if you want them smaller/larger
            # color_image = cv2.resize(color_image, (640, 480))
            # depth_image = cv2.resize(depth_image, (640, 480))

            # Stack color and depth side by side
            combined = np.hstack((color_image, depth_image))

            # ---------------------------------------------------------
            # 6) Show the video
            # ---------------------------------------------------------
            cv2.imshow("D455 Color (left) + Depth (right)", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC to quit
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        # -------------------------------------------------------------
        # 7) Cleanup
        # -------------------------------------------------------------
        print("Stopping pipeline and closing window...")
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
