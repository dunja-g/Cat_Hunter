"""
Camera Diagnostic Tool — run on the Raspberry Pi to test & fix color issues.

Usage (on Pi):
    python3 camera_test.py

Saves test images to camera_test_output/ so you can inspect them.
"""

import os
import sys
import time

def main():
    # Try to import cv2
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("[ERROR] OpenCV not installed. Run: pip install opencv-python-headless")
        sys.exit(1)

    output_dir = "camera_test_output"
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Try to open the camera ──────────────────────────
    print("=" * 55)
    print("  Camera Diagnostic Tool")
    print("=" * 55)

    camera = None
    cam_type = None

    # Try picamera2 first
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "BGR888"}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(2)  # warm up
        camera = picam2
        cam_type = "picamera2"
        print("[OK] Using picamera2")
    except Exception as e:
        print(f"[SKIP] picamera2 not available: {e}")

    # Fallback to OpenCV
    if camera is None:
        for idx in range(4):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                time.sleep(1)  # warm up
                camera = cap
                cam_type = "opencv"
                print(f"[OK] Using OpenCV camera index {idx}")
                break
            cap.release()

    if camera is None:
        print("[ERROR] No camera found! Check your USB/CSI connection.")
        sys.exit(1)

    # ── Step 2: Capture a frame ─────────────────────────────────
    print("\nCapturing test frames...")

    if cam_type == "picamera2":
        frame = camera.capture_array()
    else:
        # Read a few frames to flush the buffer
        for _ in range(5):
            camera.read()
        ret, frame = camera.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera")
            sys.exit(1)

    h, w = frame.shape[:2]
    print(f"  Resolution: {w}x{h}")
    print(f"  Channels:   {frame.shape[2] if len(frame.shape) == 3 else 1}")
    print(f"  Dtype:      {frame.dtype}")

    # ── Step 3: Save multiple color versions ────────────────────
    # Original (as-is from camera)
    path_original = os.path.join(output_dir, "1_original.jpg")
    cv2.imwrite(path_original, frame)
    print(f"\n  Saved: {path_original} (raw from camera)")

    # BGR → RGB swap (fixes the yellow/blue inversion)
    frame_swapped = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    path_swapped = os.path.join(output_dir, "2_channels_swapped.jpg")
    cv2.imwrite(path_swapped, frame_swapped)
    print(f"  Saved: {path_swapped} (red/blue channels swapped)")

    # Also try just swapping R and B manually for comparison
    frame_rb = frame.copy()
    frame_rb[:, :, 0], frame_rb[:, :, 2] = frame[:, :, 2].copy(), frame[:, :, 0].copy()
    path_rb = os.path.join(output_dir, "3_rb_manual_swap.jpg")
    cv2.imwrite(path_rb, frame_rb)
    print(f"  Saved: {path_rb} (manual R↔B swap)")

    # ── Step 4: Print diagnosis ─────────────────────────────────
    # Check the average color of each channel to guess the order
    b_avg = frame[:, :, 0].mean()
    g_avg = frame[:, :, 1].mean()
    r_avg = frame[:, :, 2].mean()

    print(f"\n  Channel averages: ch0={b_avg:.1f}  ch1={g_avg:.1f}  ch2={r_avg:.1f}")
    print()
    print("=" * 55)
    print("  WHAT TO DO NEXT")
    print("=" * 55)
    print(f"  Look at the 3 images in '{output_dir}/' folder.")
    print("  Point the camera at something with a known color")
    print("  (like a red or blue object) and check which image")
    print("  shows the correct colors.")
    print()
    print("  If '1_original.jpg' looks correct → no fix needed!")
    print("  If '2_channels_swapped.jpg' looks correct → we need")
    print("     to add cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)")
    print("  If '3_rb_manual_swap.jpg' looks correct → same fix")
    print()

    # Cleanup
    if cam_type == "picamera2":
        camera.stop()
    else:
        camera.release()

    print("Done! Check the images and let me know which one looks right.")


if __name__ == "__main__":
    main()
