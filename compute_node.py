"""
Cat Hunter — Compute Node (runs on your laptop)

Pulls the live MJPEG stream from the Raspberry Pi, runs the 5-class
cat breed detector locally, and POSTs results back to the Pi so the
Web UI and Hunt Mode stay fully functional.

Usage:
    python compute_node.py --pi-ip 192.168.1.42
    python compute_node.py --pi-ip 192.168.1.42 --weights models/best_transfer.pth --interval 0.5
"""

import argparse
import time
import sys

import cv2
import requests

from cat_detector import CatDetector


def main():
    parser = argparse.ArgumentParser(
        description="Cat Hunter — laptop-side ML compute node"
    )
    parser.add_argument(
        "--pi-ip", required=True,
        help="IP address of the Raspberry Pi (e.g. 192.168.1.42)"
    )
    parser.add_argument(
        "--pi-port", type=int, default=5000,
        help="Port the Pi's web server is running on (default: 5000)"
    )
    parser.add_argument(
        "--weights", default="models/best_transfer.pth",
        help="Path to model weights on this machine (default: models/best_transfer.pth)"
    )
    parser.add_argument(
        "--model", default="mobilenet_v2",
        choices=["mobilenet_v2", "resnet50"],
        help="Model architecture to use (default: mobilenet_v2)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.65,
        help="Confidence threshold for detection (default: 0.65)"
    )
    parser.add_argument(
        "--interval", type=float, default=0.5,
        help="Seconds between inference cycles (default: 0.5)"
    )
    args = parser.parse_args()

    base_url = f"http://{args.pi_ip}:{args.pi_port}"
    stream_url = f"{base_url}/video_feed"
    api_url = f"{base_url}/api/detection"

    # ── Load model ──────────────────────────────────────────────
    print("=" * 55)
    print("  Cat Hunter — Compute Node (Laptop)")
    print("=" * 55)
    print(f"Loading model: {args.model} from {args.weights} ...")

    detector = CatDetector(
        weights_path=args.weights,
        model_name=args.model,
        confidence_threshold=args.threshold,
    )
    print("Model loaded successfully!")
    print()

    # ── Connect to Pi video stream ──────────────────────────────
    print(f"Connecting to Pi video stream: {stream_url}")
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print(f"[ERROR] Cannot connect to {stream_url}")
        print("Make sure:")
        print("  1. The Pi is running:  python3 web_control.py /dev/ttyACM0 --no-ml")
        print("  2. Your laptop is on the same WiFi network as the Pi")
        print(f"  3. You can open {base_url} in your browser")
        sys.exit(1)

    print("Connected! Starting inference loop...")
    print(f"  Interval:   {args.interval}s")
    print(f"  Threshold:  {args.threshold}")
    print(f"  Posting to: {api_url}")
    print()

    # ── Inference loop ──────────────────────────────────────────
    frame_count = 0
    error_streak = 0
    MAX_ERRORS = 10  # reconnect after this many consecutive failures

    while True:
        ret, frame = cap.read()

        if not ret:
            error_streak += 1
            if error_streak >= MAX_ERRORS:
                print(f"[WARN] Lost {MAX_ERRORS} frames in a row, reconnecting...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(stream_url)
                error_streak = 0
            time.sleep(0.1)
            continue

        error_streak = 0
        frame_count += 1

        # Run the model
        result = detector.classify_frame(frame)

        # POST result to the Pi
        payload = {
            "detected": result["detected"],
            "confidence": result["confidence"],
            "label": result["label"],
            "inference_ms": result.get("inference_ms", 0),
        }

        try:
            resp = requests.post(api_url, json=payload, timeout=2)
            status = "✓" if resp.status_code == 200 else f"HTTP {resp.status_code}"
        except requests.RequestException as e:
            status = f"SEND FAIL: {e}"

        # Console log
        if result["detected"]:
            print(
                f"[#{frame_count}] 🐱 {result['label']} "
                f"{result['confidence']*100:.1f}% "
                f"({result.get('inference_ms', 0):.0f}ms) → {status}"
            )
        elif frame_count % 20 == 0:
            # Print a heartbeat every 20 frames so you know it's alive
            print(f"[#{frame_count}] scanning... ({status})")

        time.sleep(args.interval)

    cap.release()
    print("Bye")


if __name__ == "__main__":
    main()
