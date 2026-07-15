"""
Cat Detector — real-time inference module
Loads a trained model and classifies camera frames as cat / non-cat.

Usage:
    from cat_detector import CatDetector
    detector = CatDetector("models/best_transfer.pth")
    result = detector.classify_frame(bgr_numpy_image)
    # result = {"detected": True, "confidence": 0.94, "label": "cat"}
"""

import os
import sys
import time
import threading

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------------------------
# Add cat_detection/src to the path so we can import the model definitions
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_HERE, "cat_detection", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


class CatDetector:
    """
    Lightweight cat-vs-non-cat classifier for real-time camera frames.
    Designed to run on a Raspberry Pi (MobileNetV2 recommended).
    """

    # Class names in index order  (modular pipeline convention: 0=non-cat, 1=cat)
    CLASS_NAMES = ["non-cat", "cat"]
    CAT_INDEX = 1

    # ImageNet normalisation (matches the standalone training scripts)
    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, weights_path=None, model_name="mobilenet_v2",
                 confidence_threshold=0.65, device=None):
        """
        Parameters
        ----------
        weights_path : str or None
            Path to a .pth checkpoint.  If *None* the detector still works
            but uses a freshly-initialised model (useful for testing the
            pipeline before training is done).
        model_name : str
            "mobilenet_v2" (recommended for Pi) or "resnet50".
        confidence_threshold : float
            Minimum softmax probability to declare "cat detected".
        device : str or None
            Force "cpu" or "cuda".  Auto-detected if None.
        """
        self.confidence_threshold = confidence_threshold

        # Device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model
        from transfer_model import get_transfer_model
        self.model = get_transfer_model(model_name, freeze_backbone=False)

        # Load trained weights (if available)
        if weights_path and os.path.isfile(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device)
            # Support both raw state_dict and wrapped checkpoint
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"[CatDetector] Loaded weights from {weights_path}", flush=True)
        else:
            print("[CatDetector] No weights loaded — running with untrained model",
                  flush=True)

        self.model.to(self.device)
        self.model.eval()

        # Warm-up pass (first inference is always slow due to lazy init)
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=self.device)
            self.model(dummy)
        print(f"[CatDetector] Ready on {self.device}", flush=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_frame(self, bgr_frame):
        """
        Classify a single BGR numpy frame (OpenCV / picamera2 format).

        Returns
        -------
        dict  {"detected": bool, "confidence": float, "label": str,
               "fps_inference": float}
        """
        t0 = time.time()

        # BGR → RGB → PIL → tensor
        rgb = bgr_frame[:, :, ::-1]  # fast BGR→RGB via numpy slicing
        pil_img = Image.fromarray(rgb.astype(np.uint8))
        tensor = self._transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]

        cat_prob = probs[self.CAT_INDEX].item()
        predicted = int(probs.argmax())
        detected = predicted == self.CAT_INDEX and cat_prob >= self.confidence_threshold

        dt = time.time() - t0

        return {
            "detected": detected,
            "confidence": cat_prob,
            "label": self.CLASS_NAMES[predicted],
            "inference_ms": round(dt * 1000, 1),
        }


class DetectionLoop:
    """
    Runs cat detection in a background thread at a configurable interval.
    Designed to be plugged into web_control.py.

    Usage
    -----
        loop = DetectionLoop(detector, frame_reader_fn, on_result_fn, interval=1.0)
        loop.start()
        ...
        loop.stop()
    """

    def __init__(self, detector, read_frame_fn, on_result_fn, interval=1.0):
        """
        Parameters
        ----------
        detector : CatDetector
        read_frame_fn : callable
            Returns a BGR numpy frame (or None).
        on_result_fn : callable(dict)
            Called with each detection result dict.
        interval : float
            Seconds between inferences (default 1.0 — gives ~1 FPS
            detection on Pi, leaving CPU headroom for driving + camera).
        """
        self.detector = detector
        self.read_frame = read_frame_fn
        self.on_result = on_result_fn
        self.interval = interval
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[DetectionLoop] Started (interval={self.interval}s)", flush=True)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        print("[DetectionLoop] Stopped", flush=True)

    def _loop(self):
        while self._running:
            try:
                frame = self.read_frame()
                if frame is not None:
                    result = self.detector.classify_frame(frame)
                    self.on_result(result)
            except Exception as e:
                print(f"[DetectionLoop] Error: {e}", flush=True)
            time.sleep(self.interval)
