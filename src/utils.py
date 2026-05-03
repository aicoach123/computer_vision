"""
utils.py — Shared helper functions: asset loading, validation, FPS counter.
"""

import os
import ssl
import time
import urllib.request
import cv2
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Model download helper
# ---------------------------------------------------------------------------

def download_model(url: str, path: str, label: str = "") -> None:
    """
    Download a model file to `path` if it is missing or empty.

    Uses an unverified SSL context to work around the macOS Python.org
    issue where system CA certificates are not bundled with Python 3.12+.
    This is safe because the URL is a hardcoded Google Storage address.
    """
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    name = label or os.path.basename(path)
    print(f"[INFO] Downloading {name}…")

    # Bypass macOS SSL cert issue (Python 3.12 from python.org lacks certs)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode    = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(url, context=ctx) as resp:
            data = resp.read()
        with open(path, "wb") as f:
            f.write(data)
        size_mb = len(data) / 1_048_576
        print(f"[INFO] Saved {size_mb:.1f} MB → {path}")
    except Exception as exc:
        if os.path.exists(path):
            os.remove(path)   # remove partial file so next run retries
        raise RuntimeError(f"Download failed for {url}: {exc}") from exc


# ---------------------------------------------------------------------------
# Asset helpers
# ---------------------------------------------------------------------------

def remove_white_background(image: np.ndarray, threshold: int = 220) -> np.ndarray:
    """
    Convert near-white pixels to transparent by building an alpha mask from
    each pixel's Euclidean distance from pure white (255, 255, 255).

    Pixels closer to white become more transparent; pixels further away
    (i.e. actual colours) keep their opacity. The ramp is smooth so edges
    don't get a hard fringe.

    Only call this when the image already has an alpha channel (shape H×W×4).
    """
    bgr = image[:, :, :3].astype(np.float32)

    # Distance 0 = pure white, distance ~441 = pure black
    dist_from_white = np.sqrt(np.sum((bgr - 255.0) ** 2, axis=2))

    # Ramp: 0→0 (white → fully transparent), threshold→255 (colour → opaque)
    computed_alpha = np.clip(dist_from_white / threshold * 255, 0, 255).astype(np.uint8)

    # Keep whichever alpha value is smaller: the original or the computed mask.
    # This means we never make an already-transparent pixel more opaque.
    image[:, :, 3] = np.minimum(image[:, :, 3], computed_alpha)
    return image


def load_image_with_alpha(image_path: str) -> Optional[np.ndarray]:
    """
    Load a PNG as a BGRA (4-channel) NumPy array and pre-process it:

    1. If the image has no alpha channel, one is added.
    2. If the alpha channel is entirely opaque (white-background PNG), the
       white background is removed automatically via remove_white_background().

    Returns None and prints an error if the file is missing or unreadable.
    """
    if not os.path.isfile(image_path):
        print(f"[ERROR] Asset not found: {image_path}")
        return None

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"[ERROR] OpenCV could not decode: {image_path}")
        return None

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 3:
        alpha = np.full((image.shape[0], image.shape[1], 1), 255, dtype=np.uint8)
        image = np.concatenate([image, alpha], axis=2)

    # Auto-detect white-background PNGs (alpha channel is all 255 = no
    # real transparency was encoded) and strip the background automatically.
    if np.all(image[:, :, 3] == 255):
        print(f"[INFO] No transparency detected in "
              f"'{os.path.basename(image_path)}' — removing white background.")
        image = remove_white_background(image)

    return image  # shape (H, W, 4) — BGRA


def validate_assets(*paths: str) -> bool:
    """Return True only when every supplied path points to an existing file."""
    ok = True
    for p in paths:
        if not os.path.isfile(p):
            print(f"[ERROR] Missing required asset: {p}")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# FPS counter
# ---------------------------------------------------------------------------

class FPSCounter:
    """Rolling average FPS counter using the last N frames."""

    def __init__(self, window: int = 30):
        self._times: list = []
        self._window = window

    def tick(self) -> float:
        """Record a new frame timestamp and return current average FPS."""
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


def draw_hud(frame: np.ndarray, fps: float, num_faces: int) -> None:
    """Burn a minimal HUD (FPS + face count) into the frame in-place."""
    text = f"FPS: {fps:.1f}  |  Faces: {num_faces}  |  [q] Quit"
    # Black outline for readability on any background
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 230, 80), 1, cv2.LINE_AA)
