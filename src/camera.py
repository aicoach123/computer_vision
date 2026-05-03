"""
camera.py — Manages webcam initialization, frame reading, and cleanup.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class CameraCapture:
    """Wraps an OpenCV VideoCapture with context-manager support."""

    def __init__(self, camera_index: int = 0):
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {camera_index}. "
                "Check that a webcam is connected and not in use by another app."
            )
        # Prefer 30 fps; camera firmware will negotiate the actual rate
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"[INFO] Camera {camera_index} opened — "
              f"{int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
              f"@ {int(self._cap.get(cv2.CAP_PROP_FPS))} fps")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame. Returns (success, frame_or_None)."""
        ret, frame = self._cap.read()
        return ret, frame if ret else None

    def release(self) -> None:
        """Release the underlying VideoCapture resource."""
        if self._cap.isOpened():
            self._cap.release()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraCapture":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
