"""
face_detector.py — Real-time face detection using OpenCV Haar Cascade.
Returns pixel-space bounding boxes for every detected face in a frame.
"""

import cv2
import numpy as np
from typing import List, Tuple


# Type alias: (x, y, width, height) all in pixels
BoundingBox = Tuple[int, int, int, int]


class FaceDetector:
    """
    Wraps OpenCV's Haar Cascade frontal-face detector.
    Bundled with opencv-python — no extra downloads needed.
    """

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_face_size: int = 60,
    ):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._detector = cv2.CascadeClassifier(cascade_path)
        if self._detector.empty():
            raise RuntimeError(
                f"Could not load Haar cascade from {cascade_path}. "
                "Reinstall opencv-python."
            )
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = (min_face_size, min_face_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_faces(self, frame: np.ndarray) -> List[BoundingBox]:
        """
        Detect faces in a BGR frame.

        Returns a list of (x, y, w, h) pixel bounding boxes, one per face.
        Returns an empty list when no faces are found.
        """
        # Haar cascade works on grayscale; histogram equalisation helps in
        # variable lighting conditions (e.g. backlighting, dim rooms)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        detections = self._detector.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(detections) == 0:
            return []

        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in detections]

    # ------------------------------------------------------------------
    # Context manager (no-op close, kept for API consistency)
    # ------------------------------------------------------------------

    def close(self) -> None:
        pass

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
