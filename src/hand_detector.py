"""
hand_detector.py — Detects both hands and returns structured landmark data.

Tries mediapipe.solutions.hands first (older mediapipe).
Falls back to the Tasks API HandLandmarker (newer mediapipe ≥ 0.10).
Raises RuntimeError if neither is available.
"""

import os
import sys
import cv2
import numpy as np
from typing import List, Dict

# Shared SSL-safe downloader lives in utils (same src/ package)
sys.path.insert(0, os.path.dirname(__file__))
from utils import download_model

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_PATH = os.path.join(_MODELS_DIR, "hand_landmarker.task")


class HandDetector:
    """
    Detects up to 2 hands per frame and returns their landmarks.

    Each detected hand is represented as a dict:
        {
            'landmarks'  : list of 21 (x, y, z) tuples, normalised to [0, 1]
            'handedness' : 'Left' | 'Right'  (from camera's perspective)
            'score'      : float [0, 1]
        }
    """

    def __init__(
        self,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._det  = None
        self._mode = None
        self._init(num_hands, min_detection_confidence, min_tracking_confidence)

    # ------------------------------------------------------------------
    # Initialisation — solutions API → Tasks API
    # ------------------------------------------------------------------

    def _init(self, n: int, min_det: float, min_track: float) -> None:
        # Attempt 1 — solutions API (mediapipe < ~0.10.18)
        try:
            import mediapipe as mp
            det = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=n,
                min_detection_confidence=min_det,
                min_tracking_confidence=min_track,
            )
            self._det  = det
            self._mode = "solutions"
            print("[INFO] Hand detector: mediapipe.solutions.hands")
            return
        except AttributeError:
            pass
        except Exception as exc:
            print(f"[WARNING] solutions.hands init failed: {exc}")

        # Attempt 2 — Tasks API (mediapipe ≥ 0.10)
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            print(f"[INFO] mediapipe version: {mp.__version__}")
            download_model(_MODEL_URL, _MODEL_PATH, "hand_landmarker.task (~8 MB)")

            base_opts = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)

            # running_mode is REQUIRED by the Tasks API — IMAGE for per-frame sync inference
            try:
                opts = mp_vision.HandLandmarkerOptions(
                    base_options=base_opts,
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_hands=n,
                    min_hand_detection_confidence=min_det,
                    min_hand_presence_confidence=min_det,
                    min_tracking_confidence=min_track,
                )
            except TypeError:
                # Older Tasks API build without min_hand_presence_confidence
                opts = mp_vision.HandLandmarkerOptions(
                    base_options=base_opts,
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_hands=n,
                    min_hand_detection_confidence=min_det,
                    min_tracking_confidence=min_track,
                )

            self._det  = mp_vision.HandLandmarker.create_from_options(opts)
            self._mode = "tasks"
            self._mp   = mp
            print("[INFO] Hand detector: MediaPipe Tasks API HandLandmarker")
            return
        except Exception as exc:
            # Print full traceback so the real cause is always visible
            import traceback
            traceback.print_exc()
            raise RuntimeError(
                f"HandDetector: could not initialise either MediaPipe API.\n"
                f"  Reason: {exc}\n"
                f"  mediapipe version: {__import__('mediapipe').__version__}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Detect hands in a BGR frame; returns a list of hand dicts."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return (
            self._detect_solutions(rgb)
            if self._mode == "solutions"
            else self._detect_tasks(rgb)
        )

    # ------------------------------------------------------------------
    # Back-end implementations
    # ------------------------------------------------------------------

    def _detect_solutions(self, rgb: np.ndarray) -> List[Dict]:
        res   = self._det.process(rgb)
        hands: List[Dict] = []
        if not res.multi_hand_landmarks:
            return hands
        for i, lm_set in enumerate(res.multi_hand_landmarks):
            landmarks = [(lm.x, lm.y, lm.z) for lm in lm_set.landmark]
            label, score = "Unknown", 0.0
            if res.multi_handedness and i < len(res.multi_handedness):
                cls   = res.multi_handedness[i].classification[0]
                label = cls.label
                score = cls.score
            hands.append({"landmarks": landmarks, "handedness": label, "score": score})
        return hands

    def _detect_tasks(self, rgb: np.ndarray) -> List[Dict]:
        import mediapipe as mp
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res    = self._det.detect(mp_img)
        hands: List[Dict] = []
        for i, lm_set in enumerate(res.hand_landmarks):
            landmarks = [(lm.x, lm.y, lm.z) for lm in lm_set]
            label, score = "Unknown", 0.0
            if res.handedness and i < len(res.handedness):
                cls   = res.handedness[i][0]
                label = cls.category_name
                score = cls.score
            hands.append({"landmarks": landmarks, "handedness": label, "score": score})
        return hands

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        if hasattr(self._det, "close"):
            self._det.close()

    def __enter__(self):  return self
    def __exit__(self, *_): self.close()
