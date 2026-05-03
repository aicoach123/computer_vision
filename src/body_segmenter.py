"""
body_segmenter.py — Extracts a person/body mask from a webcam frame.

Priority chain (tries each in order, uses the first that succeeds):
  1. mediapipe.solutions.selfie_segmentation  — fastest, needs older mediapipe
  2. MediaPipe Tasks API ImageSegmenter        — works on newer mediapipe ≥ 0.10
  3. OpenCV MOG2 background subtractor        — pure OpenCV, no ML model needed
                                                (needs ~3 s of static background
                                                 at startup to calibrate)
"""

import os
import sys
import cv2
import numpy as np
from typing import Tuple

sys.path.insert(0, os.path.dirname(__file__))
from utils import download_model

_MODELS_DIR     = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_MODEL_URL      = (
    "https://storage.googleapis.com/mediapipe-models/"
    "image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)
_MODEL_PATH     = os.path.join(_MODELS_DIR, "selfie_segmenter.tflite")
_CONFIDENCE_THR = 0.45   # person vs. background decision threshold


class BodySegmenter:
    """
    Generates a foreground (person) binary mask and person-only cutout.

    segment(frame) → (person_img, mask)
        person_img : BGR frame with background zeroed out
        mask       : uint8 H×W — 255 = person, 0 = background, edges feathered
    """

    def __init__(self, model_selection: int = 1):
        self._seg  = None
        self._mode = None
        self._init(model_selection)

    # ------------------------------------------------------------------
    # Initialisation chain
    # ------------------------------------------------------------------

    def _init(self, model_selection: int) -> None:
        if self._try_solutions(model_selection):
            return
        if self._try_tasks():
            return
        self._use_mog2()

    def _try_solutions(self, model_selection: int) -> bool:
        try:
            import mediapipe as mp
            seg = mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=model_selection
            )
            self._seg  = seg
            self._mode = "solutions"
            print("[INFO] Body segmenter: mediapipe.solutions.selfie_segmentation")
            return True
        except Exception:
            return False

    def _try_tasks(self) -> bool:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            download_model(_MODEL_URL, _MODEL_PATH, "selfie_segmenter.tflite (~256 KB)")

            base_opts = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
            opts = mp_vision.ImageSegmenterOptions(
                base_options=base_opts,
                running_mode=mp_vision.RunningMode.IMAGE,
                output_confidence_masks=True,
            )
            self._seg  = mp_vision.ImageSegmenter.create_from_options(opts)
            self._mode = "tasks"
            self._mp   = mp
            print("[INFO] Body segmenter: MediaPipe Tasks API ImageSegmenter")
            return True
        except Exception as exc:
            print(f"[WARNING] Tasks API segmenter unavailable: {exc}")
            return False

    def _use_mog2(self) -> None:
        self._seg  = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=50, detectShadows=False
        )
        self._mode = "mog2"
        print(
            "[INFO] Body segmenter: OpenCV MOG2 (fallback — stand still for "
            "~3 s at startup so the background can be calibrated)."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (person_img BGR, mask uint8 H×W)."""
        if self._mode == "solutions":
            return self._seg_solutions(frame)
        elif self._mode == "tasks":
            return self._seg_tasks(frame)
        else:
            return self._seg_mog2(frame)

    # ------------------------------------------------------------------
    # Back-end implementations
    # ------------------------------------------------------------------

    def _seg_solutions(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._seg.process(rgb)
        return self._build_output(frame, result.segmentation_mask)

    def _seg_tasks(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        import mediapipe as mp
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._seg.segment(mp_img)
        conf   = result.confidence_masks[0].numpy_view()   # float32 H×W
        return self._build_output(frame, conf)

    def _seg_mog2(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raw    = self._seg.apply(frame)
        mask   = np.where(raw == 255, np.uint8(255), np.uint8(0))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.GaussianBlur(mask, (21, 21), 0)
        person_img = cv2.bitwise_and(frame, frame, mask=mask)
        return person_img, mask

    @staticmethod
    def _build_output(
        frame: np.ndarray, confidence: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert a float32 confidence map → uint8 mask → person cutout."""
        mask = (confidence > _CONFIDENCE_THR).astype(np.uint8) * 255
        # Feather edges so clone layers dissolve naturally
        mask = cv2.GaussianBlur(mask, (13, 13), 0)
        person_img = cv2.bitwise_and(frame, frame, mask=mask)
        return person_img, mask

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        if hasattr(self._seg, "close"):
            self._seg.close()

    def __enter__(self):   return self
    def __exit__(self, *_): self.close()
