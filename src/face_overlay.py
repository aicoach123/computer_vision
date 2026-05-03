"""
face_overlay.py — Resizes and alpha-blends a transparent PNG Naruto overlay
onto each detected face in the frame.
"""

import cv2
import numpy as np
from typing import List, Tuple

BoundingBox = Tuple[int, int, int, int]


class FaceOverlay:
    """
    Blends a BGRA overlay image over detected faces.

    Parameters
    ----------
    overlay_image : np.ndarray
        BGRA image (H×W×4) loaded with cv2.IMREAD_UNCHANGED.
    scale_factor : float
        Scales the overlay width relative to the detected face width.
        Height is derived automatically to preserve the image's aspect ratio,
        so the Naruto face never looks squished or stretched.
    head_coverage : float
        Fraction of the overlay height placed *above* the detected face centre.
        Tune this so the face in the overlay aligns with the real face:
          - Raise it to pull the image up (show more hair / headband).
          - Lower it to push the image down.
    feather_px : int
        Radius of the Gaussian blur applied to the alpha channel edges.
        Larger values = softer, more natural transitions into the background.
    """

    def __init__(
        self,
        overlay_image: np.ndarray,
        scale_factor: float = 1.8,
        head_coverage: float = 0.65,
        feather_px: int = 6,
    ):
        if overlay_image is None:
            raise ValueError("overlay_image must not be None.")

        self._overlay = overlay_image
        self._scale = scale_factor
        self._head_coverage = head_coverage
        self._feather_px = feather_px

        # Pre-compute the natural aspect ratio so every resize stays faithful.
        h, w = overlay_image.shape[:2]
        self._aspect = h / w  # e.g. 1.3 for a portrait-style face PNG

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, frame: np.ndarray, bounding_boxes: List[BoundingBox]) -> np.ndarray:
        """
        Draw the Naruto overlay on every detected face.
        Returns a new frame; the original is not mutated.
        """
        result = frame.copy()
        for bbox in bounding_boxes:
            result = self._draw_one(result, bbox)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_one(self, frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """Compute overlay size/position for one face and blend it in."""
        x, y, fw, fh = bbox

        # Width scales with the face; height is locked to the image's aspect
        # ratio so we never distort Naruto's face.
        ow = int(fw * self._scale)
        oh = int(ow * self._aspect)

        if ow < 1 or oh < 1:
            return frame

        # Resize with INTER_LANCZOS4 for the sharpest downscale quality.
        resized = cv2.resize(self._overlay, (ow, oh), interpolation=cv2.INTER_LANCZOS4)

        # Feather the alpha channel so overlay edges dissolve into the webcam
        # frame instead of cutting in with a hard border.
        resized = self._feather_alpha(resized, self._feather_px)

        # Anchor: face centre in the webcam frame.
        face_cx = x + fw // 2
        face_cy = y + fh // 2

        # Top-left corner of the overlay.
        # head_coverage controls how far above the face centre the overlay sits.
        ox = face_cx - ow // 2
        oy = face_cy - int(oh * self._head_coverage)

        return self._alpha_blend(frame, resized, ox, oy)

    @staticmethod
    def _feather_alpha(image: np.ndarray, radius: int) -> np.ndarray:
        """
        Blur only the alpha channel so colour pixels at the edge fade out
        smoothly rather than cutting in hard.
        """
        if radius < 1 or image.shape[2] < 4:
            return image

        k = radius * 2 + 1  # kernel must be odd
        alpha = image[:, :, 3].astype(np.float32)
        blurred = cv2.GaussianBlur(alpha, (k, k), radius * 0.5)
        out = image.copy()
        out[:, :, 3] = blurred.astype(np.uint8)
        return out

    @staticmethod
    def _alpha_blend(
        frame: np.ndarray,
        overlay_bgra: np.ndarray,
        ox: int,
        oy: int,
    ) -> np.ndarray:
        """
        Alpha-blend overlay_bgra onto frame at top-left (ox, oy).
        Clips gracefully when the overlay extends beyond the frame boundary.
        Uses float32 arithmetic for accurate per-pixel blending.
        """
        fh, fw = frame.shape[:2]
        oh, ow = overlay_bgra.shape[:2]

        # Clamped destination rectangle in the frame
        dst_x1 = max(ox, 0)
        dst_y1 = max(oy, 0)
        dst_x2 = min(ox + ow, fw)
        dst_y2 = min(oy + oh, fh)

        if dst_x1 >= dst_x2 or dst_y1 >= dst_y2:
            return frame  # entirely off-screen

        # Matching source rectangle in the overlay
        src_x1 = dst_x1 - ox
        src_y1 = dst_y1 - oy
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)

        src = overlay_bgra[src_y1:src_y2, src_x1:src_x2]
        dst = frame[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)

        alpha = src[:, :, 3:4].astype(np.float32) / 255.0
        src_bgr = src[:, :, :3].astype(np.float32)

        blended = src_bgr * alpha + dst * (1.0 - alpha)
        frame[dst_y1:dst_y2, dst_x1:dst_x2] = blended.astype(np.uint8)
        return frame
