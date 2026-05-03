"""
face_smoother.py — Temporal smoothing for face bounding boxes.

Eliminates the per-frame jitter produced by raw face detectors by running
each component of the bounding box (x, y, w, h) through an exponential
moving average (EMA) with a dead-zone guard and a grace-period buffer.

────────────────────────────────────────────────────────────
How it works
────────────────────────────────────────────────────────────

EMA formula (applied per component):
    smoothed = alpha × new  +  (1 - alpha) × previous

Dead-zone:
    If the raw change in a component is below dead_zone_px pixels, the
    previous smoothed value is kept unchanged.  This suppresses sub-pixel
    detector noise without making the overlay feel sticky on real head movement.

Grace period:
    When the detector returns no faces for up to max_miss_frames frames the
    last known smoothed box is returned unchanged.  This prevents the overlay
    from flickering off during rapid head turns or brief occlusions.
    Once the grace period expires the overlay is hidden (returns None).

────────────────────────────────────────────────────────────
Tuning guide
────────────────────────────────────────────────────────────

alpha (default 0.30)
    0.15–0.25  →  very smooth, slight lag for fast head turns
    0.30–0.40  →  good balance  ← recommended starting point
    0.50–0.70  →  highly responsive, still reduces high-freq noise

dead_zone_px (default 6)
    3–5   →  tight tracking; catches slow drift
    6–10  →  stable at rest; recommended for face overlays
    >12   →  can feel 'sticky' during slow movement

max_miss_frames (default 8  ≈ 267 ms at 30 fps)
    4–6   →  hide overlay quickly when face leaves frame
    8–12  →  tolerant of brief blinks / look-aways  ← recommended
    >15   →  overlay lingers a long time after face is gone
"""

from typing import List, Optional, Tuple

BoundingBox = Tuple[int, int, int, int]   # (x, y, w, h) in pixels


class FaceBoxSmoother:
    """
    Smooths a stream of raw bounding boxes into a stable, jitter-free box.

    Parameters
    ----------
    alpha : float
        EMA blend factor in (0, 1].  Smaller = smoother but laggier.
    dead_zone_px : int
        Per-component pixel threshold below which updates are suppressed.
    max_miss_frames : int
        Frames without a detection before the smoothed box is discarded.
    """

    def __init__(
        self,
        alpha:           float = 0.30,
        dead_zone_px:    int   = 6,
        max_miss_frames: int   = 8,
    ):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1]; got {alpha}")

        self._alpha       = alpha
        self._dead_zone   = float(dead_zone_px)
        self._max_miss    = max_miss_frames

        self._smooth:     Optional[List[float]] = None   # [x, y, w, h] floats
        self._miss_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, boxes: List[BoundingBox]) -> Optional[BoundingBox]:
        """
        Feed the raw detector output for the current frame.

        Parameters
        ----------
        boxes : list of (x, y, w, h) tuples from the face detector.
                Pass an empty list when the detector found nothing.

        Returns
        -------
        Smoothed (x, y, w, h) as ints, or None when the face is considered lost
        (grace period expired or no face was ever detected).
        """
        if not boxes:
            return self._handle_miss()

        # If multiple faces, track the largest (closest to camera)
        raw = max(boxes, key=lambda b: b[2] * b[3])
        return self._handle_hit(raw)

    def reset(self) -> None:
        """Discard all state; useful when switching camera sources."""
        self._smooth     = None
        self._miss_count = 0

    @property
    def has_face(self) -> bool:
        """True when the smoother has a valid box (not expired)."""
        return self._smooth is not None and self._miss_count <= self._max_miss

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_hit(self, raw: BoundingBox) -> BoundingBox:
        """Apply EMA to a valid new detection."""
        self._miss_count = 0
        nx, ny, nw, nh = float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])

        if self._smooth is None:
            # First ever detection — initialise directly (no lag on first frame)
            self._smooth = [nx, ny, nw, nh]
        else:
            self._smooth = [
                self._ema(self._smooth[0], nx),
                self._ema(self._smooth[1], ny),
                self._ema(self._smooth[2], nw),
                self._ema(self._smooth[3], nh),
            ]

        return self._as_int_box()

    def _handle_miss(self) -> Optional[BoundingBox]:
        """Return last known box during grace period; None once expired."""
        if self._smooth is None:
            return None                      # never had a detection
        self._miss_count += 1
        if self._miss_count > self._max_miss:
            return None                      # grace period expired → hide
        return self._as_int_box()           # hold last known position

    def _ema(self, prev: float, new: float) -> float:
        """
        EMA with dead-zone guard.

        If the change is smaller than dead_zone_px, the previous value is
        kept (suppresses jitter).  For larger changes the standard EMA
        blend is applied so the overlay smoothly follows real movement.
        """
        delta = new - prev
        if abs(delta) < self._dead_zone:
            return prev                      # change too small — ignore
        return prev + self._alpha * delta    # EMA: prev + α(new - prev)

    def _as_int_box(self) -> BoundingBox:
        """Convert the internal float state to an integer bounding box."""
        return tuple(max(0, int(round(v))) for v in self._smooth)  # type: ignore[return-value]
