"""
stable_face_tracker.py — Single-person lock-on face tracker.

Locks onto the first valid detected face and tracks it smoothly across
frames using EMA with dead-zone suppression, per-component jump clamping,
and a configurable grace period.

Once locked, only detections whose centre is within LOCK_CENTER_DIST_NORM
(as a fraction of frame width) of the current smooth box are accepted.
This prevents other faces, Naruto overlays, or clone artefacts from
hijacking the tracker.

────────────────────────────────────────────────────────────────────────
Tuning guide
────────────────────────────────────────────────────────────────────────

SMOOTHING_ALPHA  (default 0.22)
    0.12–0.18  →  very smooth, slight lag on fast head movement
    0.20–0.25  →  good balance — recommended
    0.30–0.45  →  responsive, still filters high-freq noise

DEAD_ZONE_PX  (default 5)
    3–5   →  tight tracking; catches slow drift
    6–9   →  stable at rest; good default for face overlays
    >12   →  can feel sticky during slow movement

MAX_JUMP_PX  (default 60)
    40–60  →  prevents snap jumps from noisy Haar boxes
    >80    →  effectively disables the clamp

MAX_MISSING_FRAMES  (default 12  ≈ 400 ms at 30 fps)
    6–8    →  overlay hides quickly when face leaves frame
    10–15  →  tolerant of brief blinks / look-aways  ← recommended
    >20    →  overlay lingers a long time after face is gone

LOCK_CENTER_DIST_NORM  (default 0.25)
    0.15–0.20  →  strict lock; rejects detections far from last position
    0.25–0.35  →  tolerant of wider head swings  ← recommended
    >0.40      →  may pick up other people or artefacts
"""

import math
from typing import List, Optional, Tuple

BoundingBox = Tuple[int, int, int, int]   # (x, y, w, h) pixels

# ── Default constants (all overridable via constructor) ───────────────────────
SMOOTHING_ALPHA         = 0.22
DEAD_ZONE_PX            = 5
MAX_JUMP_PX             = 60
MAX_MISSING_FRAMES      = 12
LOCK_CENTER_DIST_NORM   = 0.25


class StableFaceTracker:
    """
    Single-person lock-on tracker with EMA smoothing.

    Workflow per frame
    ------------------
    1. Receive raw detector boxes.
    2. If not yet locked → pick the largest box and lock on.
    3. If locked → find the box whose centre is closest to the current
       smooth position AND within the normalised distance threshold.
    4. Feed the chosen box through EMA (with dead-zone + jump clamp).
    5. Return the stable integer box, or None if the face is lost.

    Parameters
    ----------
    alpha : float
        EMA blend factor in (0, 1].  Smaller = smoother but laggier.
    dead_zone_px : int
        Suppress per-component movements smaller than this.
    max_jump_px : int
        Clamp single-frame per-component jumps to this limit.
    max_missing_frames : int
        Hold the last known box for this many consecutive miss frames
        before unlocking and returning None.
    lock_center_dist_norm : float
        Maximum accepted centre-to-centre distance between a new
        detection and the current smooth box, expressed as a fraction
        of the frame width.
    """

    def __init__(
        self,
        alpha:                 float = SMOOTHING_ALPHA,
        dead_zone_px:          int   = DEAD_ZONE_PX,
        max_jump_px:           int   = MAX_JUMP_PX,
        max_missing_frames:    int   = MAX_MISSING_FRAMES,
        lock_center_dist_norm: float = LOCK_CENTER_DIST_NORM,
    ):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1]; got {alpha}")

        self._alpha      = alpha
        self._dead_zone  = float(dead_zone_px)
        self._max_jump   = float(max_jump_px)
        self._max_miss   = max_missing_frames
        self._dist_norm  = lock_center_dist_norm

        self._smooth:     Optional[List[float]] = None   # [x, y, w, h]
        self._miss_count: int  = 0
        self._locked:     bool = False

    # ── Public API ───────────────────────────────────────────────────────────

    def update(
        self,
        detected_faces: List[BoundingBox],
        frame_shape:    Tuple[int, ...],
    ) -> Optional[BoundingBox]:
        """
        Feed this frame's raw detector output and get the stable box back.

        Parameters
        ----------
        detected_faces : list of (x, y, w, h) from the raw detector.
                         Pass an empty list when nothing was detected.
        frame_shape    : frame.shape  (H, W[, C]) — used to normalise
                         the centre-distance threshold.

        Returns
        -------
        Smoothed (x, y, w, h) as ints, or None when the face is lost.
        """
        frame_w = frame_shape[1]
        best    = self._select(detected_faces, frame_w)

        if best is None:
            return self._handle_miss()

        self._miss_count = 0
        self._locked     = True
        return self._apply_ema(best)

    def reset(self) -> None:
        """Discard all state — call this when switching camera sources."""
        self._smooth     = None
        self._miss_count = 0
        self._locked     = False

    @property
    def is_locked(self) -> bool:
        """True when tracking a face with a valid smooth box."""
        return self._locked and self._smooth is not None

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _select(
        self,
        boxes:   List[BoundingBox],
        frame_w: int,
    ) -> Optional[BoundingBox]:
        """
        Choose which detected box to feed into the EMA this frame.

        Before lock  → largest box wins (biggest face = closest person).
        After lock   → box whose centre is nearest the current smooth
                       position, provided it is within the distance limit.
                       Returns None if no box qualifies (counts as a miss).
        """
        if not boxes:
            return None

        if not self._locked or self._smooth is None:
            return max(boxes, key=lambda b: b[2] * b[3])

        # Centre of the current smooth box
        cx = self._smooth[0] + self._smooth[2] * 0.5
        cy = self._smooth[1] + self._smooth[3] * 0.5
        max_dist = self._dist_norm * frame_w

        best_box:  Optional[BoundingBox] = None
        best_dist: float                 = float("inf")

        for box in boxes:
            bx, by, bw, bh = box
            dcx = bx + bw * 0.5
            dcy = by + bh * 0.5
            dist = math.hypot(dcx - cx, dcy - cy)
            if dist < max_dist and dist < best_dist:
                best_dist = dist
                best_box  = box

        return best_box  # None when no box is close enough → graceful miss

    def _handle_miss(self) -> Optional[BoundingBox]:
        """Hold last known box during grace period; unlock once expired."""
        if self._smooth is None:
            return None                          # never had a detection

        self._miss_count += 1
        if self._miss_count > self._max_miss:
            self._smooth  = None
            self._locked  = False
            return None                          # grace period expired

        return self._as_int_box()               # hold last known position

    def _apply_ema(self, raw: BoundingBox) -> BoundingBox:
        """Run EMA on all four box components and return an int box."""
        nx, ny, nw, nh = float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])

        if self._smooth is None:
            # First detection — initialise without lag
            self._smooth = [nx, ny, nw, nh]
        else:
            self._smooth = [
                self._ema(self._smooth[0], nx),
                self._ema(self._smooth[1], ny),
                self._ema(self._smooth[2], nw),
                self._ema(self._smooth[3], nh),
            ]

        return self._as_int_box()

    def _ema(self, prev: float, new: float) -> float:
        """
        EMA with dead-zone guard and jump clamp.

        1. Clamp the raw delta to [-max_jump, +max_jump] so a single noisy
           detection cannot cause a large position jump.
        2. If the clamped delta is smaller than dead_zone, keep prev
           unchanged (suppresses sub-pixel Haar cascade noise).
        3. Otherwise apply the standard EMA blend.
        """
        delta = new - prev
        # Clamp before smoothing so even large raw jumps are damped
        delta = max(-self._max_jump, min(self._max_jump, delta))
        if abs(delta) < self._dead_zone:
            return prev                          # too small — suppress
        return prev + self._alpha * delta        # EMA: prev + α(Δ)

    def _as_int_box(self) -> BoundingBox:
        return tuple(max(0, int(round(v))) for v in self._smooth)  # type: ignore[return-value]
