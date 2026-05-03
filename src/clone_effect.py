"""
clone_effect.py — Naruto shadow clone renderer.

Stateless renderer: all duration / lifecycle state lives in the caller
(main.py).  Call calculate_fade_opacity() once per frame to get the
global alpha, then pass it to draw_clones() together with the person
cutout that was captured at trigger time.

Clone layout is generated from BASE_CLONE_OFFSETS × CLONE_COUNT_MULTIPLIER
levels.  With the defaults this produces 12 clones (4 base × 3 levels)
that spread progressively wider, each level more transparent than the last:

  Level 1 (×1)  offset  −220,  +220,  −430,  +430  →  82 % / 65 % opacity
  Level 2 (×2)  offset  −440,  +440,  −860,  +860  →  60 % / 48 % opacity
  Level 3 (×3)  offset  −660,  +660, −1290, +1290  →  42 % / 32 % opacity

Clones beyond the frame edge are silently clipped by _blit().

Each clone has its own fade-in ramp after it spawns (CLONE_APPEAR_SECS).
The global fade schedule across the full lifetime:
  0 s               → FADE_IN_SECS          : ramp  0 → 1
  FADE_IN_SECS      → (duration − 1 s)      : hold at 1
  last FADE_OUT_LAST seconds                 : ramp  1 → 0
"""

import cv2
import numpy as np
from typing import List, Tuple

# ── Clone layout — edit these two constants to control count and spread ───────
BASE_CLONE_OFFSETS     = [-220, 220, -430, 430]   # x-offsets for one level
CLONE_COUNT_MULTIPLIER = 3                         # levels = total clones ÷ base

# Opacity ceiling per level — add more values if CLONE_COUNT_MULTIPLIER grows.
# Each pair covers (narrow offset, wide offset) within a level.
_LEVEL_OPACITIES: List[Tuple[float, float]] = [
    (0.82, 0.65),   # level 1 — nearest, most visible
    (0.60, 0.48),   # level 2
    (0.42, 0.32),   # level 3 — farthest, most ghostlike
]


def _build_specs() -> List[Tuple[int, float, int]]:
    """Generate (x_offset, max_opacity, seq_index) for every clone."""
    specs: List[Tuple[int, float, int]] = []
    seq = 0
    for level in range(CLONE_COUNT_MULTIPLIER):
        scale = level + 1
        # Alternate opacity between narrow and wide slots within the level
        op_narrow, op_wide = (_LEVEL_OPACITIES[level]
                              if level < len(_LEVEL_OPACITIES) else (0.30, 0.22))
        half = len(BASE_CLONE_OFFSETS) // 2
        for i, base_offset in enumerate(BASE_CLONE_OFFSETS):
            # First half of BASE_CLONE_OFFSETS = near pair → op_narrow
            # Second half                      = far  pair → op_wide
            opacity = op_narrow if i < half else op_wide
            specs.append((base_offset * scale, opacity, seq))
            seq += 1
    return specs


# Pre-built at import time; regenerate by restarting if you change the constants.
_SPECS: List[Tuple[int, float, int]] = _build_specs()

# ── Timing constants ──────────────────────────────────────────────────────────
SEQ_DELAY         = 0.15   # seconds between successive clone spawns
CLONE_APPEAR_SECS = 0.25   # per-clone fade-in ramp duration after spawning
FADE_IN_SECS      = 0.50   # global fade-in at the very start
FADE_OUT_LAST     = 1.00   # global fade-out occupies the last N seconds
BLUR_RADIUS       = 5      # Gaussian blur half-width applied to clone layers


class CloneEffect:
    """
    Renders semi-transparent shadow clone copies of a person cutout.

    No internal state — the caller controls when the effect runs and for
    how long.  Typical pattern in the main loop::

        clone_fx = CloneEffect()

        # At gesture trigger:
        person_img, mask = body_seg.segment(detection_frame)
        clones_active = True
        clone_start   = time.monotonic()

        # Every frame while clones are active:
        elapsed = time.monotonic() - clone_start
        if elapsed < CLONE_DURATION_SECONDS:
            opacity = clone_fx.calculate_fade_opacity(elapsed, CLONE_DURATION_SECONDS)
            output_frame = clone_fx.draw_clones(
                output_frame, person_img, mask, elapsed, opacity
            )
        else:
            clones_active = False
    """

    def __init__(self, blur_radius: int = BLUR_RADIUS):
        self._blur = blur_radius

    # ── Public API ───────────────────────────────────────────────────────────

    @staticmethod
    def calculate_fade_opacity(elapsed: float, duration: float) -> float:
        """
        Return the global [0.0, 1.0] opacity multiplier for this frame.

        Parameters
        ----------
        elapsed  : seconds since the effect was triggered.
        duration : total intended lifetime of the effect in seconds.

        Fade schedule:
          0             → FADE_IN_SECS               : ramp 0 → 1  (0.5 s)
          FADE_IN_SECS  → duration − FADE_OUT_LAST   : hold at 1
          last FADE_OUT_LAST seconds                  : ramp 1 → 0  (1.0 s)
        """
        if elapsed <= 0.0 or elapsed >= duration:
            return 0.0
        if elapsed < FADE_IN_SECS:
            return elapsed / FADE_IN_SECS
        fade_out_start = duration - FADE_OUT_LAST
        if elapsed >= fade_out_start:
            return max(0.0, 1.0 - (elapsed - fade_out_start) / FADE_OUT_LAST)
        return 1.0

    def draw_clones(
        self,
        frame:         np.ndarray,
        person_cutout: np.ndarray,
        person_mask:   np.ndarray,
        elapsed:       float,
        opacity:       float = 1.0,
    ) -> np.ndarray:
        """
        Composite all currently-visible clone layers onto `frame`.

        Parameters
        ----------
        frame         : destination frame — a copy is returned; original unchanged.
        person_cutout : BGR person cutout captured at trigger time.
        person_mask   : uint8 H×W mask captured at trigger time (255=person).
        elapsed       : seconds since trigger — controls which clones have
                        appeared and their individual fade-in progress.
        opacity       : global multiplier from calculate_fade_opacity().
        """
        if opacity <= 0.01:
            return frame

        result = frame.copy()
        fh, fw = result.shape[:2]

        k = self._blur * 2 + 1
        blurred_person = cv2.GaussianBlur(person_cutout, (k, k), self._blur)
        blurred_mask   = cv2.GaussianBlur(person_mask,   (k, k), self._blur)

        # Render farthest clones first so nearer ones paint on top
        for x_off, max_clone_opacity, seq_idx in sorted(
            _SPECS, key=lambda s: abs(s[0]), reverse=True
        ):
            appear_at = seq_idx * SEQ_DELAY
            if elapsed < appear_at:
                continue  # this clone hasn't spawned yet in the sequence

            local_t     = elapsed - appear_at
            seq_alpha   = min(local_t / CLONE_APPEAR_SECS, 1.0)  # per-clone ramp
            layer_alpha = seq_alpha * opacity * max_clone_opacity
            if layer_alpha <= 0.01:
                continue

            result = self._blit(result, blurred_person, blurred_mask,
                                 x_off, layer_alpha, fh, fw)

        return result

    # ── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _blit(
        frame:  np.ndarray,
        person: np.ndarray,
        mask:   np.ndarray,
        x_off:  int,
        alpha:  float,
        fh:     int,
        fw:     int,
    ) -> np.ndarray:
        """Alpha-blend one shifted clone layer onto `frame` (modifies in-place)."""
        ph, pw = person.shape[:2]

        src_x1 = 0
        dst_x1 = x_off
        if dst_x1 < 0:
            src_x1 = -dst_x1
            dst_x1 = 0
        src_w = min(pw - src_x1, fw - dst_x1)
        if src_w <= 0:
            return frame
        src_x2 = src_x1 + src_w
        dst_x2 = dst_x1 + src_w

        src_h  = min(ph, fh)
        src_y2 = src_h
        dst_y2 = src_h

        src_bgr  = person[0:src_y2, src_x1:src_x2].astype(np.float32)
        src_mask = mask[0:src_y2,   src_x1:src_x2].astype(np.float32) / 255.0
        dst      = frame[0:dst_y2, dst_x1:dst_x2].astype(np.float32)

        a       = src_mask[:, :, np.newaxis] * alpha
        blended = src_bgr * a + dst * (1.0 - a)

        frame[0:dst_y2, dst_x1:dst_x2] = np.clip(blended, 0, 255).astype(np.uint8)
        return frame
