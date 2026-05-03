"""
gesture_detector.py — Detects the Naruto shadow clone hand sign.

Trigger condition (simple and fast):
  Any fingertip of one hand must be physically inside (or touching)
  the bounding box of the other hand — i.e. the hands are crossed/overlapping.
  Works regardless of which hand is on top or which orientation they are in.

Hold + cooldown:
  The overlap must be held for HOLD_DURATION seconds to fire.
  After firing, a COOLDOWN_DURATION cooldown prevents immediate re-trigger.
"""

import time
from typing import Dict, List, Tuple

HOLD_DURATION     = 0.3   # seconds of continuous overlap needed to trigger
COOLDOWN_DURATION = 3.0   # seconds before the gesture can fire again

# Expand each hand's bounding box by this amount in normalised coords
# so a finger touching the edge still counts as "inside".
BBOX_EXPAND = 0.025

# Fingertip landmark indices (thumb, index, middle, ring, pinky)
_TIPS = [4, 8, 12, 16, 20]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _bbox(hand: Dict) -> Tuple[float, float, float, float]:
    """Bounding box of all 21 landmarks: (x1, y1, x2, y2) in normalised coords."""
    lm  = hand["landmarks"]
    xs  = [p[0] for p in lm]
    ys  = [p[1] for p in lm]
    return min(xs), min(ys), max(xs), max(ys)


def _finger_over_hand(tips_hand: Dict, bbox_hand: Dict) -> bool:
    """
    Return True when any fingertip of `tips_hand` falls inside
    the (expanded) bounding box of `bbox_hand`.
    """
    x1, y1, x2, y2 = _bbox(bbox_hand)
    x1 -= BBOX_EXPAND;  y1 -= BBOX_EXPAND
    x2 += BBOX_EXPAND;  y2 += BBOX_EXPAND

    lm = tips_hand["landmarks"]
    for tip_idx in _TIPS:
        px, py = lm[tip_idx][0], lm[tip_idx][1]
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True
    return False


def _sign_matches(hands: List[Dict]) -> bool:
    """
    True when either hand has at least one fingertip inside the other hand.
    Symmetric — works regardless of which hand is on top.
    """
    if len(hands) < 2:
        return False
    a, b = hands[0], hands[1]
    return _finger_over_hand(a, b) or _finger_over_hand(b, a)


# ---------------------------------------------------------------------------
# Stateful detector
# ---------------------------------------------------------------------------

class GestureDetector:
    """
    Wraps _sign_matches() with a continuous hold gate and cooldown.

    Call update() once per frame.
    Returns True exactly once per valid trigger, then enters cooldown.
    Use holding_progress [0.0–1.0] to drive the charge-bar UI.
    """

    def __init__(
        self,
        hold_duration: float = HOLD_DURATION,
        cooldown: float = COOLDOWN_DURATION,
    ):
        self._hold         = hold_duration
        self._cooldown     = cooldown
        self._sign_start   = 0.0
        self._last_trigger = 0.0

    def update(self, hands: List[Dict]) -> bool:
        now = time.monotonic()

        if now - self._last_trigger < self._cooldown:
            return False

        if _sign_matches(hands):
            if self._sign_start == 0.0:
                self._sign_start = now
            elif now - self._sign_start >= self._hold:
                self._last_trigger = now
                self._sign_start   = 0.0
                return True
        else:
            self._sign_start = 0.0

        return False

    @property
    def holding_progress(self) -> float:
        if self._sign_start == 0.0:
            return 0.0
        return min((time.monotonic() - self._sign_start) / self._hold, 1.0)

    @property
    def in_cooldown(self) -> bool:
        return (time.monotonic() - self._last_trigger) < self._cooldown
