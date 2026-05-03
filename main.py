"""
main.py — Naruto Shadow Clone AR Filter  (Phase 1 + Phase 2)

Phase 1  →  Naruto face PNG overlay on the detected face.
Phase 2  →  Shadow clone hand sign → body segmentation → ghost clone effect.

Full pipeline per frame
───────────────────────
Webcam frame
  → flip (mirror)
  → [Phase 2] clone layers rendered FIRST (so they appear behind the person)
  → [Phase 1] Haar face detection + Naruto face overlay
  → [Phase 2] hand landmark detection
  → [Phase 2] gesture state machine  (hold sign ≥ 0.5 s → trigger)
  → HUD (FPS / gesture charge bar / "SHADOW CLONE JUTSU!" banner)
  → display
"""

import sys
import os
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from camera           import CameraCapture
from face_detector    import FaceDetector
from face_overlay     import FaceOverlay
from stable_face_tracker import StableFaceTracker
from hand_detector    import HandDetector
from gesture_detector import GestureDetector
from body_segmenter   import BodySegmenter
from clone_effect     import CloneEffect
from utils            import load_image_with_alpha, validate_assets, FPSCounter, draw_hud

# ---------------------------------------------------------------------------
# Configuration — edit these constants without touching logic below
# ---------------------------------------------------------------------------

ASSET_PATH    = os.path.join(os.path.dirname(__file__), "assets", "naruto_face.png")
CAMERA_INDEX  = 0

# Phase 1 — face overlay
OVERLAY_SCALE = 1.8    # overlay width = face_width × OVERLAY_SCALE
HEAD_COVERAGE = 0.65   # fraction of overlay height above face centre
FEATHER_PX    = 8      # alpha-edge blur radius (pixels)

# Phase 2 — clone lifetime
CLONE_DURATION_SECONDS = 10.0   # how long clones stay on screen after trigger
COOLDOWN_SECONDS       = 2.0    # minimum wait after clones expire before re-trigger

WINDOW_TITLE = "Naruto AR Filter  |  Phase 1 + 2  |  [q] quit"


# ---------------------------------------------------------------------------
# HUD helpers
# ---------------------------------------------------------------------------

def _draw_charge_bar(frame: np.ndarray, progress: float) -> None:
    """Chakra-blue charge bar at the bottom — shows gesture hold progress."""
    if progress <= 0.0:
        return
    h, w = frame.shape[:2]
    bar_h = 12
    bar_y = h - bar_h - 8
    filled = int(w * progress)

    cv2.rectangle(frame, (0, bar_y), (w, bar_y + bar_h), (20, 20, 20), -1)
    cv2.rectangle(frame, (0, bar_y), (filled, bar_y + bar_h), (210, 150, 10), -1)

    label = "HOLD SIGN…"
    cv2.putText(frame, label, (10, bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 150, 10), 1, cv2.LINE_AA)


def _draw_jutsu_banner(frame: np.ndarray) -> None:
    """Flashing 'SHADOW CLONE JUTSU!' banner while the effect is active."""
    _, w = frame.shape[:2]
    text = "SHADOW CLONE JUTSU!"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale, thick = 1.0, 2
    (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
    x = (w - tw) // 2
    y = 62
    # Dark outline for legibility on any background
    cv2.putText(frame, text, (x + 2, y + 2), font, scale, (0,   0,   0),   thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x,     y),     font, scale, (0, 210, 255),   thick,     cv2.LINE_AA)


def _draw_hand_debug(frame: np.ndarray, hands: list) -> None:
    """
    Draw wrist dots + index-finger direction arrows for each detected hand.

    Arrow colours:
      Cyan   — index finger direction (always drawn)
      Green  — finger is VERTICAL   (correct for one hand in the jutsu sign)
      Orange — finger is HORIZONTAL (correct for the other hand)

    When you see one green arrow pointing up and one orange arrow pointing
    sideways, the sign geometry is correct — just hold it for 0.5 s.
    """
    h, w = frame.shape[:2]

    for hand in hands:
        lm = hand["landmarks"]

        # Wrist dot
        wx, wy = int(lm[0][0] * w), int(lm[0][1] * h)
        cv2.circle(frame, (wx, wy), 6, (0, 255, 120), -1, cv2.LINE_AA)
        cv2.circle(frame, (wx, wy), 6, (0,   0,   0),  1, cv2.LINE_AA)

        # Index finger arrow: MCP (knuckle) → tip, doubled in length for visibility
        mcp_x, mcp_y = int(lm[5][0] * w), int(lm[5][1] * h)
        tip_x, tip_y = int(lm[8][0] * w), int(lm[8][1] * h)
        dx, dy = tip_x - mcp_x, tip_y - mcp_y

        end_x = mcp_x + dx * 2
        end_y = mcp_y + dy * 2

        import math
        mag = math.hypot(dx, dy)
        if mag > 5:
            is_vertical = abs(dy) > abs(dx)
            colour = (0, 230, 0) if is_vertical else (0, 140, 255)  # green / orange
            cv2.arrowedLine(frame, (mcp_x, mcp_y), (end_x, end_y),
                            colour, 2, tipLength=0.35, line_type=cv2.LINE_AA)
            label = "V" if is_vertical else "H"
            cv2.putText(frame, label, (end_x + 4, end_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    # ── Asset check ──────────────────────────────────────────────────────────
    if not validate_assets(ASSET_PATH):
        print("[HINT] Run:  python create_placeholder_asset.py")
        sys.exit(1)

    overlay_image = load_image_with_alpha(ASSET_PATH)
    if overlay_image is None:
        print("[ERROR] Could not load overlay image.")
        sys.exit(1)

    print("[INFO] Naruto AR Filter (Phase 1 + 2) starting — press 'q' to quit.")
    print("[INFO] Perform the Naruto shadow clone hand sign to trigger clones:")
    print("[INFO]   → Bring both hands together / overlapping in front of you")
    print("[INFO]   → One hand: index finger pointing UPWARD   (vertical)")
    print("[INFO]   → Other hand: index finger pointing SIDEWAYS (horizontal)")
    print("[INFO]   → They form a CROSS shape — like a real jutsu seal")
    print("[INFO]   → Hold the cross for 0.5 s until the gold bar fills")

    fps_counter = FPSCounter(window=30)

    # Clone lifecycle state — all managed here, CloneEffect is a pure renderer
    _clones_active  = False
    _clone_start    = 0.0    # time.monotonic() when clones were triggered
    _next_trigger   = 0.0    # earliest time a new trigger is allowed
    _cached_person  = None   # BGR person cutout captured at trigger time
    _cached_mask    = None   # uint8 mask captured at trigger time

    try:
        with CameraCapture(CAMERA_INDEX)  as cam,  \
             FaceDetector()               as face_det, \
             HandDetector()               as hand_det, \
             BodySegmenter()              as body_seg:

            face_overlay  = FaceOverlay(overlay_image, OVERLAY_SCALE,
                                        HEAD_COVERAGE, FEATHER_PX)
            face_tracker  = StableFaceTracker(
                                alpha=0.22, dead_zone_px=5, max_jump_px=60,
                                max_missing_frames=12, lock_center_dist_norm=0.25)
            gesture_det   = GestureDetector()
            clone_fx      = CloneEffect()

            while True:
                ret, raw_frame = cam.read_frame()
                if not ret or raw_frame is None:
                    continue
                raw_frame = cv2.flip(raw_frame, 1)   # mirror → selfie orientation

                # detection_frame — clean copy; every ML detector reads only this.
                # output_frame    — clone layers drawn first (behind), then the
                #                   live Naruto overlay on top; shown to the user.
                detection_frame = raw_frame.copy()
                output_frame    = raw_frame.copy()

                # ── Phase 1: Face detection (on clean frame) ──────────────────
                face_boxes   = face_det.detect_faces(detection_frame)
                smoothed_box = face_tracker.update(face_boxes, detection_frame.shape)

                # ── Phase 2A: Clone layers behind the live person ─────────────
                now = time.monotonic()
                if _clones_active and _cached_person is not None:
                    elapsed = now - _clone_start
                    if elapsed < CLONE_DURATION_SECONDS:
                        opacity = clone_fx.calculate_fade_opacity(
                            elapsed, CLONE_DURATION_SECONDS
                        )
                        output_frame = clone_fx.draw_clones(
                            output_frame,
                            _cached_person,
                            _cached_mask,
                            elapsed,
                            opacity,
                        )
                    else:
                        # Lifetime expired — clear state and start cooldown
                        _clones_active = False
                        _cached_person = None
                        _cached_mask   = None
                        _next_trigger  = now + COOLDOWN_SECONDS
                        print("[INFO] Shadow clones dispersed.")

                # Naruto face on top (solid, always in front of ghost clones)
                if smoothed_box is not None:
                    output_frame = face_overlay.apply(output_frame, [smoothed_box])

                # ── Phase 2B: Hand landmark detection on clean frame ──────────
                hands = hand_det.detect(detection_frame)

                # ── Phase 2C: Gesture — trigger only when clones are not active
                triggered = gesture_det.update(hands)

                if triggered and not _clones_active and now >= _next_trigger:
                    # Segment from the clean frame — full-frame-sized output so
                    # the face coordinates match raw_frame exactly.
                    _cached_person, _cached_mask = body_seg.segment(detection_frame)

                    # Bake the Naruto overlay into the clone source right now so
                    # every clone carries the face for the full 10 s without
                    # re-running face detection on clone frames.
                    if smoothed_box is not None:
                        _cached_person = face_overlay.apply(_cached_person, [smoothed_box])

                    _clones_active = True
                    _clone_start   = time.monotonic()
                    print("[INFO] ✦ Shadow Clone Jutsu activated! (10 s)")

                # ── HUD — all drawn on output_frame only ──────────────────────
                fps = fps_counter.tick()
                draw_hud(output_frame, fps, len(face_boxes))
                _draw_hand_debug(output_frame, hands)

                # Charge bar only when clones are not already on screen
                progress = gesture_det.holding_progress
                if progress > 0.0 and not _clones_active:
                    _draw_charge_bar(output_frame, progress)

                if _clones_active:
                    _draw_jutsu_banner(output_frame)

                cv2.imshow(WINDOW_TITLE, output_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Quit signal received.")
                    break

    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cv2.destroyAllWindows()
        print("[INFO] Naruto AR Filter closed.")


if __name__ == "__main__":
    run()
