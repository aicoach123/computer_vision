"""
create_placeholder_asset.py
----------------------------
Generates a simple Naruto-style test overlay (BGRA PNG) so you can verify
the pipeline works before dropping in the real artwork.

Run once:
    python create_placeholder_asset.py
"""

import math
import os
import cv2
import numpy as np


OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "assets", "naruto_face.png")
SIZE = 300  # canvas size in pixels


def _filled_ellipse(img, cx, cy, rx, ry, color_bgra):
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, color_bgra, -1, cv2.LINE_AA)


def create_naruto_face(size: int = SIZE) -> np.ndarray:
    """
    Draw a minimal Naruto-style face:
      - Skin-tone oval
      - Blue eyes with pupils
      - Three whisker marks on each cheek
      - Blue headband with a grey metal plate
    Returns a BGRA ndarray.
    """
    img = np.zeros((size, size, 4), dtype=np.uint8)

    cx, cy = size // 2, size // 2
    face_rx = int(size * 0.38)
    face_ry = int(size * 0.44)

    # ---- Face oval ----
    skin = (120, 175, 220, 255)          # BGR skin tone + full alpha
    _filled_ellipse(img, cx, cy, face_rx, face_ry, skin)

    # ---- Eyes ----
    eye_y = cy - int(face_ry * 0.12)
    eye_dx = int(face_rx * 0.40)
    eye_rx, eye_ry = int(size * 0.065), int(size * 0.05)

    for ex in (cx - eye_dx, cx + eye_dx):
        _filled_ellipse(img, ex, eye_y, eye_rx, eye_ry, (40, 30, 100, 255))   # iris
        _filled_ellipse(img, ex, eye_y, eye_rx // 2, eye_ry // 2, (0, 0, 0, 255))  # pupil
        # highlight
        hx, hy = ex - eye_rx // 4, eye_y - eye_ry // 4
        cv2.circle(img, (hx, hy), max(1, eye_rx // 5), (255, 255, 255, 255), -1, cv2.LINE_AA)

    # ---- Whisker marks (3 per cheek) ----
    whisker_color = (50, 50, 160, 230)
    thickness = max(2, size // 90)
    cheek_cx_off = int(face_rx * 0.45)
    cheek_y = cy + int(face_ry * 0.10)
    whisker_len = int(face_rx * 0.42)

    for side in (-1, 1):
        base_x = cx + side * cheek_cx_off
        for angle_deg in (-20, 0, 20):
            rad = math.radians(angle_deg)
            ex = base_x + side * int(math.cos(rad) * whisker_len)
            ey = cheek_y + int(math.sin(rad) * whisker_len)
            cv2.line(img, (base_x, cheek_y), (ex, ey), whisker_color, thickness, cv2.LINE_AA)

    # ---- Mouth ----
    mouth_y = cy + int(face_ry * 0.45)
    cv2.ellipse(
        img, (cx, mouth_y), (int(face_rx * 0.25), int(face_ry * 0.10)),
        0, 0, 180, (60, 60, 130, 210), 2, cv2.LINE_AA,
    )

    # ---- Headband ----
    hb_top = cy - int(face_ry * 1.02)
    hb_bot = cy - int(face_ry * 0.75)
    hb_height = hb_bot - hb_top

    # Blue band spanning full width
    for col in range(size):
        alpha_val = 230
        img[hb_top:hb_bot, col] = [150, 70, 0, alpha_val]   # blue headband (BGR)

    # Grey metal plate in the centre
    plate_w = int(size * 0.36)
    plate_x1, plate_x2 = cx - plate_w // 2, cx + plate_w // 2
    cv2.rectangle(img, (plate_x1, hb_top + 2), (plate_x2, hb_bot - 2),
                  (180, 180, 180, 255), -1)

    # Konoha leaf symbol (simplified: circle + vertical line)
    sym_cx, sym_cy = cx, hb_top + hb_height // 2
    sym_r = hb_height // 3
    cv2.circle(img, (sym_cx, sym_cy), sym_r, (60, 60, 60, 255), 1, cv2.LINE_AA)
    cv2.line(img, (sym_cx, sym_cy - sym_r), (sym_cx, sym_cy + sym_r),
             (60, 60, 60, 255), 1, cv2.LINE_AA)

    # ---- Nose bridge (subtle) ----
    nose_top = cy - int(face_ry * 0.15)
    nose_bot = cy + int(face_ry * 0.20)
    cv2.line(img, (cx, nose_top), (cx, nose_bot), (90, 130, 170, 80), 1, cv2.LINE_AA)

    return img


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    face_img = create_naruto_face(SIZE)
    success = cv2.imwrite(OUTPUT_PATH, face_img)
    if success:
        print(f"[OK] Placeholder asset saved to: {OUTPUT_PATH}")
        print("     Replace it with a real Naruto face PNG for best results.")
    else:
        print(f"[ERROR] Failed to write image to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
