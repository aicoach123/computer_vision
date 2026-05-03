# Naruto Shadow Clone AR Filter

A real-time augmented reality filter that puts a Naruto face overlay on your face and lets you summon shadow clone duplicates of yourself using the Naruto hand sign — all running live from your webcam.

---

## How to run

> **Requires Python 3.10 or 3.12** and a webcam. That's it.

```bash
# 1. Clone the repo
git clone https://github.com/aicoach123/computer_vision.git
cd computer_vision

# 2. Run — everything else is automatic
bash start.sh
```
DONE


## Other Claude Explanation and Stuff

`start.sh` will:
- Create a Python virtual environment (`.venv/`)
- Install all dependencies (`opencv-python`, `mediapipe`, `numpy`, `Pillow`)
- Download the ML models on first run (~8 MB, one time only)
- Launch the filter

Press **`q`** in the webcam window to quit.

---

## What it does

| Feature | Description |
|---|---|
| **Naruto face overlay** | Detects your face and draws the Naruto PNG on top, locked and smoothed so it doesn't jitter |
| **Shadow Clone Jutsu** | Perform the real Naruto hand sign and hold it for 0.3 s — 12 ghost copies of you spread across the screen for 10 seconds |
| **Clones have the overlay too** | Each clone is captured with the Naruto face baked in at trigger time |
| **Auto fade** | Clones fade in (0.5 s), hold, then fade out (last 1 s) |

---

## Demo — how to trigger the shadow clones

Make this hand sign in front of your camera:

```
One hand pointing UP  (index finger vertical)
Other hand pointing SIDEWAYS  (index finger horizontal)
Cross them so the fingers overlap — like a plus sign (+)
Hold for ~0.3 seconds until the gold bar at the bottom fills up
```

You will see a gold charge bar fill at the bottom of the screen.
When it fills, 12 clones appear and stay for 10 seconds.

---

## Requirements

- **Python 3.10 or 3.12** (recommended — tested on macOS)
- A working **webcam**
- **macOS, Linux, or Windows** (macOS tested)
- Internet connection on first run (downloads two small ML models automatically)

> Python 3.13 is not yet supported by MediaPipe. Use 3.10 or 3.12.

---

## Quickstart (easiest way)

### Step 1 — Clone the repository

```bash
git clone https://github.com/aicoach123/computer_vision.git
cd computer_vision
```

### Step 2 — Run the start script

```bash
bash start.sh
```

That's it. The script will:
1. Create a Python virtual environment (`.venv/`)
2. Install all dependencies automatically
3. Download the ML models on first run (~8 MB total)
4. Launch the filter

Press **`q`** in the webcam window to quit.

---

## Manual setup (if you prefer to control each step)

If you don't want to use `start.sh`, follow these steps:

### Step 1 — Make sure Python is installed

Check your version:

```bash
python3 --version
```

You need **3.10.x** or **3.12.x**. Download Python from [python.org](https://www.python.org/downloads/) if needed.

### Step 2 — Create a virtual environment

A virtual environment keeps project dependencies isolated from the rest of your system.

```bash
python3 -m venv .venv
```

### Step 3 — Activate the virtual environment

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows (Command Prompt):**
```bat
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

Your terminal prompt should now show `(.venv)` at the start.

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `opencv-python` — webcam capture and image processing
- `mediapipe` — hand landmark detection and body segmentation
- `numpy` — numerical array operations
- `Pillow` — image loading

### Step 5 — Run the filter

```bash
python main.py
```

> On first run the app will automatically download two ML models into the `models/` folder (~8 MB). You only need an internet connection this one time.

---

## Project structure

```
naruto-shadow-clone-ar/
│
├── main.py                      # Entry point — main loop and pipeline
├── start.sh                     # One-command launcher (auto-installs everything)
├── requirements.txt             # Python dependencies
├── pyrightconfig.json           # VS Code / Pylance type-checking config
│
├── assets/
│   └── naruto_face.png          # The Naruto face overlay image (BGRA PNG)
│
├── models/                      # ML models — downloaded automatically at runtime
│   ├── hand_landmarker.task     # MediaPipe hand landmark model (~7.5 MB)
│   └── selfie_segmenter.tflite  # MediaPipe body segmentation model (~244 KB)
│
└── src/                         # All modules
    ├── camera.py                # Webcam capture (context manager)
    ├── face_detector.py         # Haar cascade face detection
    ├── face_overlay.py          # Resizes + alpha-blends the Naruto PNG onto faces
    ├── stable_face_tracker.py   # Lock-on tracker with EMA smoothing (no jitter)
    ├── face_smoother.py         # Legacy EMA smoother (kept for reference)
    ├── hand_detector.py         # MediaPipe hand landmark detection
    ├── gesture_detector.py      # Shadow clone hand sign detector (hold + cooldown)
    ├── body_segmenter.py        # Person cutout via MediaPipe / OpenCV MOG2 fallback
    ├── clone_effect.py          # Stateless renderer — 12 sequential ghost clones
    └── utils.py                 # Asset loading, model download, FPS counter, HUD
```

---

## How the pipeline works (for the curious)

Every frame goes through this pipeline:

```
Webcam frame
  │
  ├─ flip horizontally (mirror / selfie view)
  │
  ├─── detection_frame = clean copy (never drawn on)
  │         │
  │         ├── Haar cascade face detector  →  StableFaceTracker (EMA + lock-on)
  │         ├── MediaPipe hand detector     →  GestureDetector (hold sign 0.3 s)
  │         └── Body segmenter             →  person cutout (on gesture trigger only)
  │
  └─── output_frame = drawing copy
            │
            ├── [if clones active] draw 12 ghost clones behind person
            ├── draw live Naruto face overlay on real face (on top of clones)
            ├── draw HUD: FPS, hand debug arrows, charge bar, banner
            └── display
```

Key design rule: **the face detector and hand detector always run on `detection_frame` (the clean raw frame), never on the output frame that has overlays drawn on it.** This prevents the Naruto overlay from confusing the face detector.

---

## Tuning / customisation

All constants are at the top of their files and commented:

| File | What to tune |
|---|---|
| `main.py` | `OVERLAY_SCALE`, `HEAD_COVERAGE`, `CLONE_DURATION_SECONDS`, `COOLDOWN_SECONDS` |
| `src/clone_effect.py` | `BASE_CLONE_OFFSETS`, `CLONE_COUNT_MULTIPLIER`, `SEQ_DELAY`, fade timings |
| `src/stable_face_tracker.py` | `SMOOTHING_ALPHA`, `DEAD_ZONE_PX`, `MAX_JUMP_PX`, `MAX_MISSING_FRAMES` |
| `src/gesture_detector.py` | `HOLD_DURATION` (how long to hold sign), `COOLDOWN_DURATION` |

---

## Troubleshooting

### Webcam not opening
- Make sure no other app (Zoom, FaceTime) is using the camera.
- Try changing `CAMERA_INDEX = 0` to `1` or `2` in `main.py`.

### Face overlay not appearing
- Make sure your face is visible and well-lit facing the camera.
- The Haar cascade works best with frontal faces in reasonable lighting.

### Hand sign not triggering
- Watch the **green/orange arrows** drawn on your hands — green = finger pointing up (vertical), orange = finger pointing sideways (horizontal).
- You need one of each, crossed so they overlap.
- Watch the **gold bar** at the bottom — hold until it fills.

### Clones look like a plain silhouette (no Naruto face on them)
- The Naruto face is baked into the clone at trigger time using whatever face box was detected at that moment. Make sure your face is detected (overlay visible) before performing the hand sign.

### `mediapipe` import errors
- Make sure you are using Python **3.10 or 3.12**, not 3.13.
- Re-run `pip install -r requirements.txt` with the virtual environment active.

### SSL certificate error on model download (macOS)
- This is a known macOS Python issue. The app works around it automatically using a custom SSL context in `src/utils.py`.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | ≥ 4.8.0 | Webcam capture, image ops, Haar cascade |
| `mediapipe` | ≥ 0.10.0 | Hand landmarks, body segmentation |
| `numpy` | ≥ 1.24.0 | Array math for blending |
| `Pillow` | ≥ 10.0.0 | PNG loading with alpha channel |

---

## License

MIT — do whatever you want with it.
