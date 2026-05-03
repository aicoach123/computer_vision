#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "[setup] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Install / upgrade dependencies
echo "[setup] Installing dependencies..."
"$PIP" install --quiet --upgrade pip
"$PIP" install --quiet -r requirements.txt

# Generate placeholder asset if no PNG exists yet
if [ ! -f "$SCRIPT_DIR/assets/naruto_face.png" ]; then
    echo "[setup] Generating placeholder asset..."
    "$PYTHON" create_placeholder_asset.py
fi

echo "[start] Launching Naruto AR Filter..."
"$PYTHON" main.py
