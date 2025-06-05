#!/bin/bash
echo "[install.sh] Installing Python dependencies..."
python3 -m venv agi_env
source agi_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt || true
echo "[install.sh] Done."
