# Mountain Vision AI

An autonomous driving / road-vision project built in Python and Blender, structured to separate dataset verification from model training.

## Project Overview

This project generates and verifies synthetic mountain road data—pairing dashcam frames with road geometry data—and prepares a modular pipeline for training vision models.

## Project Structure

- `check.py` — Verifies that `road_geometry.csv` and the `dashcam_frames/` folder exist, load properly, and stay synchronized.
- `train.py` — Dedicated script for model training (kept decoupled from data verification).
- `road_geometry.csv` — Road geometry dataset (287 rows).
- `dashcam_frames/` — Generated dashcam image frames from the Blender camera/road pipeline.
- `venv/` — Local Python virtual environment (recommended not to commit to GitHub).

## Environment Setup (macOS)

1. Navigate to the project directory and activate the virtual environment:
   ```bash
   source venv/bin/activate
   Ensure pip is available and install required dependencies (using binary wheels for faster, reliable setup on macOS):

'''bash
python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip
python3 -m pip install --only-binary=:all: numpy pandas matplotlib opencv-python
'''
2. ##Usage
Run the dataset verification script before starting model training:

Bash
python3 check.py
Once verification passes, run your training script:

Bash
python3 train.py