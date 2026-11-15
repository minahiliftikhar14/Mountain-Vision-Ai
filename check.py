import os
import pandas as pd
import cv2

# File paths
csv_path = "road_geometry.csv"
frames_folder = "dashcam_frames"

print("--- CHECK.PY STARTED ---")

# 1. Check if CSV exists
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"SUCCESS: CSV loaded. Total rows: {len(df)}")
else:
    print("ERROR: road_geometry.csv not found!")

# 2. Check if Frames folder exists
if os.path.exists(frames_folder):
    print(f"SUCCESS: Frames folder found at '{frames_folder}'")
else:
    print(f"ERROR: Folder '{frames_folder}' not found!")

print("--- CHECK.PY COMPLETED SUCCESSFULLY ---")