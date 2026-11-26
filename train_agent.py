import os
import cv2
import numpy as np
import pandas as pd

CSV_PATH = "road_geometry.csv"
FRAMES_DIR = "dashcam_frames"

# Local mountain reference origin (simulated GPS anchor)
ORIGIN_LAT = 34.0000
ORIGIN_LON = 73.0000
BASE_ALT_M = 1200.0
METERS_PER_DEG_LAT = 111320.0


def load_geometry_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Straight road par radius_m NaN hota hai; usay safely track karne ke liye flag banayein
    if "radius_m" in df.columns:
        df["is_straight_radius"] = df["radius_m"].isna()
        # Agar numeric radius chahiye toh NaN ko np.inf (infinite radius) rakhein
        df["radius_clean_m"] = df["radius_m"].fillna(np.inf)

    return df


def attach_gps_coordinates(
    df: pd.DataFrame,
    origin_lat: float = ORIGIN_LAT,
    origin_lon: float = ORIGIN_LON,
    base_alt_m: float = BASE_ALT_M,
) -> pd.DataFrame:
    df = df.copy()

    # Relative displacement from first trajectory sample so coordinates stay clean and local
    dx = df["cam_x"].astype(float) - float(df["cam_x"].iloc[0])
    dy = df["cam_y"].astype(float) - float(df["cam_y"].iloc[0])
    dz = df["cam_z"].astype(float) - float(df["cam_z"].iloc[0])

    meters_per_deg_lon = METERS_PER_DEG_LAT * np.cos(np.radians(origin_lat))

    df["gps_lat"] = origin_lat + (dy / METERS_PER_DEG_LAT)
    df["gps_lon"] = origin_lon + (dx / meters_per_deg_lon)
    df["gps_alt"] = base_alt_m + dz

    return df


def resolve_frame_path(row: pd.Series, frames_dir: str, image_col: str = "image") -> str | None:
    raw_name = str(row[image_col]).strip()
    candidates = [raw_name]

    # Agar CSV mein 'frame_0014.jpg' hai lekin folder mein '0014.jpg' hai
    if raw_name.startswith("frame_"):
        candidates.append(raw_name.replace("frame_", "", 1))

    # Agar 'frame' column integer ho toh '0014.jpg' bhi try karein
    if "frame" in row and pd.notna(row["frame"]):
        try:
            f_num = int(row["frame"])
            candidates.append(f"{f_num:04d}.jpg")
            candidates.append(f"frame_{f_num:04d}.jpg")
        except (ValueError, TypeError):
            pass

    for cand in candidates:
        if cand == ".DS_Store":
            continue
        full_path = os.path.join(frames_dir, cand)
        if os.path.exists(full_path):
            return full_path

    return None


def verify_frames(
    df: pd.DataFrame, frames_dir: str, image_col: str = "image"
) -> dict:
    missing_files = []
    corrupt_files = []
    shapes = set()
    valid_count = 0

    for idx, row in df.iterrows():
        raw_name = str(row[image_col]).strip()
        frame_path = resolve_frame_path(row, frames_dir, image_col=image_col)

        if frame_path is None:
            missing_files.append((idx, raw_name))
            continue

        img = cv2.imread(frame_path)
        if img is None:
            corrupt_files.append((idx, raw_name))
            continue

        valid_count += 1
        shapes.add(img.shape)

    return {
        "valid_count": valid_count,
        "missing_files": missing_files,
        "corrupt_files": corrupt_files,
        "unique_shapes": list(shapes),
    }


def extract_frame_vision_features(img: np.ndarray) -> dict:
    # Neeche ka road region (ROI: 55% se 95% height) lein taake sky/distant hills ka noise kam ho
    h, w, _ = img.shape
    roi = img[int(h * 0.55) : int(h * 0.95), :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    edge_density = float(np.mean(edges > 0))
    mean_brightness = float(np.mean(gray) / 255.0)

    mid_x = w // 2
    left_edges = float(np.mean(edges[:, :mid_x] > 0))
    right_edges = float(np.mean(edges[:, mid_x:] > 0))
    edge_lr_balance = float(right_edges - left_edges)

    return {
        "edge_density": round(edge_density, 5),
        "mean_brightness": round(mean_brightness, 4),
        "edge_lr_balance": round(edge_lr_balance, 5),
    }


if __name__ == "__main__":
    df = load_geometry_data(CSV_PATH)
    df = attach_gps_coordinates(df)
    print(f"CSV loaded: {len(df)} rows")
    if "is_straight_radius" in df.columns:
        print(f"Straight/NaN radius rows: {df['is_straight_radius'].sum()}")

    print(
        "Sample GPS (row 0):",
        df[["distance_m", "cam_x", "cam_y", "cam_z", "gps_lat", "gps_lon", "gps_alt"]].iloc[0].to_dict(),
    )

    frame_report = verify_frames(df, FRAMES_DIR)
    print(f"Valid frames: {frame_report['valid_count']}/{len(df)}")
    print(f"Missing frames: {len(frame_report['missing_files'])}")
    print(f"Corrupt frames: {len(frame_report['corrupt_files'])}")
    print(f"Image shapes: {frame_report['unique_shapes']}")

    first_path = resolve_frame_path(df.iloc[0], FRAMES_DIR)
    if first_path is not None:
        sample_img = cv2.imread(first_path)
        if sample_img is not None:
            print("Sample vision features (row 0):", extract_frame_vision_features(sample_img))