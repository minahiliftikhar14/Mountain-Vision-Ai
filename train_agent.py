import os
import cv2
import numpy as np
import pandas as pd

CSV_PATH = "road_geometry.csv"
FRAMES_DIR = "dashcam_frames"

# Optional visual playback toggle (set True when you want a popup window to play frames)
PREVIEW_PLAYBACK = False
PREVIEW_DELAY_MS = 200

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


def summarize_gps_coordinates(df: pd.DataFrame) -> dict:
    """Summarize mapped GPS coordinates across all rows in the dataset."""
    return {
        "rows_mapped": int(df[["gps_lat", "gps_lon", "gps_alt"]].dropna().shape[0]),
        "lat_range": (
            round(float(df["gps_lat"].min()), 6),
            round(float(df["gps_lat"].max()), 6),
        ),
        "lon_range": (
            round(float(df["gps_lon"].min()), 6),
            round(float(df["gps_lon"].max()), 6),
        ),
        "alt_range_m": (
            round(float(df["gps_alt"].min()), 3),
            round(float(df["gps_alt"].max()), 3),
        ),
        "total_distance_m": (
            round(float(df["distance_m"].max()), 2)
            if "distance_m" in df.columns
            else None
        ),
    }


def resolve_frame_path(
    row: pd.Series, frames_dir: str, image_col: str = "image"
) -> str | None:
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


def compute_adaptive_roi_ratios(
    img: np.ndarray,
    slope_percent: float | None = None,
    min_y_clamp: tuple[float, float] = (0.25, 0.60),
    max_y_clamp: tuple[float, float] = (0.85, 1.00),
) -> tuple[float, float]:
    """Sub-step 2.1: Automatically adjust y_min_ratio and y_max_ratio per frame."""
    h, _ = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Top background reference (upper 15% of frame)
    top_band = gray[: max(1, int(h * 0.15)), :]
    bg_mean = float(np.mean(top_band))

    # Row-wise deviation from background + horizontal gradient activity (road edges/markings)
    row_diff = np.mean(np.abs(gray - bg_mean), axis=1)
    grad_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    row_grad = np.mean(grad_x, axis=1)

    row_activity = row_diff + 0.5 * row_grad
    threshold = max(4.0, float(np.percentile(row_activity, 65)) * 0.6)
    active_rows = np.where(row_activity > threshold)[0]

    if len(active_rows) > 0:
        y_min_ratio = (float(active_rows[0]) / h) - 0.03
        y_max_ratio = (float(active_rows[-1] + 1) / h) + 0.02
    else:
        y_min_ratio = 0.35
        y_max_ratio = 1.00

    # Optional subtle adjustment if slope_percent is available in CSV
    if slope_percent is not None and not np.isnan(slope_percent):
        slope_shift = float(np.clip(-slope_percent * 0.002, -0.04, 0.04))
        y_min_ratio += slope_shift

    y_min_ratio = float(np.clip(y_min_ratio, min_y_clamp[0], min_y_clamp[1]))
    y_max_ratio = float(np.clip(y_max_ratio, max_y_clamp[0], max_y_clamp[1]))

    if y_max_ratio <= y_min_ratio + 0.15:
        y_max_ratio = min(max_y_clamp[1], y_min_ratio + 0.25)

    return round(y_min_ratio, 4), round(y_max_ratio, 4)


def crop_road_roi(
    img: np.ndarray,
    y_min_ratio: float | None = None,
    y_max_ratio: float | None = None,
    x_min_ratio: float = 0.0,
    x_max_ratio: float = 1.0,
    slope_percent: float | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Sub-step 2.1: Crop road ROI using per-frame automatic ratios (or explicit ones)."""
    if y_min_ratio is None or y_max_ratio is None:
        auto_min, auto_max = compute_adaptive_roi_ratios(img, slope_percent=slope_percent)
        y_min_ratio = auto_min if y_min_ratio is None else y_min_ratio
        y_max_ratio = auto_max if y_max_ratio is None else y_max_ratio

    h, w = img.shape[:2]
    y1 = max(0, min(h - 1, int(h * y_min_ratio)))
    y2 = max(y1 + 1, min(h, int(h * y_max_ratio)))
    x1 = max(0, min(w - 1, int(w * x_min_ratio)))
    x2 = max(x1 + 1, min(w, int(w * x_max_ratio)))

    return img[y1:y2, x1:x2], (y_min_ratio, y_max_ratio)


def compute_hsv_road_mask(roi_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """Sub-step 2.2: Convert ROI to HSV and build tuned road/border/lane masks."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    _, s_chan, v_chan = cv2.split(hsv)

    # 1) Dark asphalt road surface (low saturation, focused dark value band)
    asphalt_mask = cv2.inRange(hsv, (0, 0, 20), (180, 45, 65))

    # 2) Bright center lane markings (low saturation, clearly brighter value than dark road)
    lane_dash_mask = cv2.inRange(hsv, (0, 0, 75), (180, 45, 255))

    # 3) Colored road borders (noticeable saturation with active brightness)
    colored_border_mask = cv2.inRange(hsv, (0, 35, 30), (180, 255, 255))

    combined_mask = cv2.bitwise_or(asphalt_mask, lane_dash_mask)
    combined_mask = cv2.bitwise_or(combined_mask, colored_border_mask)

    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    stats = {
        "hsv_road_mask_ratio": round(float(np.mean(combined_mask > 0)), 5),
        "hsv_border_mask_ratio": round(float(np.mean(colored_border_mask > 0)), 5),
        "hsv_mean_s": round(float(np.mean(s_chan)), 2),
        "hsv_mean_v": round(float(np.mean(v_chan)), 2),
    }
    return combined_mask, stats


def extract_all_hsv_features(
    df: pd.DataFrame, frames_dir: str, image_col: str = "image"
) -> tuple[pd.DataFrame, dict]:
    """Sub-step 2.2: Compute HSV road/border mask features across all frames."""
    df = df.copy()
    road_ratios = []
    border_ratios = []
    mean_s_vals = []
    mean_v_vals = []
    processed = 0

    for _, row in df.iterrows():
        frame_path = resolve_frame_path(row, frames_dir, image_col=image_col)
        img = cv2.imread(frame_path) if frame_path is not None else None

        if img is None:
            road_ratios.append(np.nan)
            border_ratios.append(np.nan)
            mean_s_vals.append(np.nan)
            mean_v_vals.append(np.nan)
            continue

        slope_val = (
            float(row["slope_percent"])
            if "slope_percent" in row and pd.notna(row["slope_percent"])
            else None
        )
        y_min_r = (
            float(row["roi_y_min"])
            if "roi_y_min" in row and pd.notna(row["roi_y_min"])
            else None
        )
        y_max_r = (
            float(row["roi_y_max"])
            if "roi_y_max" in row and pd.notna(row["roi_y_max"])
            else None
        )

        roi, _ = crop_road_roi(
            img,
            y_min_ratio=y_min_r,
            y_max_ratio=y_max_r,
            slope_percent=slope_val,
        )
        _, stats = compute_hsv_road_mask(roi)

        road_ratios.append(stats["hsv_road_mask_ratio"])
        border_ratios.append(stats["hsv_border_mask_ratio"])
        mean_s_vals.append(stats["hsv_mean_s"])
        mean_v_vals.append(stats["hsv_mean_v"])
        processed += 1

    df["hsv_road_mask_ratio"] = road_ratios
    df["hsv_border_mask_ratio"] = border_ratios
    df["hsv_mean_s"] = mean_s_vals
    df["hsv_mean_v"] = mean_v_vals

    valid_road = df["hsv_road_mask_ratio"].dropna()
    valid_border = df["hsv_border_mask_ratio"].dropna()
    valid_s = df["hsv_mean_s"].dropna()
    valid_v = df["hsv_mean_v"].dropna()

    summary = {
        "frames_processed": processed,
        "road_mask_ratio_range": (round(float(valid_road.min()), 5), round(float(valid_road.max()), 5)),
        "border_mask_ratio_range": (round(float(valid_border.min()), 5), round(float(valid_border.max()), 5)),
        "mean_s_range": (round(float(valid_s.min()), 2), round(float(valid_s.max()), 2)),
        "mean_v_range": (round(float(valid_v.min()), 2), round(float(valid_v.max()), 2)),
    }
    return df, summary


def detect_gaussian_canny_edges(
    roi_bgr: np.ndarray,
    road_mask: np.ndarray | None = None,
    blur_ksize: tuple[int, int] = (5, 5),
    sigma: float = 1.2,
    low_threshold: int = 50,
    high_threshold: int = 150,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sub-step 2.3: Apply Gaussian blur + Canny edge detection and optional HSV road masking."""
    _, w, _ = roi_bgr.shape
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_ksize, sigmaX=sigma, sigmaY=sigma)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    if road_mask is not None:
        dilated_mask = cv2.dilate(road_mask, np.ones((5, 5), np.uint8), iterations=1)
        masked_edges = cv2.bitwise_and(edges, dilated_mask)
    else:
        masked_edges = edges

    mid_x = w // 2
    left_edges = float(np.mean(edges[:, :mid_x] > 0))
    right_edges = float(np.mean(edges[:, mid_x:] > 0))

    stats = {
        "canny_edge_density": round(float(np.mean(edges > 0)), 5),
        "hsv_masked_edge_density": round(float(np.mean(masked_edges > 0)), 5),
        "canny_lr_balance": round(float(right_edges - left_edges), 5),
    }
    return edges, masked_edges, stats


def extract_all_edge_features(
    df: pd.DataFrame, frames_dir: str, image_col: str = "image"
) -> tuple[pd.DataFrame, dict]:
    """Sub-step 2.3: Run Gaussian blur + Canny edge detection across all frames."""
    df = df.copy()
    canny_densities = []
    masked_densities = []
    lr_balances = []
    processed = 0

    for _, row in df.iterrows():
        frame_path = resolve_frame_path(row, frames_dir, image_col=image_col)
        img = cv2.imread(frame_path) if frame_path is not None else None

        if img is None:
            canny_densities.append(np.nan)
            masked_densities.append(np.nan)
            lr_balances.append(np.nan)
            continue

        slope_val = (
            float(row["slope_percent"])
            if "slope_percent" in row and pd.notna(row["slope_percent"])
            else None
        )
        y_min_r = (
            float(row["roi_y_min"])
            if "roi_y_min" in row and pd.notna(row["roi_y_min"])
            else None
        )
        y_max_r = (
            float(row["roi_y_max"])
            if "roi_y_max" in row and pd.notna(row["roi_y_max"])
            else None
        )

        roi, _ = crop_road_roi(
            img,
            y_min_ratio=y_min_r,
            y_max_ratio=y_max_r,
            slope_percent=slope_val,
        )
        road_mask, _ = compute_hsv_road_mask(roi)
        _, _, stats = detect_gaussian_canny_edges(roi, road_mask=road_mask)

        canny_densities.append(stats["canny_edge_density"])
        masked_densities.append(stats["hsv_masked_edge_density"])
        lr_balances.append(stats["canny_lr_balance"])
        processed += 1

    df["canny_edge_density"] = canny_densities
    df["hsv_masked_edge_density"] = masked_densities
    df["canny_lr_balance"] = lr_balances

    valid_ce = df["canny_edge_density"].dropna()
    valid_me = df["hsv_masked_edge_density"].dropna()
    valid_lr = df["canny_lr_balance"].dropna()

    summary = {
        "frames_processed": processed,
        "canny_edge_density_range": (round(float(valid_ce.min()), 5), round(float(valid_ce.max()), 5)),
        "hsv_masked_edge_density_range": (round(float(valid_me.min()), 5), round(float(valid_me.max()), 5)),
        "canny_lr_balance_range": (round(float(valid_lr.min()), 5), round(float(valid_lr.max()), 5)),
    }
    return df, summary


def extract_visual_curvature_features(
    road_mask: np.ndarray, canny_lr_balance: float = 0.0
) -> dict:
    """Sub-step 2.4: Compute normalized road center offset and visual curvature score."""
    h, w = road_mask.shape[:2]
    center_x = w / 2.0

    ys, xs = np.where(road_mask > 0)
    if len(xs) == 0:
        return {
            "road_center_offset": 0.0,
            "visual_curvature_score": round(float(canny_lr_balance), 5),
        }

    overall_x = float(np.mean(xs))
    road_center_offset = (overall_x - center_x) / max(1.0, center_x)

    half_h = max(1, h // 2)
    top_xs = np.where(road_mask[:half_h, :] > 0)[1]
    bot_xs = np.where(road_mask[half_h:, :] > 0)[1]

    if len(top_xs) > 0 and len(bot_xs) > 0:
        top_norm = (float(np.mean(top_xs)) - center_x) / max(1.0, center_x)
        bot_norm = (float(np.mean(bot_xs)) - center_x) / max(1.0, center_x)
        bend_shift = top_norm - bot_norm
    else:
        bend_shift = 0.0

    curvature_score = 0.65 * bend_shift + 0.35 * (canny_lr_balance * 100.0)

    return {
        "road_center_offset": round(float(road_center_offset), 5),
        "visual_curvature_score": round(float(curvature_score), 5),
    }


def extract_all_curvature_features(
    df: pd.DataFrame, frames_dir: str, image_col: str = "image"
) -> tuple[pd.DataFrame, dict]:
    """Sub-step 2.4: Extract visual curvature & road offset features across all frames."""
    df = df.copy()
    offsets = []
    curvature_scores = []
    processed = 0

    for _, row in df.iterrows():
        frame_path = resolve_frame_path(row, frames_dir, image_col=image_col)
        img = cv2.imread(frame_path) if frame_path is not None else None

        if img is None:
            offsets.append(np.nan)
            curvature_scores.append(np.nan)
            continue

        slope_val = (
            float(row["slope_percent"])
            if "slope_percent" in row and pd.notna(row["slope_percent"])
            else None
        )
        y_min_r = (
            float(row["roi_y_min"])
            if "roi_y_min" in row and pd.notna(row["roi_y_min"])
            else None
        )
        y_max_r = (
            float(row["roi_y_max"])
            if "roi_y_max" in row and pd.notna(row["roi_y_max"])
            else None
        )
        lr_bal = (
            float(row["canny_lr_balance"])
            if "canny_lr_balance" in row and pd.notna(row["canny_lr_balance"])
            else 0.0
        )

        roi, _ = crop_road_roi(
            img,
            y_min_ratio=y_min_r,
            y_max_ratio=y_max_r,
            slope_percent=slope_val,
        )
        road_mask, _ = compute_hsv_road_mask(roi)
        curv = extract_visual_curvature_features(road_mask, canny_lr_balance=lr_bal)

        offsets.append(curv["road_center_offset"])
        curvature_scores.append(curv["visual_curvature_score"])
        processed += 1

    df["road_center_offset"] = offsets
    df["visual_curvature_score"] = curvature_scores

    valid_off = df["road_center_offset"].dropna()
    valid_curv = df["visual_curvature_score"].dropna()

    summary = {
        "frames_processed": processed,
        "road_center_offset_range": (round(float(valid_off.min()), 5), round(float(valid_off.max()), 5)),
        "visual_curvature_score_range": (round(float(valid_curv.min()), 5), round(float(valid_curv.max()), 5)),
    }
    return df, summary


def collect_all_roi_ratios(
    df: pd.DataFrame, frames_dir: str, image_col: str = "image"
) -> tuple[pd.DataFrame, dict]:
    """Compute adaptive ROI ratios across all frames and return updated df + summary."""
    df = df.copy()
    y_mins = []
    y_maxs = []
    processed = 0

    for _, row in df.iterrows():
        slope_val = (
            float(row["slope_percent"])
            if "slope_percent" in row and pd.notna(row["slope_percent"])
            else None
        )
        frame_path = resolve_frame_path(row, frames_dir, image_col=image_col)
        img = cv2.imread(frame_path) if frame_path is not None else None

        if img is None:
            y_mins.append(np.nan)
            y_maxs.append(np.nan)
            continue

        y_min_r, y_max_r = compute_adaptive_roi_ratios(img, slope_percent=slope_val)
        y_mins.append(y_min_r)
        y_maxs.append(y_max_r)
        processed += 1

    df["roi_y_min"] = y_mins
    df["roi_y_max"] = y_maxs

    valid_mins = df["roi_y_min"].dropna()
    valid_maxs = df["roi_y_max"].dropna()

    summary = {
        "frames_processed": processed,
        "y_min_range": (round(float(valid_mins.min()), 4), round(float(valid_mins.max()), 4)),
        "y_min_mean": round(float(valid_mins.mean()), 4),
        "y_max_range": (round(float(valid_maxs.min()), 4), round(float(valid_maxs.max()), 4)),
        "y_max_mean": round(float(valid_maxs.mean()), 4),
    }
    return df, summary


def preview_frames_playback(
    df: pd.DataFrame,
    frames_dir: str,
    delay_ms: int = 35,
    image_col: str = "image",
) -> None:
    """Optional visual playback window showing ROI boundaries across frames."""
    window_name = "Mountain Vision AI - ROI Playback (press 'q' or ESC to quit)"
    for idx, row in df.iterrows():
        frame_path = resolve_frame_path(row, frames_dir, image_col=image_col)
        if frame_path is None:
            continue
        img = cv2.imread(frame_path)
        if img is None:
            continue

        y_min_r = float(row["roi_y_min"]) if "roi_y_min" in row and pd.notna(row["roi_y_min"]) else 0.35
        y_max_r = float(row["roi_y_max"]) if "roi_y_max" in row and pd.notna(row["roi_y_max"]) else 1.00

        vis = img.copy()
        h, w = vis.shape[:2]
        y1 = int(h * y_min_r)
        y2 = int(h * y_max_r)

        cv2.line(vis, (0, y1), (w, y1), (0, 255, 255), 2)
        cv2.line(vis, (0, y2), (w, y2), (0, 255, 0), 2)
        label = f"Frame {idx} | ROI y: ({y_min_r:.3f}, {y_max_r:.3f})"
        cv2.putText(vis, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        display_img = cv2.resize(vis, (1280, 480))
        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


def extract_frame_vision_features(
    img: np.ndarray,
    slope_percent: float | None = None,
    y_min_ratio: float | None = None,
    y_max_ratio: float | None = None,
) -> dict:
    roi, (y_min_r, y_max_r) = crop_road_roi(
        img,
        y_min_ratio=y_min_ratio,
        y_max_ratio=y_max_ratio,
        slope_percent=slope_percent,
    )
    _, w, _ = roi.shape

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
        "roi_y_ratios": (y_min_r, y_max_r),
        "roi_shape": roi.shape,
        "edge_density": round(edge_density, 5),
        "mean_brightness": round(mean_brightness, 4),
        "edge_lr_balance": round(edge_lr_balance, 5),
    }


def extract_all_frame_vision_features(
    df: pd.DataFrame, frames_dir: str, image_col: str = "image"
) -> tuple[pd.DataFrame, dict]:
    """Extract frame vision features across all rows and attach them to df."""
    df = df.copy()
    edge_densities = []
    mean_brightnesses = []
    edge_lr_balances = []
    processed = 0

    for _, row in df.iterrows():
        frame_path = resolve_frame_path(row, frames_dir, image_col=image_col)
        img = cv2.imread(frame_path) if frame_path is not None else None

        if img is None:
            edge_densities.append(np.nan)
            mean_brightnesses.append(np.nan)
            edge_lr_balances.append(np.nan)
            continue

        slope_val = (
            float(row["slope_percent"])
            if "slope_percent" in row and pd.notna(row["slope_percent"])
            else None
        )
        y_min_r = (
            float(row["roi_y_min"])
            if "roi_y_min" in row and pd.notna(row["roi_y_min"])
            else None
        )
        y_max_r = (
            float(row["roi_y_max"])
            if "roi_y_max" in row and pd.notna(row["roi_y_max"])
            else None
        )

        feats = extract_frame_vision_features(
            img,
            slope_percent=slope_val,
            y_min_ratio=y_min_r,
            y_max_ratio=y_max_r,
        )
        edge_densities.append(feats["edge_density"])
        mean_brightnesses.append(feats["mean_brightness"])
        edge_lr_balances.append(feats["edge_lr_balance"])
        processed += 1

    df["edge_density"] = edge_densities
    df["mean_brightness"] = mean_brightnesses
    df["edge_lr_balance"] = edge_lr_balances

    valid_ed = df["edge_density"].dropna()
    valid_br = df["mean_brightness"].dropna()
    valid_lr = df["edge_lr_balance"].dropna()

    summary = {
        "frames_processed": processed,
        "edge_density_range": (round(float(valid_ed.min()), 5), round(float(valid_ed.max()), 5)),
        "mean_brightness_range": (round(float(valid_br.min()), 4), round(float(valid_br.max()), 4)),
        "edge_lr_balance_range": (round(float(valid_lr.min()), 5), round(float(valid_lr.max()), 5)),
    }
    return df, summary


if __name__ == "__main__":
    df = load_geometry_data(CSV_PATH)
    df = attach_gps_coordinates(df)
    print(f"CSV loaded: {len(df)} rows")
    if "is_straight_radius" in df.columns:
        print(f"Straight/NaN radius rows: {df['is_straight_radius'].sum()}")

    print("All-rows GPS summary:", summarize_gps_coordinates(df))

    frame_report = verify_frames(df, FRAMES_DIR)
    print(f"Valid frames: {frame_report['valid_count']}/{len(df)}")
    print(f"Missing frames: {len(frame_report['missing_files'])}")
    print(f"Corrupt frames: {len(frame_report['corrupt_files'])}")
    print(f"Image shapes: {frame_report['unique_shapes']}")

    df, roi_summary = collect_all_roi_ratios(df, FRAMES_DIR)
    print("All-frames ROI summary:", roi_summary)

    df, hsv_summary = extract_all_hsv_features(df, FRAMES_DIR)
    print("All-frames HSV summary:", hsv_summary)

    df, edge_summary = extract_all_edge_features(df, FRAMES_DIR)
    print("All-frames Canny edge summary:", edge_summary)

    df, curvature_summary = extract_all_curvature_features(df, FRAMES_DIR)
    print("All-frames curvature summary:", curvature_summary)

    df, vision_summary = extract_all_frame_vision_features(df, FRAMES_DIR)
    print("All-frames vision summary:", vision_summary)

    if PREVIEW_PLAYBACK:
        preview_frames_playback(df, FRAMES_DIR, delay_ms=PREVIEW_DELAY_MS)