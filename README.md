🏔️ Mountain Vision AI

Synthetic Vision-Guided Autonomous Navigation & MDP Control Framework

Mountain Vision AI is a multimodal autonomous navigation and supervisory control prototype engineered specifically for complex, elevation-varying mountain road terrains. Combining a custom Blender 3D synthetic mountain dataset, an OpenCV road-perception pipeline, lookahead bend prediction, and a Markov Decision Process (MDP) solved via the Bellman Optimality Equation, the system computes safe, context-aware recommendations for steering angle, target speed, and gear selection.

🎯 Project Vision & Motivation

Standard lane-keeping systems often struggle on mountain passes due to sharp switchbacks, blind crests, S-curves, and continuous elevation gradients where gravity and load significantly impact vehicle dynamics. Mountain Vision AI bridges visual perception and sequential decision-making through a two-tier architecture:

Perception ("Eye"): Extracts road geometry, boundary cues, and curvature signals from driver-POV dashcam frames paired with 3D spatial trajectory telemetry.

Decision Engine ("Brain"): Formulates mountain driving as a discrete Markov Decision Process (MDP) and solves the Bellman Optimality Equation to output optimal control actions tailored to upcoming bend sharpness and road slope.

🏗️ 6-Component System Architecture

The end-to-end workflow is structured into six cooperating stages:

┌──────────────────────────────────────────────────────────────────────────────┐
│  1. Data Ingestion, Synchronization & Simulated GPS Attachment               │
│     (road_geometry.csv [287 rows] + dashcam_frames/ [0014.jpg–0300.jpg])     │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  2. Computer Vision Road & Curvature Extraction                              │
│     (Adaptive ROI Cropping ──► HSV Masking ──► Gaussian Blur ──► Canny)      │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  3. Upcoming Bend & Angle Prediction                                         │
│     (Lookahead Trajectory + Vision Cues ──► Turn Direction, Angle & Alerts)  │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  4. MDP State, Action & Reward Formulation                                   │
│     (State: Bend Angle, Distance, Slope, Speed, Gear | Action: Steer/Gear/V) │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  5. Bellman Optimality Control Solver                                        │
│     (Dynamic Programming Value Iteration ──► V*(s) & Optimal Policy π*(s))   │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  6. Pipeline Execution, Driver Alerts & Matplotlib Telemetry Visualization   │
│     (End-to-End Evaluation Logs + Multi-Panel Trajectory & Control Plots)    │
└──────────────────────────────────────────────────────────────────────────────┘


Component Breakdown

Stage

Module / Focus

Core Responsibilities

Component 1

Data Ingestion, Sync & GPS Attachment

Loads road_geometry.csv (287 entries), pairs each record with RGB dashcam frames (0014.jpg–0300.jpg) using fallback filename resolution and cv2.imread verification, safely flags straight segments (radius_m = NaN $\rightarrow$ is_straight_radius & np.inf), and projects 3D coordinates (cam_x, cam_y, cam_z, distance_m) into local simulated GPS fields (gps_lat, gps_lon, gps_alt).

Component 2

Computer Vision Road & Curvature Extraction

Applies per-frame Adaptive Road ROI Cropping, HSV color-space conversion & road/border masking, Gaussian filtering, and Canny edge detection to extract road edge density, left-right edge balance, road center offset, and visual curvature scores.

Component 3

Upcoming Bend & Angle Prediction

Analyzes lookahead road geometry (curvature_1_per_m, radius_m, road_yaw_deg, turn, bend_type, distance_m), GPS progression, and visual cues to forecast the next approaching bend's direction, angle ($\Delta\theta$), and distance, generating timely driver alerts.

Component 4

MDP State, Action & Reward Formulation

Models mountain driving under elevation and curvature constraints. States capture bend angle, distance-to-bend, slope (slope_percent), and vehicle speed/gear; actions output (steering_angle, gear, target_speed) governed by safety- and comfort-driven rewards.

Component 5

Bellman Optimality Control Solver

Uses Dynamic Programming to solve $V^*(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s' \mid s,a) V^*(s') \right]$ and derive the deterministic optimal policy $\pi^*(s)$ across road topologies.

Component 6

Execution, Alerts & Visualization

Runs the full pipeline across the dataset, logs bend alerts and control recommendations, and renders Matplotlib diagnostic plots for GPS/3D trajectory, bend angles, and optimal control profiles.

🛣️ Synthetic Dataset Specifications (Blender 3D)

The multimodal dataset was custom-built in Blender 3D using a Bezier centerline curve and a driver-POV camera rig:

Total Track Span: ~1000 m (984.69 m active synchronized trajectory across 287 frames, 720 × 1920 RGB resolution).

Visual Lane Cues: Asphalt road surface with solid yellow left boundary line, solid blue right boundary line, and central white dashed lane markings.

Road Topologies Included:

Straight Segments (133 frames) — baseline state transitions.

Gentle & Moderate Turns — sweeping mountain curves with smooth lateral transitions.

S-Curves / Chicanes — alternating left-right steering reversals.

Inclined Turns & Elevation Grades — continuous Z-axis elevation climbs (cam_z progression) testing slope-aware gear and throttle decisions.

Hairpin Switchbacks — tight, high-curvature turns requiring low gear and maximum steering authority.

📂 Repository Structure

Mountain-Vision-Ai/
├── check.py                        # Fast dataset & directory integrity verification script
├── train_agent.py                  # Core perception, ROI/HSV/Canny extraction & MDP pipeline
├── road_geometry.csv               # Synchronized 287-row ground-truth geometry & trajectory dataset
├── dashcam_frames/                 # Driver-POV RGB image sequence (0014.jpg – 0300.jpg)
├── Mountain Vision dataset.blend   # Source Blender 3D mountain environment file
└── README.md                       # Project documentation


⚙️ Environment Setup & Installation (macOS / Linux)

Clone the repository and navigate into the project folder:

git clone https://github.com/minahiliftikhar14/Mountain-Vision-Ai.git
cd Mountain-Vision-Ai


Create and activate a Python virtual environment:

python3 -m venv venv
source venv/bin/activate


Upgrade pip and install dependencies:

python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip
python3 -m pip install --only-binary=:all: numpy pandas matplotlib opencv-python


🚀 Usage

Verify Dataset & Frame Synchronization:
Confirm that road_geometry.csv (287 rows) and dashcam_frames/ are properly aligned before running the pipeline:

python3 check.py


Run the Vision & Navigation Pipeline:
Execute data ingestion, simulated GPS mapping, adaptive ROI cropping, HSV/Canny feature extraction, and control evaluation:

python3 train_agent.py


Tip: To visually inspect the per-frame adaptive ROI crop across all 287 frames in a playback window, set PREVIEW_PLAYBACK = True at the top of train_agent.py (press q or ESC to exit playback).

🛠️ Tech Stack

Simulation & 3D Data Generation: Blender 3D, Blender Python API

Computer Vision & Image Processing: OpenCV (cv2 — Adaptive ROI, HSV Color Space, Gaussian Filtering, Canny Edge Detection, Sobel Gradients)

Data Engineering & Scientific Computing: Python, NumPy, Pandas

Control & Decision-Making: Markov Decision Processes (MDP), Dynamic Programming (Bellman Optimality Equation)

Telemetry & Visualization: Matplotlib

🗺️ Roadmap

[x] Phase 1: Blender 3D mountain road environment modeling & multimodal dataset export (287 synchronized frames + geometry CSV)

[x] Phase 2 (Component 1): Data ingestion, frame verification, NaN straight-road handling, and local simulated GPS projection

[x] Phase 3 (Component 2): Adaptive ROI cropping, HSV road/border masking, Gaussian blur, Canny edge extraction, and visual curvature metrics

[ ] Phase 4 (Component 3): Lookahead upcoming bend & angle predictor with supervisory driver alerts

[ ] Phase 5 (Components 4 & 5): MDP state/action/reward formulation & Bellman Optimality solver for steering, gear, and speed

[ ] Phase 6 (Component 6): End-to-end Matplotlib telemetry plots & Sim2Real edge-dashcam deployment exploration