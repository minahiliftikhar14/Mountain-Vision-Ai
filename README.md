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
