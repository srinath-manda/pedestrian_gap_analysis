# Pedestrian Gap Acceptance Analysis

Automated analysis of pedestrian gap acceptance behaviour at urban intersections
using YOLOv8, ByteTrack, DeepFace, and Binary Logistic Regression.

---

## Requirements

- Windows 10/11 (64-bit)
- Python 3.11 (recommended) — https://www.python.org/downloads/
- NVIDIA GPU with CUDA (optional but recommended for speed)

---

## Installation (Step by Step)

### Step 1 — Clone the project

```bash
git clone https://github.com/srinath-manda/pedestrian_gap_analysis.git
cd pedestrian-gap-acceptance
```

Or if sharing as a ZIP — extract it and open a terminal in that folder.

---

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs: ultralytics, deepface, tensorflow, statsmodels, pandas, numpy,
opencv-contrib-python, matplotlib, scipy, hypothesis

> First run will also auto-download:
> - YOLOv8n model weights (~6 MB)
> - DeepFace gender model (~537 MB)
> - DeepFace age model (~539 MB)

---

### Step 4 — Install ffmpeg (for .MTS video conversion)

```bash
winget install ffmpeg
```

Then **close and reopen your terminal** so ffmpeg is on PATH.

---

## Usage

### Step 1 — Convert your .MTS video to MP4

```bash
ffmpeg -i "C:\path\to\your_video.MTS" -c copy "C:\path\to\output.mp4"
```

### Step 2 — Pick your crossing polygon

Run this to open a browser-based polygon picker on the first frame:

```bash
python pick_polygon.py --video "C:\path\to\output.mp4" --output "output"
```

Click on the crossing zone in the browser, press **Done**, copy the generated command.

### Step 3 — Run the full pipeline

```bash
python -m pedestrian_gap_analysis.main ^
  --video "C:\path\to\output.mp4" ^
  --output "output" ^
  --polygon "x1,y1 x2,y2 x3,y3 x4,y4"
```

### Step 4 — Generate statistical report and plots

```bash
python generate_report.py ^
  --csv "output\gap_acceptance_dataset.csv" ^
  --output "output"
```

---

## Output Files

| File | Description |
|---|---|
| `gap_acceptance_dataset.csv` | Per-pedestrian crossing records |
| `logit_model_summary.txt` | Binary logistic regression summary |
| `statistical_report.txt` | Full OLS regression report with stats |
| `annotated_output.mp4` | Video with bounding boxes and labels |
| `plot_01_gap_duration_histogram.png` | Gap duration distribution |
| `plot_02_gap_by_gender.png` | Gap duration by gender |
| `plot_03_gap_by_age_group.png` | Gap duration by age group |
| `plot_04_gap_by_platoon.png` | Gap duration by platoon behaviour |
| `plot_05_gap_vs_vehicle_speed.png` | Gap vs vehicle speed scatter |
| `plot_06_gap_vs_time_headway.png` | Gap vs time headway scatter |
| `plot_07_ols_coefficients.png` | OLS regression coefficients |
| `plot_08_residuals_qqplot.png` | Residuals and Q-Q plot |
| `plot_09_summary_dashboard.png` | Full summary dashboard |

---

## CSV Columns

| Column | Type | Description |
|---|---|---|
| track_id | int | ByteTrack pedestrian ID |
| gender | str | Male / Female / Unknown |
| age_group | str | Young / Middle / Old / Unknown |
| platoon | str | Group / Alone |
| gap_seconds | float | Accepted gap duration (seconds) |
| time_headway | float | Mean inter-vehicle headway (seconds) |
| vehicle_speed | float | Estimated vehicle speed (px/s) |
| gap_type | int | 1 = Straight Gap, 0 = Rolling Gap |

---

## Troubleshooting

**OpenCV GUI error (cvNamedWindow)**
Use `pick_polygon.py` instead — it opens in your browser, no OpenCV window needed.

**CUDA not available**
Add `--device cpu` to the main command to force CPU mode.

**lap package warning**
Ultralytics will auto-install it on first run. Safe to ignore.

**protobuf conflict warnings**
These come from other packages (streamlit, wandb). They do not affect this project.
