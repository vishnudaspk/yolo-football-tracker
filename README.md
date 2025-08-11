# yolo-football-tracker

**YOLOv8-based intelligent football tracking system with stickman pose rendering**

---

## Overview

This project uses YOLOv8 models to detect and track two key objects in sports footage:

* **Football**: Detected using bounding boxes (YOLOv8x).
* **Players**: Detected as persons (YOLOv8x), and overlaid with stickman pose estimation (YOLOv8x-pose).

The system supports enhanced video preprocessing (exposure correction, contrast tuning), object re-detection, motion-based filtering, and trail visualizations.

---

## Features

* Real-time video analysis
* Multi-model architecture using:

  * `yolov8x.pt` for general detection
  * `yolov8x-pose.pt` for human pose estimation
  * `yolov8n.pt` (optional, lighter model for edge deployment)
* Trail visualization for ball movement
* Dynamic stickman pose rendering for moving players
* Adaptive exposure enhancement
* Smooth bounding boxes with temporal averaging

---

## Demo (Screenshots)

* ![Sample Output 1](images/1.png)
* ![Sample Output 2](images/2.png)
* ![Sample Output 3](images/3.png)
* ![Sample Output 4](images/4.png)
* ![Sample Output 4](images/4.png)

---

## Project Structure

```bash
yolo-football-tracker/
├── videos/                # Input test videos
├── outputs/               # Annotated video outputs
├── main.py                # Main tracking and detection pipeline
├── .gitignore             # Ignored files/folders
├── README.md              # Project documentation
├── weights/               # (Optional) Store YOLO weights here (excluded in git)
└── tracker_configs/       # Optional tracker YAMLs (e.g. bytetrack.yaml)
```

---

## Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>Example <code>requirements.txt</code></summary>

```text
ultralytics
opencv-python
numpy
```

</details>

### 2. Run main script

```bash
python main.py
```

### 3. View output

Check the annotated video in `outputs/annotated_test2.mp4`

---

## Configuration

All runtime parameters are set at the top of `main.py`, including:

* Confidence thresholds
* Movement detection pixel limits
* Frame enhancement settings
* Smoothing parameters
* Tracker type: `bytetrack.yaml` or `botsort.yaml`

---

## YOLO Models Used

| Model             | Purpose                          |
| ----------------- | -------------------------------- |
| `yolov8x.pt`      | Ball and player detection        |
| `yolov8x-pose.pt` | Pose estimation for players      |
| `yolov8n.pt`      | Lightweight detection (optional) |

---

## Notes & Tips

* For overexposed videos, preprocessing enhances results.
* Static ball detection uses a dual-pass strategy to avoid missed detections.
* Trail length and smoothing are adjustable for aesthetic tuning.

---

## Acknowledgments

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* OpenCV, NumPy, and Python community

---

## Contact

Feel free to reach out via [GitHub Issues](https://github.com/your-username/yolo-football-tracker/issues) for bugs or suggestions.

---
