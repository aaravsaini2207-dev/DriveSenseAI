# DriveSense AI 🚗

### An End-to-End Computer Vision Pipeline for Road Scene Understanding

DriveSense AI is a computer vision project that combines object detection, multi-object tracking, lane detection, motion analysis, and monocular depth estimation to analyze road scenes from video.

The project processes road footage through a modular pipeline and generates annotated video output alongside performance and evaluation reports.

## ✨ Features

* **Object Detection:** Detects road-scene objects using YOLO11n.
* **Multi-Object Tracking:** Maintains object track IDs across video frames using ByteTrack.
* **Lane Detection:** Identifies visible lane markings.
* **Motion Analysis:** Estimates motion-related information from consecutive frames.
* **Monocular Depth Estimation:** Uses Depth Anything V2 Small to estimate relative scene depth.
* **Object-Level Depth:** Aggregates depth-map values within tracked object bounding boxes.
* **Performance Profiling:** Measures processing time and latency across pipeline stages.
* **Evaluation Tools:** Includes scripts for benchmarking, annotation inspection, tracking analysis, and depth-consistency analysis.

## 🏗️ Architecture

```text
                Input Road Video
                       |
                       v
                Video Processing
                       |
                       v
                 Frame Resizing
                       |
          +------------+------------+
          |            |            |
          v            v            v
    Object Detection  Lane        Motion
          |          Detection    Analysis
          v
    ByteTrack Tracking
          |
          v
    Tracked Objects
          |
          +----------------------+
          |                      |
          v                      v
    Depth Estimation      Object-Level Depth
          |                      ^
          v                      |
    Relative Depth Map -----------+
          |
          v
    Profiling and Video Output
```

*Conceptual overview; the implementation and execution order are defined by the pipeline code.*

## 🧰 Tech Stack

| Component                  | Technology                             |
| -------------------------- | -------------------------------------- |
| Language                   | Python                                 |
| Object detection           | YOLO11n                                |
| Multi-object tracking      | ByteTrack                              |
| Monocular depth estimation | Depth Anything V2 Small                |
| Computer vision            | OpenCV                                 |
| Deep learning              | PyTorch                                |
| Evaluation                 | Custom Python scripts and JSON metrics |

## 📁 Project Structure

```text
DriveSenseAI/
├── src/
│   └── drivesense/
│       ├── config.py
│       ├── depth.py
│       ├── lanes.py
│       ├── motion.py
│       ├── object_depth.py
│       ├── pipeline.py
│       ├── profiling.py
│       ├── tracker.py
│       └── video.py
├── evaluation/
│   ├── auto_annotate.py
│   ├── benchmark.py
│   ├── evaluate_depth.py
│   ├── evaluate_tracking.py
│   ├── extract_frames.py
│   ├── create_contact_sheet.py
│   ├── summarize_annotations.py
│   └── generate_report.py
├── data/
│   ├── videos/
│   └── outputs/
│       ├── metrics/
│       ├── benchmark_output.mp4
│       └── EVALUATION_REPORT.md
├── run_pipeline.py
└── README.md
```

## 📊 Performance and Evaluation

The pipeline was tested on a road video with the following characteristics:

* **Input resolution:** 3840 × 2160
* **Source frame rate:** 29.97 FPS
* **Frames processed:** 338
* **Latest recorded benchmark processing time:** 69.67 seconds
* **Latest recorded processing throughput:** 4.85 FPS
* **Depth inference calls:** 23, using a 15-frame refresh interval

### Tracking diagnostics

The separate tracking evaluation recorded:

| Metric                            |          Result |
| --------------------------------- | --------------: |
| Frames processed                  |             338 |
| Unique track IDs observed         |             105 |
| Object observations               |             818 |
| Average observations per track ID |            7.79 |
| Longest track                     | 64 observations |
| Track IDs with gaps               |              53 |

These are tracker-output statistics, not ground-truth tracking accuracy measurements.

### Depth-consistency diagnostics

The separate depth evaluation recorded:

| Metric                                   | Result |
| ---------------------------------------- | -----: |
| Frames processed                         |    338 |
| Depth inference calls                    |     34 |
| Track IDs with valid depth               |    105 |
| Track IDs with at least 3 observations   |     70 |
| Mean absolute consecutive depth change   | 0.0586 |
| Mean normalized consecutive depth change | 0.1060 |

The depth evaluation used a 10-frame refresh interval, unlike the latest benchmark's 15-frame interval. The results therefore represent separate configurations.

These depth-change values measure consistency between consecutive observations. They do not measure depth accuracy or physical distance in metres.

For detailed results, see [`data/outputs/EVALUATION_REPORT.md`](data/outputs/EVALUATION_REPORT.md).

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone <https://github.com/aaravsaini2207-dev>
cd DriveSenseAI
```

### 2. Create a virtual environment

Python 3.12 was used in the development environment.

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the dependencies listed in the project's `requirements.txt`.

```bash
pip install -r requirements.txt
```

The repository also requires the appropriate model weights and Depth Anything V2 checkpoint. Follow the project's setup instructions for obtaining these files.

### 4. Configure Python imports

From the project root, run:

```powershell
$env:PYTHONPATH = "."
```

### 5. Run the pipeline

```bash
python -u run_pipeline.py
```

The pipeline writes its processed video to:

```text
data/outputs/drivesense_output.mp4
```

Ensure the input video and required model checkpoints are available before running.

## 🧪 Evaluation

Run the benchmark:

```bash
python -u -m evaluation.benchmark --depth-interval 15
```

Evaluate tracking:

```bash
python -u -m evaluation.evaluate_tracking
```

Evaluate depth consistency:

```bash
python -u -m evaluation.evaluate_depth
```

Generate the evaluation report:

```bash
python -u -m evaluation.generate_report
```

Evaluation outputs are stored in `data/outputs/` and `data/outputs/metrics/`.

## ⚠️ Limitations

* Processing speed is below real-time video frame rates in the current tested configuration.
* Monocular depth outputs are relative estimates, not calibrated distances in metres.
* Detection and tracking accuracy have not been established using comprehensive ground-truth annotations.
* Tracking gaps can result from occlusion, missed detections, objects leaving the frame, or ID fragmentation.
* Performance depends on hardware, system load, model settings, and inference configuration.
* This project is a road-scene analysis prototype, not a validated autonomous-driving or vehicle-safety system.

## 🔭 Future Improvements

* Evaluate detection performance against manually annotated ground-truth boxes.
* Measure tracking quality using ground-truth trajectories.
* Validate depth estimates against suitable depth ground truth.
* Improve inference throughput and investigate pipeline bottlenecks.
* Test lane detection across different road layouts and lighting conditions.
* Add reproducible benchmark comparisons across configurations.
* Expand visual documentation with annotated sample frames and output videos.

## 👤 Author

**Aarav Saini**

* GitHub: [aaravsaini2207-dev](https://github.com/aaravsaini2207-dev)
* LinkedIn: [Aarav Saini](https://www.linkedin.com/in/aarav-saini/)

---

*Built as a computer vision and deep learning project focused on road-scene understanding, modular inference, and performance evaluation.*
