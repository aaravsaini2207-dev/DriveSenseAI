# DriveSense AI 🚗

### An End-to-End Computer Vision Pipeline for Road Scene Understanding

DriveSense AI is a modular computer vision pipeline that analyzes road-scene videos using object detection, multi-object tracking, lane detection, motion analysis, and monocular depth estimation.

It combines YOLO11n, ByteTrack, and Depth Anything V2 Small to process road footage and generate annotated video output alongside performance and evaluation reports.

## Features

* **Object Detection:** Detects objects using YOLO11n.
* **Multi-Object Tracking:** Tracks detected objects across video frames using ByteTrack.
* **Lane Detection:** Identifies visible lane markings.
* **Motion Analysis:** Estimates image-plane motion from consecutive frames.
* **Monocular Depth Estimation:** Generates relative depth maps using Depth Anything V2 Small.
* **Object-Level Depth:** Aggregates depth-map values within tracked object bounding boxes.
* **Performance Profiling:** Measures processing time across pipeline stages.
* **Evaluation Tools:** Provides scripts for benchmarking, tracking diagnostics, depth-consistency analysis, and annotation inspection.
* **Automated Tests:** Includes unit tests for video utilities and object-level depth estimation.

## Architecture

```text
Input Road Video
       |
       v
Video Processing and Resizing
       |
       +-----------------------------+
       |              |              |
       v              v              v
Object Detection   Lane Detection   Motion Analysis
       |
       v
ByteTrack Tracking
       |
       v
Tracked Objects
       |
       +---------------------------+
       |                           |
       v                           v
Depth Estimation             Object-Level Depth
       |                           ^
       v                           |
Relative Depth Map ---------------+
       |
       v
Profiling and Annotated Video Output
```

*Conceptual overview; actual execution order is defined by the pipeline implementation.*

## Tech Stack

| Component                  | Technology                             |
| -------------------------- | -------------------------------------- |
| Language                   | Python                                 |
| Object detection           | YOLO11n                                |
| Multi-object tracking      | ByteTrack                              |
| Monocular depth estimation | Depth Anything V2 Small                |
| Computer vision            | OpenCV                                 |
| Deep learning              | PyTorch                                |
| Testing                    | pytest                                 |
| Evaluation                 | Custom Python scripts and JSON metrics |

## Project Structure

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
│   ├── benchmark.py
│   ├── evaluate_depth.py
│   ├── evaluate_tracking.py
│   ├── extract_frames.py
│   ├── create_contact_sheet.py
│   ├── auto_annotate.py
│   ├── summarize_annotations.py
│   └── generate_report.py
├── tests/
│   ├── test_video.py
│   └── test_object_depth.py
├── data/
│   └── outputs/
│       └── EVALUATION_REPORT.md
├── run_pipeline.py
├── requirements.txt
└── README.md
```

## Requirements

* Python 3.12 recommended
* Git
* Internet access for installing dependencies and obtaining model files
* A CPU capable of running PyTorch inference

A GPU is not required, but CPU inference may be slow.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/aaravsaini2207-dev/DriveSenseAI.git
cd DriveSenseAI
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, consult the Python documentation for platform-specific virtual environment instructions.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CPU-only PyTorch installation, use the installation command recommended by the official PyTorch installation selector if the default dependency installation does not provide a suitable CPU build.

### 4. Obtain the Depth Anything V2 source code

The Depth Anything V2 repository and checkpoint are not included in this Git repository.

Clone the upstream project into the expected directory:

```powershell
git clone https://github.com/DepthAnything/Depth-Anything-V2.git external/Depth-Anything-V2
```

### 5. Obtain the depth checkpoint

Download the official **Depth Anything V2 Small** checkpoint from the upstream project's documented checkpoint source.

Place the checkpoint at:

```text
external/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth
```

The checkpoint is a large model file and is intentionally excluded from this repository.

Refer to the official [Depth Anything V2 repository](https://github.com/DepthAnything/Depth-Anything-V2) for model details and checkpoint availability.

### 6. Provide a YOLO model and input video

The pipeline uses the YOLO11n model file:

```text
yolo11n.pt
```

Ensure the model is available at the path expected by your project configuration. Ultralytics may download supported pretrained weights automatically when the configured model name is used and network access is available.

Place your input road video at the path configured in the project. The development video is not included in the repository.

### 7. Configure Python imports

From the project root in PowerShell:

```powershell
$env:PYTHONPATH = "."
```

### 8. Run the pipeline

```powershell
python -u run_pipeline.py
```

The current pipeline writes its processed video to:

```text
data/outputs/drivesense_output.mp4
```

The input video, YOLO weights, and Depth Anything V2 checkpoint must be available at the locations expected by the code.

## Testing

Run the automated tests:

```powershell
python -m pytest -q
```

The current test suite contains 13 tests covering video utilities and object-level depth estimation.

## Benchmarking and Evaluation

Run the pipeline benchmark:

```powershell
python -u -m evaluation.benchmark --depth-interval 15
```

Evaluate tracking diagnostics:

```powershell
python -u -m evaluation.evaluate_tracking
```

Evaluate depth consistency:

```powershell
python -u -m evaluation.evaluate_depth
```

Generate the evaluation report:

```powershell
python -u -m evaluation.generate_report
```

Evaluation artifacts are written under `data/outputs/` and `data/outputs/metrics/`.

## Performance and Evaluation Results

The following results were recorded during development on a road video with these characteristics:

* **Input resolution:** 3840 × 2160
* **Source frame rate:** 29.97 FPS
* **Frames processed:** 338
* **Recorded processing time:** 69.67 seconds
* **Recorded throughput:** 4.85 FPS
* **Depth inference calls:** 23, using a 15-frame refresh interval

### Tracking diagnostics

| Metric                            |          Result |
| --------------------------------- | --------------: |
| Frames processed                  |             338 |
| Unique track IDs observed         |             105 |
| Object observations               |             818 |
| Average observations per track ID |            7.79 |
| Longest track                     | 64 observations |
| Track IDs with gaps               |              53 |

These are statistics from tracker output, not measurements of ground-truth tracking accuracy.

### Depth-consistency diagnostics

| Metric                                   | Result |
| ---------------------------------------- | -----: |
| Frames processed                         |    338 |
| Depth inference calls                    |     34 |
| Track IDs with valid depth               |    105 |
| Track IDs with at least 3 observations   |     70 |
| Mean absolute consecutive depth change   | 0.0586 |
| Mean normalized consecutive depth change | 0.1060 |

The depth-consistency evaluation used a 10-frame refresh interval, whereas the benchmark above used a 15-frame interval. These results therefore come from different configurations.

Depth-change values describe consistency between observations. They do not establish depth accuracy or physical distance in metres.

For additional details, see [the evaluation report](data/outputs/EVALUATION_REPORT.md).

## Limitations

* Processing throughput in the recorded configuration is below real-time video frame rates.
* Monocular depth outputs are relative estimates, not calibrated physical distances.
* Detection and tracking accuracy have not been established using comprehensive ground-truth annotations.
* Tracking gaps may result from occlusion, missed detections, objects leaving the frame, or identity fragmentation.
* Performance depends on hardware, system load, model settings, and inference configuration.
* This project is a road-scene analysis prototype, not a validated autonomous-driving or vehicle-safety system.

## Future Improvements

* Evaluate object detection against manually annotated ground-truth boxes.
* Measure tracking quality using ground-truth trajectories.
* Validate depth estimates against suitable depth ground truth.
* Improve inference throughput and investigate pipeline bottlenecks.
* Test lane detection across different road layouts and lighting conditions.
* Compare configurations using reproducible benchmarks.
* Add representative annotated frames and output-video demonstrations.

## Author

**Aarav Saini**

* GitHub: [aaravsaini2207-dev](https://github.com/aaravsaini2207-dev)
* LinkedIn: [Aarav Saini](https://www.linkedin.com/in/aarav-saini/)

---

*Built as a computer vision and deep learning project focused on road-scene understanding, modular inference, and performance evaluation.*
