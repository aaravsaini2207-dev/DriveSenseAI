# DriveSense AI — Evaluation Report

Generated: 2026-10-30 01:29

## 1. Project Overview

DriveSense AI is a computer-vision pipeline combining object detection, multi-object tracking, lane detection, motion analysis, and monocular relative-depth estimation.

Evaluation video: `data/videos/road.mp4`

## 2. Multi-Object Tracking

- Frames processed: 338
- Evaluation processing time: 65.53 seconds
- Evaluation throughput: 5.16 FPS
- Unique track IDs observed: 105
- Object observations: 818
- Average observations per track ID: 7.79
- Median observations per track ID: 5
- Longest track: 64 observations
- Tracks with gaps: 53
- Reappearance gaps recorded: 168

### Class observations

| Class | Observations |
|---|---:|
| bicycle | 49 |
| person | 350 |
| car | 385 |
| traffic light | 3 |
| cell phone | 31 |

**Interpretation:** Track IDs and observation counts describe the tracker’s output. They do not establish identity accuracy, ID-switch rate, or tracking precision without ground-truth annotations.

## 3. Relative-Depth Consistency

- Frames processed: 338
- Evaluation processing time: 114.87 seconds
- Evaluation throughput: 2.94 FPS
- Depth refresh interval: 10 frames
- Depth inference calls: 34
- Track IDs with valid depth: 105
- Track IDs with at least three observations: 70
- Valid depth observations: 818
- Mean absolute consecutive depth change: 0.0586
- Mean normalized consecutive depth change: 0.1060

**Interpretation:** Consecutive depth changes measure temporal consistency, not depth accuracy. Changes may reflect object motion, camera motion, tracking errors, or model variation.

The depth model produces relative depth estimates. These values must not be interpreted as calibrated distances in metres.

## 4. Pipeline Performance Benchmark

- Frames processed: N/A
- Processing time: N/A seconds
- Average FPS: N/A
- Depth inference calls: N/A

Raw benchmark JSON: `data/outputs/benchmark_results.json`

## 5. Evaluation Limitations

1. **Detection accuracy:** Ground-truth bounding boxes and class labels are required to calculate precision, recall, and mAP.
2. **Tracking accuracy:** Ground-truth trajectories are required to calculate metrics such as IDF1, HOTA, and MOTA.
3. **Depth accuracy:** Calibrated depth ground truth is required to measure absolute or relative depth error.
4. **Temporal consistency:** Consecutive depth-change statistics are diagnostic indicators, not accuracy metrics.
5. **Performance:** FPS depends on hardware, system load, warm-up, model settings, and inference configuration.
6. **Visual review:** Qualitative inspection complements numerical evaluation but does not replace ground-truth testing.

## 6. Next Improvements

- Validate detection results against manually annotated frames.
- Evaluate tracking using annotated object trajectories.
- Investigate missed detections and track fragmentation.
- Profile detection, depth inference, and frame resizing separately.
- Repeat benchmarks under consistent conditions and report median FPS.
- Validate lane detection on additional road scenes and lighting conditions.

## 7. Reproducibility

The source evaluation outputs are stored under `data/outputs/metrics/`. Preserve the original JSON files alongside this report so the reported values can be independently checked.
