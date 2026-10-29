import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "data" / "outputs" / "metrics"
OUTPUTS = ROOT / "data" / "outputs"

TRACKING_FILE = METRICS / "tracking_metrics.json"
DEPTH_FILE = METRICS / "depth_metrics.json"
BENCHMARK_FILE = OUTPUTS / "benchmark_results.json"
REPORT_FILE = OUTPUTS / "EVALUATION_REPORT.md"


def load_json(path):
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def fmt(value, digits=2):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def generate_report():
    tracking = load_json(TRACKING_FILE)
    depth = load_json(DEPTH_FILE)
    benchmark = load_json(BENCHMARK_FILE)

    lines = [
        "# DriveSense AI — Evaluation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 1. Project Overview",
        "",
        "DriveSense AI is a computer-vision pipeline combining object "
        "detection, multi-object tracking, lane detection, motion analysis, "
        "and monocular relative-depth estimation.",
        "",
        "Evaluation video: `data/videos/road.mp4`",
        "",
    ]

    # Tracking evaluation
    lines += ["## 2. Multi-Object Tracking", ""]

    if tracking:
        lines += [
            f"- Frames processed: {tracking.get('frames_processed', 'N/A')}",
            f"- Evaluation processing time: "
            f"{fmt(tracking.get('processing_seconds'))} seconds",
            f"- Evaluation throughput: "
            f"{fmt(tracking.get('evaluation_fps'))} FPS",
            f"- Unique track IDs observed: "
            f"{tracking.get('total_unique_track_ids', 'N/A')}",
            f"- Object observations: "
            f"{tracking.get('total_object_observations', 'N/A')}",
            f"- Average observations per track ID: "
            f"{fmt(tracking.get('average_track_observations'))}",
            f"- Median observations per track ID: "
            f"{tracking.get('median_track_observations', 'N/A')}",
            f"- Longest track: "
            f"{tracking.get('longest_track_observations', 'N/A')} observations",
            f"- Tracks with gaps: {tracking.get('tracks_with_gaps', 'N/A')}",
            f"- Reappearance gaps recorded: "
            f"{tracking.get('total_reappearance_gaps', 'N/A')}",
            "",
            "### Class observations",
            "",
            "| Class | Observations |",
            "|---|---:|",
        ]

        for label, count in tracking.get("class_observations", {}).items():
            lines.append(f"| {label} | {count} |")

        lines += [
            "",
            "**Interpretation:** Track IDs and observation counts describe "
            "the tracker’s output. They do not establish identity accuracy, "
            "ID-switch rate, or tracking precision without ground-truth "
            "annotations.",
            "",
        ]
    else:
        lines += ["Tracking metrics file not found.", ""]

    # Depth evaluation
    lines += ["## 3. Relative-Depth Consistency", ""]

    if depth:
        lines += [
            f"- Frames processed: {depth.get('frames_processed', 'N/A')}",
            f"- Evaluation processing time: "
            f"{fmt(depth.get('processing_seconds'))} seconds",
            f"- Evaluation throughput: "
            f"{fmt(depth.get('evaluation_fps'))} FPS",
            f"- Depth refresh interval: "
            f"{depth.get('depth_interval', 'N/A')} frames",
            f"- Depth inference calls: "
            f"{depth.get('depth_inference_calls', 'N/A')}",
            f"- Track IDs with valid depth: "
            f"{depth.get('tracks_with_valid_depth', 'N/A')}",
            f"- Track IDs with at least three observations: "
            f"{depth.get('tracks_with_at_least_3_observations', 'N/A')}",
            f"- Valid depth observations: "
            f"{depth.get('total_valid_depth_observations', 'N/A')}",
            f"- Mean absolute consecutive depth change: "
            f"{fmt(depth.get('overall_mean_absolute_consecutive_change'), 4)}",
            f"- Mean normalized consecutive depth change: "
            f"{fmt(depth.get('overall_mean_normalized_consecutive_change'), 4)}",
            "",
            "**Interpretation:** Consecutive depth changes measure temporal "
            "consistency, not depth accuracy. Changes may reflect object "
            "motion, camera motion, tracking errors, or model variation.",
            "",
            "The depth model produces relative depth estimates. These values "
            "must not be interpreted as calibrated distances in metres.",
            "",
        ]
    else:
        lines += ["Depth metrics file not found.", ""]

    # Benchmark
    lines += ["## 4. Pipeline Performance Benchmark", ""]

    if benchmark:
        # Support either a flat JSON report or a nested benchmark summary.
        data = benchmark
        if isinstance(benchmark, dict):
            data = benchmark.get("summary", benchmark)

        if isinstance(data, dict):
            lines += [
                f"- Frames processed: "
                f"{data.get('frames_processed', data.get('frames', 'N/A'))}",
                f"- Processing time: "
                f"{fmt(data.get('processing_seconds', data.get('total_seconds')))} "
                f"seconds",
                f"- Average FPS: "
                f"{fmt(data.get('average_fps', data.get('fps')))}",
                f"- Depth inference calls: "
                f"{data.get('depth_inference_calls', data.get('depth_calls', 'N/A'))}",
                "",
            ]

        lines += [
            "Raw benchmark JSON: `data/outputs/benchmark_results.json`",
            "",
        ]
    else:
        lines += [
            "Benchmark JSON was not found at "
            "`data/outputs/benchmark_results.json`.",
            "",
            "The latest benchmark recorded separately in the terminal was:",
            "",
            "- Frames processed: 338",
            "- Processing time: 69.67 seconds",
            "- Average processing speed: 4.85 FPS",
            "- Depth inference calls: 23",
            "- Depth interval: 15 frames",
            "",
            "These figures are from the latest terminal output, not "
            "necessarily the run represented by the tracking or depth files.",
            "",
        ]

    # Limitations
    lines += [
        "## 5. Evaluation Limitations",
        "",
        "1. **Detection accuracy:** Ground-truth bounding boxes and class "
        "labels are required to calculate precision, recall, and mAP.",
        "2. **Tracking accuracy:** Ground-truth trajectories are required "
        "to calculate metrics such as IDF1, HOTA, and MOTA.",
        "3. **Depth accuracy:** Calibrated depth ground truth is required "
        "to measure absolute or relative depth error.",
        "4. **Temporal consistency:** Consecutive depth-change statistics "
        "are diagnostic indicators, not accuracy metrics.",
        "5. **Performance:** FPS depends on hardware, system load, warm-up, "
        "model settings, and inference configuration.",
        "6. **Visual review:** Qualitative inspection complements numerical "
        "evaluation but does not replace ground-truth testing.",
        "",
        "## 6. Next Improvements",
        "",
        "- Validate detection results against manually annotated frames.",
        "- Evaluate tracking using annotated object trajectories.",
        "- Investigate missed detections and track fragmentation.",
        "- Profile detection, depth inference, and frame resizing separately.",
        "- Repeat benchmarks under consistent conditions and report median FPS.",
        "- Validate lane detection on additional road scenes and lighting "
        "conditions.",
        "",
        "## 7. Reproducibility",
        "",
        "The source evaluation outputs are stored under "
        "`data/outputs/metrics/`. Preserve the original JSON files alongside "
        "this report so the reported values can be independently checked.",
        "",
    ]

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print(f"Evaluation report generated:\n{REPORT_FILE}")


if __name__ == "__main__":
    generate_report()