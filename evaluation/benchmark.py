import argparse
import json
import platform
from datetime import datetime
from pathlib import Path

from src.drivesense.pipeline import DriveSensePipeline


def main():
    parser = argparse.ArgumentParser(description="DriveSense AI Benchmark")
    parser.add_argument(
        "--input",
        default="data/videos/road.mp4",
        help="Path to input video",
    )
    parser.add_argument(
        "--output",
        default="data/outputs/benchmark_output.mp4",
        help="Path to output video",
    )
    parser.add_argument(
        "--depth-interval",
        type=int,
        default=15,
        help="Run depth estimation every N frames",
    )
    args = parser.parse_args()

    if args.depth_interval < 1:
        parser.error("--depth-interval must be at least 1")

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        parser.error(f"Input video not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Starting DriveSense AI benchmark...")

    pipeline = DriveSensePipeline(
        max_width=640,
        depth_interval=args.depth_interval,
    )

    report = pipeline.run(
        video_path=str(input_path),
        output_path=str(output_path),
    )

    results = {
        "project": "DriveSense AI",
        "timestamp": datetime.now().astimezone().isoformat(),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "device": "CPU",
        },
        "configuration": {
            "input_video": str(input_path.resolve()),
            "output_video": str(output_path.resolve()),
            "max_width": 640,
            "depth_interval": args.depth_interval,
        },
        "metrics": {
            "frames_processed": report.frames_processed,
            "processing_seconds": report.processing_seconds,
            "average_fps": report.average_fps,
            "depth_inference_calls": report.depth_inference_calls,
        },
        "stage_timings": report.stage_timings,
    }

    results_path = Path("data/outputs/benchmark_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n========== BENCHMARK SUMMARY ==========")
    print(f"Frames: {report.frames_processed}")
    print(f"Processing time: {report.processing_seconds:.2f}s")
    print(f"Average FPS: {report.average_fps:.2f}")
    print(f"Depth calls: {report.depth_inference_calls}")
    print(f"JSON results: {results_path.resolve()}")


if __name__ == "__main__":
    main()