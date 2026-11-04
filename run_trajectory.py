from pathlib import Path

from src.drivesense.trajectory_pipeline import TrajectoryPipeline


def main():
    print("STEP 1: Starting DriveSenseAI Trajectory Analysis")

    video_path = Path("data/videos/traffic_static.mp4")
    output_path = Path("data/outputs/trajectory_output.mp4")

    print("STEP 2: Initializing trajectory pipeline")

    pipeline = TrajectoryPipeline()

    print("STEP 3: Running trajectory analysis")

    report = pipeline.run(
        video_path=video_path,
        output_path=output_path,
        display=True,
    )

    print("\n========== TRAJECTORY ANALYSIS COMPLETE ==========")
    print(f"Frames processed: {report.frames_processed}")
    print(f"Processing time: {report.processing_seconds:.2f}s")
    print(f"Average FPS: {report.average_fps:.2f}")
    print(f"Output: {report.output_path}")


if __name__ == "__main__":
    main()