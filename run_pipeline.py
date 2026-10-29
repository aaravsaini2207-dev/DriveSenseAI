print("STEP 1: Starting DriveSense AI", flush=True)

from src.drivesense.pipeline import DriveSensePipeline

print("STEP 2: Pipeline module imported", flush=True)


def main() -> None:
    print("STEP 3: Initializing pipeline", flush=True)

    pipeline = DriveSensePipeline(
    max_width=640,
    depth_interval=15,
)

    print("STEP 4: Pipeline initialized", flush=True)

    pipeline.run(
        video_path="data/videos/road.mp4",
        output_path="data/outputs/drivesense_output.mp4",
        display=True,
    )


if __name__ == "__main__":
    main()
