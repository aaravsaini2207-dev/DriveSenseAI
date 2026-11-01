import imageio_ffmpeg
import subprocess

input_video = "data/outputs/drivesense_output.mp4"
output_video = "data/outputs/drivesense_output_slow.mp4"

# Change this to 22 for a 22-second video.
target_duration = 22

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

# Read the input video's duration.
probe = subprocess.run(
    [
        ffmpeg, "-i", input_video,
        "-f", "null", "-"
    ],
    capture_output=True,
    text=True,
)

import re

log = probe.stderr
match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", log)

if not match:
    raise RuntimeError("Could not determine input video duration.")

hours, minutes, seconds = match.groups()
original_duration = (
    int(hours) * 3600
    + int(minutes) * 60
    + float(seconds)
)

speed_factor = original_duration / target_duration

if speed_factor <= 0 or speed_factor > 1:
    raise ValueError("Target duration must exceed the original duration.")

print(f"Original duration: {original_duration:.2f} seconds")
print(f"Target duration:   {target_duration:.2f} seconds")
print(f"Playback speed:    {speed_factor:.3f}x")

subprocess.run(
    [
        ffmpeg, "-y",
        "-i", input_video,
        "-vf", f"setpts={1 / speed_factor:.8f}*PTS",
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_video,
    ],
    check=True,
)

print(f"\nDone! Saved slowed video to: {output_video}")
