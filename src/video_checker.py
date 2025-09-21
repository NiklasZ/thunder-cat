#!/usr/bin/env python3
"""Filters for videos with a certain minimum length (seconds) and from the last few days.
"""
import datetime
import os
import subprocess
from typing import List, Tuple


def get_video_duration_s(filepath: str) -> float:
    """
    Returns the duration of a video file in seconds using ffprobe.
    Raises an exception if ffprobe fails or the file is not a valid video.
    """
    command = [
        "ffprobe",
        "-v",
        "error",  # Suppress all output except errors
        "-show_entries",
        "format=duration",  # Get the duration from the format section
        "-of",
        "default=noprint_wrappers=1:nokey=1",  # Output only the raw duration value
        filepath,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def list_long_videos(search_dir: str, min_duration_s: int = 15, max_days_old: int = 3):
    """
    Finds videos in a directory that are longer than a given duration
    and were modified within a specified number of days.

    Args:
        search_dir (str): The directory to search for videos.
        min_duration_s (int): The minimum duration in seconds for a video to be listed.
        max_days_old (int): The maximum age of a video file in days based on its modification time.
    """
    results: List[Tuple[str, float]] = []
    now = datetime.datetime.now()
    time_threshold = now - datetime.timedelta(days=max_days_old)
    video_extensions = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv")

    print(
        f"Searching for videos in '{search_dir}' modified in the last {max_days_old} "
        f"days and longer than {min_duration_s} seconds..."
    )

    for root, dirs, files in os.walk(search_dir):
        dirs.sort()  # Process directories in a consistent order
        for filename in sorted(files):
            if not filename.lower().endswith(video_extensions):
                continue

            filepath = os.path.join(root, filename)

            try:
                # 1. Check file modification date
                file_mod_timestamp = os.path.getmtime(filepath)
                file_mod_datetime = datetime.datetime.fromtimestamp(file_mod_timestamp)

                if file_mod_datetime < time_threshold:
                    continue  # Skip file if it's too old

                # 2. Check video duration
                duration = get_video_duration_s(filepath)
                if duration > min_duration_s:
                    results.append((filepath, duration))

            except FileNotFoundError:
                print("Error: 'ffprobe' could not be found.")
                print("Please ensure ffmpeg is installed and accessible in your system's PATH.")
                return
            except (subprocess.CalledProcessError, ValueError, OSError) as e:
                print(f"Warning: Could not process file '{filepath}'. Skipping. Reason: {e}")
                continue

    # Pretty-print the results
    if not results:
        print("\nNo matching videos found.")
        return

    print(f"\nFound {len(results)} matching videos.")
    max_path_len = max((len(path) for path, _ in results), default=20)

    print(f"\n{'Video File'.ljust(max_path_len)}  {'Duration (s)'}")
    print(f"{'-' * max_path_len}  {'-' * 12}")

    for path, duration in results:
        print(f"{path.ljust(max_path_len)}  {duration:<12.2f}")


if __name__ == "__main__":
    # --- Configuration ---
    # The directory where your videos are stored.
    VIDEO_DIRECTORY = "data/log"

    # Minimum video duration in seconds.
    K_SECONDS = 25

    # How many days back to look for recently modified files.
    D_DAYS = 3
    list_long_videos(
        search_dir=VIDEO_DIRECTORY,
        min_duration_s=K_SECONDS,
        max_days_old=D_DAYS,
    )
