import json
import os
import subprocess
import threading
import time
from datetime import datetime, timedelta
from io import TextIOWrapper
from typing import List

import numpy as np
from cv2.typing import MatLike

from configuration import (
    VideoTarget,
    configure_logger,
    current_timestamp,
    to_date_str,
    to_timestamp,
)

DEFAULT_VIDEO_LOG_DIR = "data/log"

logger = configure_logger()


def check_current_memory_usage_gb(directory: str) -> float:
    total_size = 0
    for dirpath, _, filenames in os.walk(directory):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            # Skip if the file is a broken symlink
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
    return total_size / 1024**3


def get_oldest_file(directory: str) -> str:
    oldest_file = None
    oldest_time = time.time()  # Current time

    for dirpath, _, filenames in os.walk(directory):
        for file in filenames:
            file_path = os.path.join(dirpath, file)
            if os.path.exists(file_path):
                file_time = os.path.getctime(file_path)  # Creation time
                if file_time < oldest_time:
                    oldest_time = file_time
                    oldest_file = file_path

    return oldest_file


def delete_empty_folders(directory: str):
    """
    Recursively deletes all empty folders in the specified directory.

    Args:
        directory (str): The path to the directory to clean.
    """
    # Walk the directory tree bottom-up
    for dirpath, dirnames, filenames in os.walk(directory, topdown=False):
        # If the directory is empty (no files and no subdirectories)
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
                print(f"Deleted empty folder: {dirpath}")
            except OSError as e:
                print(f"Error deleting folder {dirpath}: {e}")


def manage_directory_size(directory: str, max_size_gb: float):
    """Checks whether given directory uses more than the specified maxmimum.
    If so, it deletes the oldest files until it is below the limit again.
    """
    current_size = check_current_memory_usage_gb(directory)
    logger.info(f"Current size of {directory}: {current_size:.2f}/{max_size_gb}GB")
    delete_empty_folders(directory)
    while check_current_memory_usage_gb(directory) >= max_size_gb:
        oldest_file = get_oldest_file(directory)
        if oldest_file:
            os.remove(oldest_file)
            logger.info(f"Deleted oldest file {oldest_file}")
        else:
            logger.info("Found no files to delete!")
            break
    if current_size >= max_size_gb:
        current_size = check_current_memory_usage_gb(directory)
        logger.info(f"Size of {directory} after cleanup: {current_size:.2f}/{max_size_gb}GB")


# We use FFMPEG directly to have more control over the encoder and use a less
# memory intensive format.
def generate_ffmpeg_cmd(width: int, height: int, fps: int, video_path: str) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        video_path,
    ]


def save_history_frames(width: int, height: int, fps: int, video_path: str, frames: List[MatLike]):
    ffmpeg_cmd = generate_ffmpeg_cmd(width, height, fps, video_path)
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    for f in frames:
        process.stdin.write(f.tobytes())

    process.stdin.close()
    process.wait()


class VideoLogger(VideoTarget):
    def __init__(
        self,
        fps: int,
        width_px: int,
        height_px: int,
        logging_dir=DEFAULT_VIDEO_LOG_DIR,
        max_log_size_gb=10,
        check_memory_every=18000,  # 10 min at 30 FPS
        max_video_frames=108000,  # 1 hour at 30 FPS
    ):
        self.logging_dir = logging_dir
        self.max_log_size_gb = max_log_size_gb
        self.check_memory_every = check_memory_every
        self.max_video_frames = max_video_frames
        self.frame_counter = 0
        self.fps = fps
        self.width_px = width_px
        self.height_px = height_px
        self.last_frame_written: datetime | None = None
        self.last_frame_written_str: str | None = None
        self.base_log_path: str | None = None
        self.video_writer: subprocess.Popen[bytes] | None = None
        self.annotation_writer: TextIOWrapper | None = None

    def write(self, frame: MatLike, annotations: dict | None = None):
        """Writes provided frame to video file. Also creates identically named log
        file containing annotations for that particular frame.
        """
        annotations = annotations or {}
        annotations = {k: v for k, v in annotations.items() if k != "class_logits"}
        # Periodically check memory use
        if self.frame_counter % self.check_memory_every == 0:
            manage_directory_size(self.logging_dir, self.max_log_size_gb)

        current_time = datetime.now()
        # Start a new log file if:
        # 1. there is no existing one or no video writer
        # 2. it has been 1 minute since we last wrote
        # 3. we have already written the maximum frames to this one
        if (
            not self.last_frame_written
            or not self.video_writer
            or (current_time - self.last_frame_written > timedelta(minutes=1))
            or self.frame_counter > self.max_video_frames
        ):
            self._close_current_writers()
            self.frame_counter = 0
            self.last_frame_written = current_time
            self.last_frame_written_str = to_timestamp(self.last_frame_written)
            date_str = to_date_str(self.last_frame_written)
            self.base_log_path = os.path.join(self.logging_dir, date_str)
            os.makedirs(self.base_log_path, exist_ok=True)
            video_path = os.path.join(self.base_log_path, f"{self.last_frame_written_str}.mp4")

            ffmpeg_cmd = generate_ffmpeg_cmd(self.width_px, self.height_px, self.fps, video_path)
            self.video_writer = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

            annotations_path = os.path.join(self.base_log_path, f"{self.last_frame_written_str}.log")
            self.annotation_writer = open(annotations_path, "w")
            annotations["start_time"] = self.last_frame_written_str
            logger.info(f"Started writing to new files: {annotations_path}, {video_path}")

        self.video_writer.stdin.write(frame.tobytes())

        if "back_sub_history" in annotations:
            history = annotations.pop("back_sub_history")
            # Make a copy as we are passing this to a thread
            history_cpy = [np.copy(arr) for arr in history]
            history_path = os.path.join(self.base_log_path, f"{self.last_frame_written_str}_history.mp4")
            logger.info(f"Saving background subtractor state to {history_path}")

            thread = threading.Thread(
                target=save_history_frames,
                args=(self.width_px, self.height_px, self.fps, history_path, history_cpy),
                daemon=True,
            )
            thread.start()

        if len(annotations):
            self.annotation_writer.write(f"Frame {self.frame_counter}: {json.dumps(annotations)}\n")
            self.annotation_writer.flush()

        self.last_frame_written = current_time
        self.frame_counter += 1

    def _close_current_writers(self):
        if self.video_writer:
            self.video_writer.stdin.close()
            self.video_writer.wait()
            self.video_writer = None
        if self.annotation_writer:
            self.annotation_writer.write(
                f"Frame {self.frame_counter}: {json.dumps({'end_time':current_timestamp()})}\n"
            )
            self.annotation_writer.close()
            self.annotation_writer = None
            logger.info("Finished writing files")

    def close(self):
        self._close_current_writers()
