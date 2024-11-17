import logging
import os
import sys
from dataclasses import dataclass

import cv2 as cv
from cv2.typing import MatLike

from camera import get_camera_paths, get_device_idx_for_format


def configure_logger(log_level: str | int | None = None) -> logging.Logger:
    """Creates a logger, whose level can be set via parameter or env var LOG_LEVEL"""
    logger = logging.getLogger()

    if not logger.hasHandlers():  # Avoid duplicate handlers if reused
        # Create a handler for stdout
        if log_level is None:
            log_level = os.environ.get("LOG_LEVEL", logging.DEBUG)
        logger.setLevel(log_level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)

        # Define the format for log messages
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(handler)

    return logger


@dataclass
class CameraSource:
    device_name: str
    video_format: str
    width_px: int
    height_px: int
    framerate: int


@dataclass
class FileSource:
    file_path: str


def configure_video_source(source: FileSource | CameraSource) -> cv.VideoCapture:
    """Create an OpenCV capture object based on whether it is a file or camera source"""
    # Configure video source
    if isinstance(source, CameraSource):
        width_px = source.width_px
        height_px = source.height_px

        device_paths = get_camera_paths(source.device_name)
        device_idx = get_device_idx_for_format(device_paths, source.video_format, width_px, height_px)

        cap = cv.VideoCapture(device_idx)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, width_px)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, height_px)

        fourcc = cv.VideoWriter_fourcc(*source.video_format)
        cap.set(cv.CAP_PROP_FOURCC, fourcc)
    elif isinstance(source, FileSource):
        cap = cv.VideoCapture(source.file_path)
    else:
        raise Exception(f"Unhandled video source type {type(source)}")

    return cap


class VideoTarget:
    def write(self, frame: MatLike, annotations: dict | None = None):
        raise NotImplementedError("Should be implemented by child class")

    def close(self):
        raise NotImplementedError("Should be implemented by child class")
