import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime

import cv2 as cv
import pandas as pd
from cv2.typing import MatLike

from camera import get_camera_paths, get_device_idx_for_format

# We do some manual annotation as we sometimes capture frames for classification
# that aren't of good quality or ended up picking noise over the intended target
VERIFIED_LABEL = "verified"  # class label matches (manually verified)
JUNK_LABEL = "junk"  # class label does not match (manually verified)
SOUND_FOLDER = "data/sound"


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


logger = configure_logger()


def to_timestamp(d: datetime) -> str:
    return d.strftime("%Y_%m_%d-%H_%M_%S")


def current_timestamp() -> str:
    return to_timestamp(datetime.now())


@dataclass
class CameraSource:
    device_name: str
    video_format: str
    width_px: int
    height_px: int


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
        cap.set(cv.CAP_PROP_BUFFERSIZE, 5)
        logger.info(f"Opening camera: {source.device_name}")
    elif isinstance(source, FileSource):
        cap = cv.VideoCapture(source.file_path)
        logger.info(f"Loading video: {source.file_path}")
    else:
        raise Exception(f"Unhandled video source type {type(source)}")

    return cap


class VideoTarget:
    def write(self, frame: MatLike, annotations: dict | None = None):
        raise NotImplementedError("Should be implemented by child class")

    def close(self):
        raise NotImplementedError("Should be implemented by child class")


def filter_junk(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["label"] != JUNK_LABEL]


def cls_str(idx: int) -> str:
    return f"cls_{idx}"


def cat_cls_str(class_id: int) -> str:
    return f"cat_cls_{class_id}"


MODEL_FOLDER_PATH = "model"
