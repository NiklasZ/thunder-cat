import os
import subprocess
from dataclasses import dataclass

import cv2 as cv

from src.camera import get_camera_paths, get_device_idx_for_format
from src.motion_detection import label_motion


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


def stream_video(source: CameraSource | FileSource):

    # Set receiver info
    host_ip_env = "HOST_IP"
    host_ip = os.environ.get(host_ip_env)
    host_port_env = "HOST_PORT"
    host_port = os.environ.get(host_port_env)

    assert host_ip, f'Host IP address env "{host_ip_env}" not set'
    assert host_port, f'Host port env "{host_port_env}" not set'

    # Configure video source
    if isinstance(source, CameraSource):
        width_px = source.width_px
        height_px = source.height_px
        framerate = source.framerate

        device_paths = get_camera_paths(source.device_name)
        device_idx = get_device_idx_for_format(device_paths, source.video_format, width_px, height_px)

        cap = cv.VideoCapture(device_idx)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, width_px)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, height_px)

        fourcc = cv.VideoWriter_fourcc(*source.video_format)
        cap.set(cv.CAP_PROP_FOURCC, fourcc)
    elif isinstance(source, FileSource):
        cap = cv.VideoCapture(source.file_path)
        width_px = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height_px = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        framerate = cap.get(cv.CAP_PROP_FPS)
    else:
        raise Exception(f"Unhandled video source type {type(source)}")

    # Check source
    if not cap.isOpened():
        print("Cannot open video source")
        exit()

    # FFMPEG UDP streaming config
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",  # Input format
        "-pixel_format",
        "bgr24",  # Correct parameter for pixel format
        "-video_size",
        f"{width_px}x{height_px}",  # Correct parameter for frame size
        "-framerate",
        f"{framerate}",  # Correct parameter for frame rate
        "-i",
        "-",  # Input from stdin
        "-vcodec",
        "libx264",  # Use H.264 codec for better compatibility
        "-preset",
        "ultrafast",  # Use ultrafast preset to reduce latency
        "-tune",
        "zerolatency",  # Optimize for low latency
        "-g",
        "25",  # Set keyframe interval
        "-pix_fmt",
        "yuv420p",  # Set pixel format for compatibility
        "-f",
        "mpegts",  # Output format
        f"udp://{host_ip}:{host_port}?pkt_size=1316",  # UDP address with packet size
    ]

    # Stream video
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    try:
        back_sub = cv.createBackgroundSubtractorMOG2(history=2000, varThreshold=20, detectShadows=True)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, _ = label_motion(frame, back_sub, minimum_area=1000)
            process.stdin.write(frame.tobytes())

    except Exception as e:
        print("Streaming interrupted.")
        print(f"Error: {e}")

    finally:
        cap.release()
        process.stdin.close()
        process.wait()
        print("Streaming stopped.")


if __name__ == "__main__":
    # source = FileSource("data/video/evaluation/not_cat/notcat2.mp4")

    source = CameraSource(device_name="USB 2.0 Camera", video_format="MJPG", width_px=640, height_px=480, framerate=30)
    stream_video(source)
