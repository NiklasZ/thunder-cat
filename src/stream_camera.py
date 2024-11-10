import cv2 as cv
import subprocess
import os

from src.camera import get_camera_paths, get_device_idx_for_format


def stream_camera(
    device_idx: int, video_format: str, width_px: int, height_px: int, framerate: int
):

    # Set receiver info
    host_ip_env = "HOST_IP"
    host_ip = os.environ.get(host_ip_env)
    host_port_env = "HOST_PORT"
    host_port = os.environ.get(host_port_env)

    assert host_ip, f'Host IP address env "{host_ip_env}" not set'
    assert host_port, f'Host port env "{host_port_env}" not set'

    # Configure camera
    cap = cv.VideoCapture(device_idx)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    cap.set(cv.CAP_PROP_FRAME_WIDTH, width_px)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height_px)

    fourcc = cv.VideoWriter_fourcc(*video_format)
    cap.set(cv.CAP_PROP_FOURCC, fourcc)

    # FFMPEG UDP sending config
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

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break
            process.stdin.write(frame.tobytes())

    except:
        print("Streaming interrupted.")

    finally:
        cap.release()
        process.stdin.close()
        process.wait()
        print("Streaming stopped.")


if __name__ == "__main__":
    device_name = "USB 2.0 Camera"
    video_format = "MJPG"
    width_px = 1920
    height_px = 1080
    frame_rate = 30

    device_paths = get_camera_paths(device_name)
    device_idx = get_device_idx_for_format(
        device_paths, video_format, width_px, height_px
    )
    stream_camera(device_idx, video_format, width_px, height_px, frame_rate)
