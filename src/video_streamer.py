import os
import subprocess

from cv2.typing import MatLike

from configuration import VideoTarget


class VideoStreamer(VideoTarget):
    def __init__(
        self,
        fps: int,
        width_px: int,
        height_px: int,
    ):

        # Set receiver info
        host_ip_env = "HOST_IP"
        self.host_ip = os.environ.get(host_ip_env)
        host_port_env = "HOST_PORT"
        self.host_port = os.environ.get(host_port_env)

        assert self.host_ip, f'Host IP address env "{host_ip_env}" not set'
        assert self.host_port, f'Host port env "{host_port_env}" not set'

        self.fps = fps
        self.width_px = width_px
        self.height_px = height_px
        self.video_writer: subprocess.Popen[bytes] | None = None

    def write(self, frame: MatLike, annotations: dict | None = None):
        """Streams video to target host IP and port number via UDP."""

        if not self.video_writer:
            # FFMPEG UDP streaming config
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",  # Input format
                "-pixel_format",
                "bgr24",  # Correct parameter for pixel format
                "-video_size",
                f"{self.width_px}x{self.height_px}",  # Correct parameter for frame size
                "-framerate",
                f"{self.fps}",  # Correct parameter for frame rate
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
                f"udp://{self.host_ip}:{self.host_port}?pkt_size=1316",  # UDP address with packet size
            ]

            # Stream video
            self.video_writer = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        self.video_writer.stdin.write(frame.tobytes())

    def close(self):
        if self.video_writer:
            self.video_writer.stdin.close()
            self.video_writer.wait()
            self.video_writer = None
