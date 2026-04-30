import glob
import os
import re
import subprocess


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except OSError:
        return ""


def _run_v4l2(device: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["v4l2-ctl", "-d", device, *args],
        capture_output=True,
        text=True,
        timeout=2,
    )


def _device_card_name(device: str) -> str:
    name = os.path.basename(device)
    sys_name = f"/sys/class/video4linux/{name}/name"
    return _read_text(sys_name)


def get_camera_paths(device_name: str) -> list[str]:
    """Find /dev/video* nodes whose sysfs-reported camera name matches device_name."""
    video_devices = sorted(
        glob.glob("/dev/video*"),
        key=lambda p: int(re.search(r"\d+$", p).group()),
    )

    matches = []
    inspected = []

    for device in video_devices:
        card_name = _device_card_name(device)
        if card_name:
            inspected.append(f"{device}: {card_name}")

        if device_name not in card_name:
            continue

        try:
            result = _run_v4l2(device, "--list-formats-ext")
        except subprocess.TimeoutExpired:
            continue

        if result.returncode == 0 and result.stdout.strip():
            matches.append(device)

    if matches:
        return matches

    raise Exception(f'Could not find usable device "{device_name}". Found: {inspected}')


def get_device_idx_for_format(
    devices: list[str],
    video_format: str,
    width_px: int,
    height_px: int,
) -> int:
    """Uses v4l2-ctl to find device index that matches a particular resolution and video format."""
    resolution = f"{width_px}x{height_px}"
    outputs = []

    for device in devices:
        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", device, "--list-formats-ext"],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except subprocess.TimeoutExpired:
            outputs.append(f"{device}: TIMEOUT")
            continue

        output = result.stdout or ""
        outputs.append(f"{device}:\n{output}")

        if result.returncode == 0 and resolution in output and video_format in output:
            return int(device.rsplit("video", 1)[1])

    output_str = "\n" + "\n".join(outputs)

    raise Exception(
        f"Couldn't find device idx for format {video_format}, {width_px}x{height_px}. " f"Found only: {output_str}"
    )
