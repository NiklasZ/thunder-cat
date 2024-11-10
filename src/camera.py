import re
import subprocess


def get_camera_paths(device_name: str) -> list[str]:
    """Uses v4l2-ctl to get list of found video device paths for a given hardware str."""
    output = subprocess.check_output("v4l2-ctl --list-devices", shell=True, text=True)
    devices = output.strip().split("\n\n")

    # Iterate through each camera block to find the device name
    for device in devices:
        if device_name in device:
            # Find all video device entries associated with this camera
            video_devices = re.findall(r"/dev/video\d+", device)
            # Convert the first match to an integer index
            if len(video_devices):
                return video_devices
            else:
                raise Exception(
                    f"Could not find any associated video devices for:\n{device}"
                )

    raise Exception(f'Could not find device "{device_name}". Found: {devices}')


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
        result = subprocess.run(
            ["v4l2-ctl", "-d", device, "--list-formats-ext"],
            capture_output=True,
            text=True,
        )
        output = result.stdout
        outputs.append(output)

        if resolution in output and video_format in output:
            return int(device[-1])

    output_str = "\n" + "\n".join(outputs)

    raise Exception(
        f"Couldn't find device idx for format {video_format}, {height_px}x{width_px}. Found only: {output_str}"
    )
