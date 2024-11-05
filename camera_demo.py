import subprocess
import numpy as np
import cv2 as cv
import re


def get_camera_indices() -> list[int]:
    output = subprocess.check_output(
        "v4l2-ctl --list-devices", shell=True, text=True
    )
    devices = output.strip().split("\n\n")

    device_name = "USB 2.0 Camera"

    # Iterate through each camera block to find the device name
    for device in devices:
        if device_name in device:
            # Find all video device entries associated with this camera
            video_devices = re.findall(r"/dev/video(\d+)", device)
            # Convert the first match to an integer index
            if video_devices:
                return [int(index) for index in video_devices]

    raise Exception(f'Could not find device "{device_name}". Found: {devices}')


# Copied from https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
def play_video(device_idx: int, enable_mjpg: bool = True):
    """Streams video in window

    Args:
        device_idx (int): device idx to use
        enable_mjpg (bool, optional): Enables higher resolution MJPG format. 
            Defaults to True.
    """
    cap = cv.VideoCapture(device_idx)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    if enable_mjpg:
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        cap.set(cv.CAP_PROP_FOURCC, fourcc)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv.imshow("frame", gray)
        if cv.waitKey(1) == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    # print(cv.getBuildInformation())
    device_indices = get_camera_indices()
    print(f"Camera indices: {device_indices}")
    chosen_idx = device_indices[0]
    print(f"Using idx {chosen_idx}..")
    play_video(chosen_idx)
