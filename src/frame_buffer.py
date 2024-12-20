from queue import Full, Queue
from threading import Event, Thread

import cv2 as cv

from configuration import configure_logger

logger = configure_logger()


def capture_frames(
    cap: cv.VideoCapture, frame_buffer: Queue, stop_event: Event, no_more_frames_event: Event, blocking: bool
):
    """Capture frames from the camera and store them in the buffer."""
    while not stop_event.is_set():  # Check if the stop signal is set
        ret, frame = cap.read()
        if not ret:
            logger.info("Could not capture another frame from camera.")
            break
        try:
            # Put frames either blockin
            if blocking:
                frame_buffer.put(frame)
            else:
                frame_buffer.put_nowait(frame)
        except Full:
            logger.warning("Frame buffer is full. Dropping frame.")

    no_more_frames_event.set()
    logger.info("No more frames to process. Capture thread exiting...")


def create_frame_buffer_thread(
    cap: cv.VideoCapture, buffer_size: int, blocking: bool
) -> tuple[Thread, Event, Event, Queue]:
    stop_event = Event()
    no_more_frames_event = Event()
    frame_buffer = Queue(maxsize=buffer_size)
    capture_thread = Thread(
        target=capture_frames, args=(cap, frame_buffer, stop_event, no_more_frames_event, blocking), daemon=True
    )
    capture_thread.start()
    logger.info(f"Launched thread {capture_thread.ident} to capture and buffer frames")
    return capture_thread, stop_event, no_more_frames_event, frame_buffer
