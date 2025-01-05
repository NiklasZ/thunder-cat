from queue import Full, Queue
from threading import Event, Thread

import cv2 as cv

from configuration import configure_logger

logger = configure_logger()


def capture_frames(
    cap: cv.VideoCapture,
    frame_buffer: Queue,
    stop_event: Event,
    no_more_frames_event: Event,
    blocking: bool,
    original_fps: int,
    reduced_fps: int,
):
    """Capture frames from the camera and store them in the buffer."""
    dropped_frame_counter = 0
    next_frame_time = 0
    increased_time_per_frame = 1 / reduced_fps
    original_per_frame_time = 1 / original_fps
    assert (
        reduced_fps <= original_fps
    ), f"Reduced FPS {reduced_fps} should be less than or equal to original FPS {original_fps}"

    while not stop_event.is_set():  # Check if the stop signal is set
        ret, frame = cap.read()
        if not ret:
            logger.info("Could not capture another frame from camera.")
            break

        # Frame rate reduction logic
        # Skip every kth frame to reduce the frame rate.
        # For example: if the original FPS is 20 and the reduced FPS is 16,
        # then we skip every 5th frame to achieve the reduced FPS.
        next_frame_time += original_per_frame_time
        if next_frame_time < increased_time_per_frame:
            continue
        next_frame_time -= increased_time_per_frame

        try:
            # Check if the buffer is almost full and drop the frame if necessary
            if frame_buffer.full():
                dropped_frame_counter += 1
                if dropped_frame_counter % 100 == 0:
                    logger.warning(f"Dropped {dropped_frame_counter} frames.")
                continue

            # Put frames into the buffer
            if blocking:
                frame_buffer.put(frame)
            else:
                frame_buffer.put_nowait(frame)
        except Full:
            dropped_frame_counter += 1
            logger.error("Frame buffer is full. Dropping frame.")

    no_more_frames_event.set()
    logger.info("No more frames to process. Capture thread exiting...")


def create_frame_buffer_thread(
    cap: cv.VideoCapture, buffer_size: int, blocking: bool, original_fps: int, reduced_fps: int
) -> tuple[Thread, Event, Event, Queue]:
    stop_event = Event()
    no_more_frames_event = Event()
    frame_buffer = Queue(maxsize=buffer_size)
    capture_thread = Thread(
        target=capture_frames,
        args=(cap, frame_buffer, stop_event, no_more_frames_event, blocking, original_fps, reduced_fps),
        daemon=True,
    )
    capture_thread.start()
    logger.info(f"Launched thread {capture_thread.ident} to capture and buffer frames")
    return capture_thread, stop_event, no_more_frames_event, frame_buffer
