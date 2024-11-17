import cv2 as cv

from configuration import (
    CameraSource,
    FileSource,
    VideoTarget,
    configure_logger,
    configure_video_source,
)
from src.motion_detection import detect_motion
from video_logger import VideoLogger

logger = configure_logger()


def main(
    source: CameraSource | FileSource,
    targets: list[VideoTarget],
    label_video: bool,
    min_consecutive_motion_frames: int,
    stop_recording_after: int,
):
    cap = configure_video_source(source)

    # Check source
    if not cap.isOpened():
        print("Cannot open video source")
        exit()

    try:
        # TODO move all the extra tooling away to its own method at some point
        back_sub = cv.createBackgroundSubtractorMOG2(history=2000, varThreshold=23, detectShadows=True)
        frame_counter = 0
        frames_since_last_motion = 0
        consecutive_change_frames = 0
        motion_detected_once = False
        logger.info("Started video source reading")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, bounding_boxes = detect_motion(frame, back_sub, draw_bounding_boxes=label_video, minimum_area=500)
            change_detected = bool(len(bounding_boxes))

            if change_detected:
                consecutive_change_frames += 1
            else:
                consecutive_change_frames = 0

            motion_detected = consecutive_change_frames > min_consecutive_motion_frames

            if motion_detected:
                frames_since_last_motion = 0
                motion_detected_once = True
                if frame_counter % 10 == 0:
                    logger.info(f"Motion detected @{bounding_boxes}")
            else:
                frames_since_last_motion += 1

            # We write if:
            # We have detected motion for sufficient consecutive frames
            # or we recently detected motion
            if (motion_detected_once and frames_since_last_motion < stop_recording_after) or motion_detected:
                annotation = {"bounding_boxes": bounding_boxes} if motion_detected else None
                for t in targets:
                    t.write(frame, annotation)

            # After sufficient inactivity, close targets
            if frames_since_last_motion > stop_recording_after:
                for t in targets:
                    t.close()

            frame_counter += 1

    except Exception as e:
        print("s interrupted.")
        print(f"Error: {e}")

    finally:
        cap.release()
        for t in targets:
            t.close()
        print("Streaming stopped.")


if __name__ == "__main__":
    # source = FileSource("data/video/evaluation/cat/cat7.mp4")
    width_px = 640
    height_px = 480
    fps = 30
    label_video = False
    min_consecutive_motion_frames = 10
    stop_recording_after = fps * 10

    source = CameraSource(
        device_name="USB 2.0 Camera", video_format="MJPG", width_px=width_px, height_px=height_px, framerate=fps
    )

    targets = [
        # VideoStreamer(width_px=width_px, height_px=height_px, fps=fps),
        VideoLogger(fps=fps, width_px=width_px, height_px=height_px),
    ]
    main(
        source,
        targets,
        stop_recording_after=stop_recording_after,
        min_consecutive_motion_frames=min_consecutive_motion_frames,
        label_video=label_video,
    )
