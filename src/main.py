import time

import cv2 as cv
from cv2.typing import MatLike

from classifier import classify_cat
from configuration import (
    CameraSource,
    FileSource,
    VideoTarget,
    configure_logger,
    configure_video_source,
)
from src.motion_detection import ClusterBoundingBox, detect_motion
from video_logger import VideoLogger

logger = configure_logger()


def main(
    source: CameraSource | FileSource,
    targets: list[VideoTarget],
    label_video: bool,
    min_consecutive_motion_frames: int,
    stop_recording_after: int,
    background_subtractor_kwargs: dict,
    motion_detection_config: dict,
    classifier_frame_buffer_size: int,
    classifier_config: dict,
):
    cap = configure_video_source(source)

    start_time = time.time()

    # Check source
    if not cap.isOpened():
        print("Cannot open video source")
        exit()

    try:
        # TODO move all the extra tooling away to its own method at some point
        back_sub = cv.createBackgroundSubtractorMOG2(**background_subtractor_kwargs)
        frame_counter = 0
        frames_since_last_motion = 0
        consecutive_change_frames = 0
        motion_detected_once = False
        classifier_frames: list[tuple[MatLike, list[ClusterBoundingBox]]] = []
        logger.info("Started video source reading")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            logger.debug(f"Frame {frame_counter}")
            frame, bounding_boxes = detect_motion(frame, back_sub, **motion_detection_config)
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
                    logger.debug(f"Motion detected @{bounding_boxes}")
            else:
                frames_since_last_motion += 1

            # We write if:
            # We have detected motion for sufficient consecutive frames
            # or we recently detected motion
            if (motion_detected_once and frames_since_last_motion < stop_recording_after) or motion_detected:

                if len(bounding_boxes):
                    classifier_frames.append((frame, bounding_boxes))

                if len(classifier_frames) == classifier_frame_buffer_size:
                    # TODO use classification result
                    classify_cat(classifier_frames, **classifier_config)
                    classifier_frames = []

                annotation = {"bounding_boxes": str(bounding_boxes)} if motion_detected else None
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

    print(f"Ran for {(time.time() - start_time):.2f}s")


if __name__ == "__main__":
    # Day cat (old)
    # source = FileSource("data/video/evaluation/cat/day_cat/2024_11_17-16_42_47.mp4")

    # Day cat (new)
    # source = FileSource("data/video/evaluation/cat/day_cat/2024_11_25-15_07_05.mp4")

    # Day cat (3)
    # source = FileSource("data/video/evaluation/cat/day_cat/2024_11_25-15_58_50.mp4")

    # Night cat (new)
    # source = FileSource("data/video/evaluation/cat/night_cat/2024_11_24-22_16_15.mp4")

    # Peeps
    source = FileSource("data/video/evaluation/not_cat/2024_11_17-13_22_10.mp4")

    # Noise
    # source = FileSource("data/video/evaluation/garbage/2024_11_17-13_23_15.mp4")

    width_px = 640
    height_px = 480
    fps = 30
    label_video = False
    min_consecutive_motion_frames = 7
    stop_recording_after = fps * 10
    classifier_frame_buffer_size = 30

    # source = CameraSource(
    #     device_name="USB 2.0 Camera", video_format="MJPG", width_px=width_px, height_px=height_px, framerate=fps
    # )

    background_subtractor_config = {
        "history": 500,
        "varThreshold": 50,
        "detectShadows": True,
    }

    clustering_config = {
        "min_initial_cluster_size": 3,
        "cluster_distance_threshold": 50,
        "min_final_cluster_size": 30,
        "min_final_cluster_density": 0.002,
    }

    motion_detection_config = {
        "mask_pixels_below": 127,
        "draw_bounding_boxes": label_video,
        "sufficient_motion_thresh": 50,
        "show_background_sub_output": False,
        "clustering_config": clustering_config,
    }

    classifier_config = {
        "save_classifier_frames": True,
    }

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
        background_subtractor_kwargs=background_subtractor_config,
        motion_detection_config=motion_detection_config,
        classifier_frame_buffer_size=classifier_frame_buffer_size,
        classifier_config=classifier_config,
    )
