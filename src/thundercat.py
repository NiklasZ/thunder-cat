import glob
import os
import random
import subprocess
import time

import cv2 as cv
from cv2.typing import MatLike

from binary_classifier import binary_classify_cat_vs_other
from configuration import (
    SOUND_FOLDER,
    CameraSource,
    FileSource,
    VideoTarget,
    configure_logger,
    configure_video_source,
)
from imagenet_classifier import classify_cat_multiclass
from libaudio import get_device_id_matching, is_playing, play_sound
from src.motion_detection import ClusterBoundingBox, detect_motion
from video_logger import VideoLogger

logger = configure_logger()


thundercat_config = {
    "width_px": 640,
    "height_px": 480,
    "fps": 30,
    "label_video": False,
    "min_consecutive_motion_frames": 5,
    "stop_recording_after": 30 * 5,
    "classifier_frame_buffer_size": 30,
    "sound_device_name": "T60",
    "sound_device_volume": 100,  # from 0 to 100%
    "binary_classifier_path": "model/2024_12_08-15_39_25_binary_cat_classifier.pkl",
}

background_subtractor_config = {
    "history": 500,
    "varThreshold": 50,
    "detectShadows": True,
}

clustering_config = {
    "min_initial_cluster_size": 3,
    "cluster_distance_threshold": 50,
    "pad_bounding_box_px": 5,
    "min_final_cluster_size": 30,
    "min_final_cluster_density": 0.002,
    "min_final_bounding_box_length": 60,
}

motion_detection_config = {
    "mask_pixels_below": 127,
    "draw_bounding_boxes": thundercat_config["label_video"],
    "sufficient_motion_thresh": 50,
    "show_background_sub_output": False,
    "clustering_config": clustering_config,
}

classifier_config = {
    "save_classifier_frames": False,
}


def thundercat(
    source: CameraSource | FileSource,
    targets: list[VideoTarget],
    min_consecutive_motion_frames: int,
    stop_recording_after: int,
    background_subtractor_kwargs: dict,
    motion_detection_config: dict,
    classifier_frame_buffer_size: int,
    classifier_config: dict,
    binary_classifier_path: str,
    sound_device_name: str,
    sound_device_volume: int,
):
    cap = configure_video_source(source)

    start_time = time.time()

    # Check source
    if not cap.isOpened():
        print("Cannot open video source")
        exit()

    try:
        back_sub = cv.createBackgroundSubtractorMOG2(**background_subtractor_kwargs)
        frame_counter = 0
        classifications_counter = 0
        frames_since_last_motion = 0
        consecutive_change_frames = 0
        initial_motion_detected = False
        current_time = time.time()
        classifier_frames: list[tuple[MatLike, list[ClusterBoundingBox]]] = []

        # Configure audio device
        sound_device_id = get_device_id_matching(sound_device_name)
        subprocess.run(f"amixer set Master {sound_device_volume}%", shell=True, capture_output=True)

        logger.info("Started video source reading")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_counter % 1000 == 0 and frame_counter != 0:
                new_time = time.time()
                fps_estimate = 1000 / (new_time - current_time)
                logger.info(f"FPS: {fps_estimate:.2f}")
                current_time = new_time

            logger.debug(f"Frame {frame_counter}")
            frame, bounding_boxes = detect_motion(frame, back_sub, **motion_detection_config)
            change_detected = bool(len(bounding_boxes))

            if change_detected:
                consecutive_change_frames += 1
                frames_since_last_motion = 0
            else:
                consecutive_change_frames = 0
                frames_since_last_motion += 1

            initial_motion_detected = (
                initial_motion_detected or consecutive_change_frames > min_consecutive_motion_frames
            )

            if initial_motion_detected:
                if frame_counter % 10 == 0:
                    logger.debug(f"Motion detected @{bounding_boxes}")

            # We write if:
            # We have detected motion for sufficient consecutive frames
            # or we recently detected motion
            if initial_motion_detected and frames_since_last_motion < stop_recording_after:
                annotation = {}

                # Accumulate classification frames
                if len(bounding_boxes):
                    classifier_frames.append((frame, bounding_boxes))
                    annotation["bounding_boxes"] = str(bounding_boxes)

                # After enough frames are gathered, classify
                if len(classifier_frames) == classifier_frame_buffer_size:
                    cls_result = classify_cat_multiclass(
                        classifier_frames, classifications_counter, **classifier_config
                    )
                    is_cat = binary_classify_cat_vs_other(cls_result.class_scores, binary_classifier_path)

                    if is_cat:
                        logger.info("Cat detected!")
                        if not is_playing():
                            # TODO gradually increase sound for consecutive calls
                            sound_files = sorted(glob.glob(os.path.join(SOUND_FOLDER, "*.wav")))
                            idx = random.randint(0, len(sound_files) - 1)
                            logger.info(f"Playing sound file {sound_files[idx]}")
                            play_sound(sound_files[idx], sound_device_id)

                    classifier_frames = []
                    classifications_counter += 1

                    annotation |= {
                        "is_cat": is_cat,
                        "cat_rankings": cls_result.cat_rankings,
                        "top_20_classes": cls_result.top_20_classes,
                        "class_logits": cls_result.class_scores.tolist(),
                    }

                for t in targets:
                    t.write(frame, annotation)

            # After sufficient inactivity, close targets
            if frames_since_last_motion > stop_recording_after:
                for t in targets:
                    t.close()
                initial_motion_detected = False

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
    # source = FileSource("data/video/evaluation/other/2024_12_04-07_11_14.mp4")

    # Peeps 2
    # source = FileSource("data/log/batch_2/2024_11_25-19_46_09.mp4")

    # Noise
    # source = FileSource("data/video/evaluation/other/2024_11_25-16_45_09.mp4")

    source = CameraSource(
        device_name="USB 2.0 Camera",
        video_format="MJPG",
        width_px=thundercat_config["width_px"],
        height_px=thundercat_config["height_px"],
        framerate=thundercat_config["fps"],
    )

    targets = [
        # VideoStreamer(width_px=width_px, height_px=height_px, fps=fps),
        VideoLogger(
            fps=thundercat_config["fps"],
            width_px=thundercat_config["width_px"],
            height_px=thundercat_config["height_px"],
        ),
    ]
    thundercat(
        source,
        targets,
        stop_recording_after=thundercat_config["stop_recording_after"],
        min_consecutive_motion_frames=thundercat_config["min_consecutive_motion_frames"],
        background_subtractor_kwargs=background_subtractor_config,
        motion_detection_config=motion_detection_config,
        classifier_frame_buffer_size=thundercat_config["classifier_frame_buffer_size"],
        classifier_config=classifier_config,
        binary_classifier_path=thundercat_config["binary_classifier_path"],
        sound_device_name=thundercat_config["sound_device_name"],
        sound_device_volume=thundercat_config["sound_device_volume"],
    )
