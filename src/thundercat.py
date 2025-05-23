import glob
import os
import random
import subprocess
import time
import traceback
from collections import deque

import cv2 as cv
from cv2.typing import MatLike

from binary_classifier import (
    binary_classify_cat_vs_other,
    load_binary_classifier,
    load_binary_classifier_config,
)
from configuration import (
    SOUND_FOLDER,
    CameraSource,
    FileSource,
    VideoTarget,
    configure_logger,
    configure_video_source,
)
from frame_buffer import create_frame_buffer_thread
from imagenet_classifier import classify_cat_multiclass, load_imagenet_model
from libaudio import get_device_id_matching, is_playing, play_sound
from src.motion_detection import ClusterBoundingBox, detect_motion
from video_logger import VideoLogger

logger = configure_logger()


thundercat_config = {
    "width_px": 640,
    "height_px": 480,
    "fps": 25,
    "buffer_size": 30 * 60,
    "label_video": False,
    "min_consecutive_motion_frames": 3,
    "stop_recording_after": 30 * 10,
    "classifier_frame_buffer_size": 30,
    "sound_device_name": "T60",
    "initial_sound_device_volume": 70,  # from 0 to 100%
    "imagenet_classifier_name": "mobilenet_v4",
    "binary_classifier_config_path": "model/2024_12_18-07_19_36_parameters.json",
    "binary_classifier_path": "model/2024_12_18-07_19_36_binary_cat_classifier.pkl",
    "toggle_sound": True,
    "max_log_size_gb": 50,
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
    "min_final_cluster_size": 25,
    "min_final_cluster_density": 0.002,
    "min_final_bounding_box_length": 50,
}

motion_detection_config = {
    "mask_pixels_below": 127,
    "draw_bounding_boxes": thundercat_config["label_video"],
    "sufficient_motion_thresh": 40,
    "show_background_sub_output": False,
    "clustering_config": clustering_config,
}

classifier_config = {"save_classifier_frames": False, "num_samples": 3}


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
    binary_classifier_config_path: str,
    imagenet_classifier_name: str,
    sound_device_name: str,
    initial_sound_device_volume: int,
    toggle_sound: bool,
    buffer_size: int,
    fps: int,
):
    cap = configure_video_source(source)
    start_time = time.time()
    stop_event = None
    last_sound_play_time = 0
    current_volume = initial_sound_device_volume

    # Check source
    if not cap.isOpened():
        logger.error("Cannot open video source")
        exit()

    try:
        back_sub = cv.createBackgroundSubtractorMOG2(**background_subtractor_kwargs)
        back_sub_frame_history = deque(maxlen=background_subtractor_kwargs["history"] + min_consecutive_motion_frames)

        # If available, populate the subtractor with existing video frames
        if isinstance(source, FileSource):
            back_sub_history_path = f"{os.path.splitext(source.file_path)[0]}_history.mp4"
            if os.path.exists(back_sub_history_path):
                history_source = FileSource(back_sub_history_path)
                history_cap = configure_video_source(history_source)
                logger.info("Populating subtractor with past file history...")
                while True:
                    ret, frame = history_cap.read()
                    if not ret:
                        break
                    back_sub.apply(frame)
                    back_sub_frame_history.append(frame)

        frame_counter = 0
        recording_frame_counter = 0
        frames_since_last_motion = 0
        consecutive_change_frames = 0
        initial_motion_detected = False
        classifier_frames: list[tuple[int, MatLike, list[ClusterBoundingBox]]] = []
        # a buffer of frames that contain motion before we started recording
        # this way we can write them in later once enough motion is detected.
        initial_motion_frames = []

        # Configure audio device
        sound_device_id = get_device_id_matching(sound_device_name)
        subprocess.run(f"amixer set Master {initial_sound_device_volume}%", shell=True, capture_output=True)

        binary_classifier_config = load_binary_classifier_config(
            binary_classifier_config_path, imagenet_classifier_name
        )
        load_binary_classifier(binary_classifier_path)
        load_imagenet_model(imagenet_classifier_name)
        logger.info("Loaded models")

        logger.info("Started video source reading")
        # Only block the frame buffer if we are reading from a file.
        should_block = isinstance(source, FileSource)
        # Have a thread buffer frames for us as we will have the occasional latency spike when we do
        # expensive opoerations like starting new videos or calling models
        original_fps = cap.get(cv.CAP_PROP_FPS)
        frame_capture_thread, stop_event, no_more_frames__event, frame_buffer = create_frame_buffer_thread(
            cap, buffer_size, should_block, original_fps, fps
        )

        current_time = time.time()
        while not no_more_frames__event.is_set() or frame_buffer.qsize():
            frame = frame_buffer.get(timeout=60)

            if frame_counter % 1000 == 0 and frame_counter != 0:
                new_time = time.time()
                fps_estimate = 1000 / (new_time - current_time)
                logger.info(f"FPS: {fps_estimate:.2f}, Buffer usage: {(frame_buffer.qsize()/buffer_size * 100):.1f}%")
                current_time = new_time

            logger.debug(f"Frame {frame_counter}")
            back_sub_frame_history.append(frame)
            frame, bounding_boxes = detect_motion(frame, back_sub, **motion_detection_config)
            change_detected = bool(len(bounding_boxes))

            if change_detected:
                consecutive_change_frames += 1
                frames_since_last_motion = 0
            else:
                consecutive_change_frames = 0
                frames_since_last_motion += 1

            if consecutive_change_frames > 0 and not initial_motion_detected:
                initial_motion_frames.append(frame)
            else:
                initial_motion_frames = []

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

                # Save the initial state that led to the motion detection
                if len(initial_motion_frames):
                    for idx, f in enumerate(initial_motion_frames[:min_consecutive_motion_frames]):
                        for t in targets:
                            annotation = (
                                {
                                    "Initial Motion Frame": recording_frame_counter,
                                    "back_sub_history": list(back_sub_frame_history)[min_consecutive_motion_frames:],
                                }
                                if idx == 0
                                else {}
                            )
                            t.write(f, annotation)
                            recording_frame_counter += 1

                # Handle current frames
                annotation = {}

                # Accumulate classification frames
                if len(bounding_boxes):
                    classifier_frames.append((recording_frame_counter, frame, bounding_boxes))
                    annotation["bounding_boxes"] = str(bounding_boxes)

                # After enough frames are gathered, classify
                if len(classifier_frames) == classifier_frame_buffer_size:
                    cls_result = classify_cat_multiclass(classifier_frames, **classifier_config)
                    raw_scores = [c.class_scores for c in cls_result]
                    cat_probs = binary_classify_cat_vs_other(raw_scores)
                    threshold = binary_classifier_config["training_parameters"]["classification_threshold"]
                    is_cat = [bool(p >= threshold) for p in cat_probs]

                    # We consider it a cat if all classifications are cat
                    if all(is_cat):
                        logger.debug("Cat detected!")
                        if toggle_sound and not is_playing():
                            current_time = time.time()
                            if current_time - last_sound_play_time <= 15:
                                current_volume = min(current_volume + 10, 100)
                            else:
                                current_volume = initial_sound_device_volume
                            last_sound_play_time = current_time

                            subprocess.run(f"amixer set Master {current_volume}%", shell=True, capture_output=True)
                            sound_files = sorted(glob.glob(os.path.join(SOUND_FOLDER, "*.wav")))
                            idx = random.randint(0, len(sound_files) - 1)
                            logger.info(f"Playing sound file {sound_files[idx]} at volume {current_volume}%")
                            play_sound(sound_files[idx], sound_device_id)
                            annotation["sound_played"] = sound_files[idx]

                    classifier_frames = []

                    annotation |= {
                        "is_cat": is_cat,
                        "cat_probs": cat_probs,
                        "frame_indices": [c.frame_idx for c in cls_result],
                        "cat_rankings": [c.cat_rankings for c in cls_result],
                        "model_name": imagenet_classifier_name,
                        "top_20_classes": [c.top_20_classes for c in cls_result],
                        "class_logits": [c.class_scores.tolist() for c in cls_result],
                    }

                for t in targets:
                    t.write(frame, annotation)
                    recording_frame_counter += 1

            # After sufficient inactivity, close targets
            if frames_since_last_motion > stop_recording_after:
                for t in targets:
                    t.close()
                initial_motion_detected = False
                recording_frame_counter = 0
                classifier_frames = []

            frame_counter += 1

    except Exception as e:
        logger.error("Stream interrupted.")
        logger.error(f"Error: {e}")
        print("Stack trace:")
        traceback.print_exc()

    finally:
        if stop_event is not None:
            stop_event.set()
            frame_capture_thread.join()
        cap.release()
        for t in targets:
            t.close()
        logger.info("Streaming stopped.")

    logger.info(f"Ran for {(time.time() - start_time):.2f}s")


if __name__ == "__main__":
    # Day cat (old)
    #  source = FileSource("data/video/evaluation/cat/day_cat/2024_11_17-16_42_47.mp4")

    # Day cat (new)
    # source = FileSource("data/video/evaluation/cat/day_cat/2024_11_25-15_07_05.mp4")

    # Day cat (3)
    # source = FileSource("data/video/evaluation/cat/day_cat/2024_11_25-15_58_50.mp4")

    # Day cat (4)
    # source = FileSource("data/video/evaluation/cat/day_cat/2024_12_09-08_07_35.mp4")

    # Night cat
    # source = FileSource("data/video/evaluation/cat/night_cat/2024_11_24-22_16_15.mp4")

    # Night cat (new)
    # source = FileSource("data/video/evaluation/cat/night_cat/2024_12_15-23_35_01.mp4")

    # Peeps
    # source = FileSource("data/video/evaluation/other/2024_12_04-07_11_14.mp4")

    # Peeprs 3
    # source = FileSource("data/log/2025_03_30/2025_03_30-08_47_49.mp4")

    # Cat-like peeps
    # source = FileSource("data/video/evaluation/other/2024_11_29-23_17_21.mp4")

    # Noise
    # source = FileSource("data/video/evaluation/other/2024_11_25-16_45_09.mp4")

    # Doc example
    # source = FileSource("doc/day_after_example.mp4")

    source = CameraSource(
        device_name="USB 2.0 Camera",
        video_format="MJPG",
        width_px=thundercat_config["width_px"],
        height_px=thundercat_config["height_px"],
    )

    targets = [
        # VideoStreamer(width_px=width_px, height_px=height_px, fps=fps),
        VideoLogger(
            fps=thundercat_config["fps"],
            width_px=thundercat_config["width_px"],
            height_px=thundercat_config["height_px"],
            max_log_size_gb=thundercat_config["max_log_size_gb"],
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
        imagenet_classifier_name=thundercat_config["imagenet_classifier_name"],
        binary_classifier_path=thundercat_config["binary_classifier_path"],
        binary_classifier_config_path=thundercat_config["binary_classifier_config_path"],
        sound_device_name=thundercat_config["sound_device_name"],
        initial_sound_device_volume=thundercat_config["initial_sound_device_volume"],
        toggle_sound=thundercat_config["toggle_sound"],
        buffer_size=thundercat_config["buffer_size"],
        fps=thundercat_config["fps"],
    )
