import glob
import os
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
from cv2.typing import MatLike
from tqdm import tqdm

from binary_classifier import train_binary_classifier
from configuration import (
    JUNK_LABEL,
    VERIFIED_LABEL,
    FileSource,
    VideoTarget,
    cat_cls_str,
    cls_str,
    configure_logger,
    filter_junk,
)
from imagenet_classes import CAT_CLASSES, MAX_CLASS_RANK
from thundercat import (
    background_subtractor_config,
    classifier_config,
    motion_detection_config,
    thundercat,
    thundercat_config,
)

logger = configure_logger()
# Columns for all dataframes
column_dtypes = (
    {
        "file_name": pd.StringDtype(),  # source video
        "frame_idx": pd.Int64Dtype(),  # frame idx from video
        "model_name": pd.StringDtype(),  # name of imagenet model
    }
    | {cat_cls_str(c): pd.Int64Dtype() for c in CAT_CLASSES}  # Cat class rankings
    | {  # Label denoting whether the entry is usable for training (mostly generated, some manual)
        "label": pd.StringDtype()
    }
    | {cls_str(idx): pd.Float32Dtype() for idx in range(MAX_CLASS_RANK)}  # class scores from imagenet
)


def save_csv(df: pd.DataFrame, csv_path: str):
    """
    Safely save a DataFrame to a CSV fFile, ensuring the original file is only replaced
    if the new file is written successfully.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        csv_path (str): The path to the CSV file.
        column_dtypes (dict): Dictionary of column data types for conversion.
    """
    df = df.astype(column_dtypes)

    # Use a temporary file to write the data
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", dir=os.path.dirname(csv_path), suffix=".csv"
        ) as temp_file:
            df.to_csv(temp_file.name, index=False)
        # Replace the original file with the temporary file
        os.replace(temp_file.name, csv_path)
    finally:
        # Clean up temporary file if it still exists
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


class VideoStatGatherer(VideoTarget):
    def __init__(self):
        # {frame_idx: {cat_class: rank}}
        self.cat_rankings = {}
        self.class_logits = {}
        self.model_name = None

    def write(self, frame: MatLike, annotations: dict | None = None):
        if annotations and "cat_rankings" in annotations:
            self.model_name = annotations["model_name"]
            for i in range(len(annotations["cat_rankings"])):
                f_idx = annotations["frame_indices"][i]
                self.cat_rankings[f_idx] = annotations["cat_rankings"][i]
                self.class_logits[f_idx] = annotations["class_logits"][i]

    def close(self):
        return


# TODO plot ROC curve


def get_video_stats(video_dir: str, video_folder: str, analysis_dir: str) -> pd.DataFrame:
    """Process all videos in the provided directory and write their classification data into
        a dataframe.
    Args:
        video_dir (str): source video directory
        video_folder (str): source video folder
        analysis_dir (str): directory to write analysis results to.

    Returns:
        pd.DataFrame: dataframe of video classifications.
    """

    csv_dir = os.path.join(analysis_dir, video_folder)
    csv_path = os.path.join(csv_dir, "analysis.csv")

    # Load existing file to skip re-analysing older videos
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, dtype=column_dtypes, delimiter=",", skipinitialspace=True)
        df.columns = df.columns.str.strip()
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    else:
        column_config = {col: pd.Series(dtype=dtype) for col, dtype in column_dtypes.items()}
        df = pd.DataFrame(column_config)

    os.makedirs(csv_dir, exist_ok=True)

    mp4_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    logger.info(f"Found {len(mp4_paths)} video files in {video_dir}")
    for idx, p in tqdm(list(enumerate(mp4_paths))):
        logger.debug(f"Processing {idx+1}/{len(mp4_paths)}...")
        # Skip any already processed files
        file_name = os.path.basename(p)
        if (df["file_name"] == file_name).any():
            continue

        vsg = VideoStatGatherer()
        source = FileSource(p)
        file_name_wo_ext = file_name.split(".")[0]
        # We tweak the imagenet classifier to output more information for us
        updated_classifier_config = classifier_config | {
            "classifier_frame_save_folder": csv_dir,
            "classifier_frame_prefix": file_name_wo_ext,
            "save_classifier_frames": True,
        }
        # Stream and classify video
        thundercat(
            source,
            [vsg],
            stop_recording_after=thundercat_config["stop_recording_after"],
            min_consecutive_motion_frames=thundercat_config["min_consecutive_motion_frames"],
            background_subtractor_kwargs=background_subtractor_config,
            motion_detection_config=motion_detection_config,
            classifier_frame_buffer_size=thundercat_config["classifier_frame_buffer_size"],
            classifier_config=updated_classifier_config,
            binary_classifier_path=thundercat_config["binary_classifier_path"],
            sound_device_name=thundercat_config["sound_device_name"],
            sound_device_volume=thundercat_config["sound_device_volume"],
            imagenet_classifier_name=thundercat_config["imagenet_classifier_name"],
            buffer_size=thundercat_config["buffer_size"],
            toggle_sound=False,
        )

        for (frame_idx, class_rankings), class_logits in zip(vsg.cat_rankings.items(), vsg.class_logits.values()):
            row = (
                {
                    "file_name": file_name,
                    "frame_idx": frame_idx,
                    "model_name": vsg.model_name,
                }
                | {cat_cls_str(c): r for c, r in class_rankings.items()}
                | {cls_str(i): l for i, l in enumerate(class_logits)}
            )
            df.loc[len(df)] = row

        # If there was nothing to classify for the video, add an empty row
        if not len(vsg.cat_rankings):
            df.loc[len(df)] = {"file_name": file_name}

        save_csv(df, csv_path)
    return df


def plot_class_histograms(
    day_cat_df: pd.DataFrame, night_cat_df: pd.DataFrame, other_df: pd.DataFrame, analysis_dir: str
):
    """Plots histograms of the cat classes and compares how different classes rank according to imagenet.

    Args:
        day_cat_df (pd.DataFrame): daytime cat data
        night_cat_df (pd.DataFrame): nighttime cat data
        other_df (pd.DataFrame): non-cat data
        analysis_dir (str): dir for analysis results
    """

    f_night_cat_df = filter_junk(night_cat_df)
    f_day_cat_df = filter_junk(day_cat_df)
    f_other_df = filter_junk(other_df)

    for c in CAT_CLASSES:
        model_name = f_day_cat_df["model_name"].iloc[0]
        plt.hist(f_night_cat_df[cat_cls_str(c)], bins=200, alpha=0.5, label="night_cat", color="blue")
        plt.hist(f_day_cat_df[cat_cls_str(c)], bins=200, alpha=0.5, label="day_cat", color="orange")
        plt.hist(f_other_df[cat_cls_str(c)], bins=200, alpha=0.5, label="other", color="purple")

        plt.xlabel("Rank")
        plt.ylabel("Counts")
        plt.title(f"Classification Rankings for Cat Class {c} - {model_name}")
        plt.legend()

        # Save the plot as an image file
        file_path = os.path.join(analysis_dir, f"{cat_cls_str(c)}_ranks.png")
        plt.savefig(file_path, dpi=600, bbox_inches="tight")
        plt.clf()


def label_deviant_cat_entries(df: pd.DataFrame, class_id: int, class_threshold: int, cat_median: float):
    """Labels cat footage entries that deviate too much from the median. This is mostly to flag
    possibly low quality samples that can make the training data worse.

    Args:
        df (pd.DataFrame): cat dataframe
        class_id (int): cat class id to check against
        class_threshold (int): threshold to be considered an outlier
        cat_median (float): calculated median rank for that cat class id.
    """
    col_name = cat_cls_str(class_id)
    col_label = "label"

    def label_row(row):
        if pd.isna(row[col_label]) or row[col_label] not in [JUNK_LABEL, VERIFIED_LABEL]:
            deviation = abs(row[col_name] - cat_median)
            # Mark outliers for manual expection
            return "typical" if deviation < class_threshold else "outlier"
        else:
            return row[col_label]

    df[col_label] = df.apply(label_row, axis=1)


def label_proximal_other_entries(df: pd.DataFrame, class_id: int, class_threshold: int, cat_median: float):
    """Labels non-cat footage entries that is too close to the cat class median. This is mostly to flag
    footage that may accidentally contain cats.

    Args:
        df (pd.DataFrame): non-cat dataframe
        class_id (int): cat class id to check against
        class_threshold (int): threshold to be considered proximal
        cat_median (float): calculated median rank for that cat class id.
    """
    col_name = cat_cls_str(class_id)
    col_label = "label"

    def label_row(row):
        if pd.isna(row[col_label]) or row[col_label] not in [JUNK_LABEL, VERIFIED_LABEL]:
            deviation = abs(row[col_name] - cat_median)
            # Mark proximal classifcations for manual expection
            return "proximal" if deviation < class_threshold else "different"
        else:
            return row[col_label]

    df[col_label] = df.apply(label_row, axis=1)


def classify_videos(
    day_cat_folder: str,
    day_cat_dir: str,
    night_cat_folder: str,
    night_cat_dir: str,
    other_folder: str,
    other_dir: str,
    analysis_dir: str,
    false_pos_weight: float,
    false_neg_weight: float,
):
    """Analyse videos of cats and non-cats. Identify outliers. Saves plots of
    cat class distributions. Trains binary cat vs non-cat classifier and outputs
    performance.
    Args:
        day_cat_folder (str): folder of day cat videos
        day_cat_dir (str): dir of day cat videos
        night_cat_folder (str): folder of night cat videos
        night_cat_dir (str): dir of night cat videos
        other_folder (str): folder of non-cat videos
        other_dir (str): dir of non-cat videos
        analysis_dir (str): directory of analysis output
        false_pos_weight (float): penalty for false positives
        false_neg_weight (float): penalty for false negatives
    """
    day_cat_df = get_video_stats(day_cat_dir, day_cat_folder, analysis_dir)
    night_cat_df = get_video_stats(night_cat_dir, night_cat_folder, analysis_dir)
    other_df = get_video_stats(other_dir, other_folder, analysis_dir)

    logger.info("Labelling outliers in captured frames...")

    proximity_threshold = 20
    # Deviations in 281, 285 classifications of day cat footage
    day_cat_median_281 = day_cat_df[cat_cls_str(281)].median()
    label_deviant_cat_entries(day_cat_df, 281, proximity_threshold, day_cat_median_281)
    day_cat_median_285 = day_cat_df[cat_cls_str(285)].median()
    label_deviant_cat_entries(day_cat_df, 285, proximity_threshold, day_cat_median_285)

    # Deviations in 281, 285 classifications of night cat footage
    night_cat_median_281 = night_cat_df[cat_cls_str(281)].median()
    label_deviant_cat_entries(night_cat_df, 281, proximity_threshold, night_cat_median_281)
    night_cat_median_285 = night_cat_df[cat_cls_str(285)].median()
    label_deviant_cat_entries(night_cat_df, 285, proximity_threshold, night_cat_median_285)

    combined_cat_df = pd.concat((day_cat_df, night_cat_df))
    combined_cat_median_281 = combined_cat_df[cat_cls_str(281)].median()
    combined_cat_median_285 = combined_cat_df[cat_cls_str(285)].median()
    label_proximal_other_entries(other_df, 281, proximity_threshold, combined_cat_median_281)
    label_proximal_other_entries(other_df, 285, proximity_threshold, combined_cat_median_285)

    csv_path = os.path.join(analysis_dir, day_cat_folder, "analysis.csv")
    save_csv(day_cat_df, csv_path)

    csv_path = os.path.join(analysis_dir, night_cat_folder, "analysis.csv")
    save_csv(night_cat_df, csv_path)

    csv_path = os.path.join(analysis_dir, other_folder, "analysis.csv")
    save_csv(other_df, csv_path)

    plot_class_histograms(day_cat_df, night_cat_df, other_df, analysis_dir)

    day_cat_outliers = (day_cat_df["label"] == "outlier").sum()
    night_cat_outliers = (night_cat_df["label"] == "outlier").sum()
    other_proximals = (other_df["label"] == "proximal").sum()

    if day_cat_outliers or night_cat_outliers or other_proximals:
        logger.info("Detected undetermined labels")
        logger.info(f"{day_cat_outliers} outliers in day_cat_df")
        logger.info(f"{night_cat_outliers} outliers in night_cat_df")
        logger.info(f"{other_proximals} proximals in other_df")
    else:
        logger.info("Training binary classifier...")
        train_binary_classifier(combined_cat_df, other_df, false_pos_weight, false_neg_weight)


if __name__ == "__main__":

    analysis_dir = os.path.join("data", "analysis")
    video_dir = os.path.join("data", "video", "evaluation")

    day_cat_folder = "day_cat"
    day_cat_dir = os.path.join(video_dir, "cat", day_cat_folder)
    night_cat_folder = "night_cat"
    night_cat_dir = os.path.join(video_dir, "cat", night_cat_folder)
    other_folder = "other"
    other_dir = os.path.join(video_dir, other_folder)

    # Training parameters
    false_pos_weight = 25
    false_neg_weight = 1

    classify_videos(
        day_cat_folder,
        day_cat_dir,
        night_cat_folder,
        night_cat_dir,
        other_folder,
        other_dir,
        analysis_dir,
        false_pos_weight,
        false_neg_weight,
    )
