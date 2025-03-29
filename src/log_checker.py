import json
import os
import re


def list_detected_cats(log_dir: str, required_true_count: int = 1):
    # Regex pattern to match frame lines
    frame_pattern = re.compile(r"Frame (\d+): (\{.*\})")

    # Traverse the directory tree
    for root, _, files in os.walk(log_dir):
        for log_filename in sorted(files):
            if not log_filename.endswith(".log"):
                continue  # Skip non-log files

            log_filepath = os.path.join(root, log_filename)
            video_filepath = log_filepath.replace(".log", ".mp4")

            with open(log_filepath, "r") as log_file:
                for line in log_file:
                    match = frame_pattern.search(line)
                    if not match:
                        raise Exception(f"Cannot parse line:\n{line}")

                    frame_number, json_data = match.groups()
                    parsed_data = json.loads(json_data)
                    if "is_cat" in parsed_data and parsed_data["is_cat"].count(True) == required_true_count:
                        print(f"Log: {log_filepath}, Frame: {frame_number}, Video: {video_filepath}")


if __name__ == "__main__":
    print("1/3 match cats:")
    list_detected_cats("data/log", required_true_count=1)
    print("2/3 match cats:")
    list_detected_cats("data/log", required_true_count=2)
    print("3/3 match cats:")
    list_detected_cats("data/log", required_true_count=3)
