import json
import os
import re


def list_detected_cats(log_dir: str, required_true_count: int = 1):
    frame_pattern = re.compile(r"Frame (\d+): (\{.*\})")

    results = []

    for root, dirs, files in os.walk(log_dir):
        dirs.sort()
        for log_filename in sorted(files):
            if not log_filename.endswith(".log"):
                continue

            log_filepath = os.path.join(root, log_filename)
            video_filepath = log_filepath.replace(".log", ".mp4")
            matching_frames = []

            with open(log_filepath, "r") as log_file:
                for line in log_file:
                    match = frame_pattern.search(line)
                    if not match:
                        raise Exception(f"{log_filepath}: Cannot parse line:\n{line}")

                    frame_number, json_data = match.groups()
                    parsed_data = json.loads(json_data)
                    if "is_cat" in parsed_data and parsed_data["is_cat"].count(True) == required_true_count:
                        matching_frames.append(int(frame_number))

            if matching_frames:
                first = min(matching_frames)
                last = max(matching_frames)
                results.append((log_filepath, video_filepath, first, last))

    # Pretty-print results
    max_path_len = max((len(path) for path, _, _, _ in results), default=0)
    print(f"{'Log File'.ljust(max_path_len)}  {'Frames'.ljust(11)}  Video File")
    print(f"{'-'*max_path_len}  {'-'*11}  {'-'*40}")
    for log_path, video_path, first, last in results:
        frame_range = f"{first}" if first == last else f"{first}-{last}"
        print(f"{log_path.ljust(max_path_len)}  {frame_range.ljust(11)}  {video_path}")


if __name__ == "__main__":
    print("1/3 match cats:")
    list_detected_cats("data/log", required_true_count=1)
    print("\n2/3 match cats:")
    list_detected_cats("data/log", required_true_count=2)
    print("\n3/3 match cats:")
    list_detected_cats("data/log", required_true_count=3)
