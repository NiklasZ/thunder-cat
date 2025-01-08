# Note: This script is used to generate snapshots from the video files.
# The ffprobe is used to get the duration of the video file and get the offsets for the snapshots.
# ffprobe -i day_after_difference_subtraction.mp4 -show_entries format=duration -v quiet -of csv="p=0"

ffmpeg -y -i day_after_example.mp4 -vframes 1 -ss 00:00:01.127 day_after_snapshot.png
ffmpeg -y -i day_after_difference_subtraction.mp4 -vframes 1 -ss 00:00:00.500 day_after_difference_snapshot.png
ffmpeg -y -i day_after_bounding_box.mp4 -vframes 1 -ss 00:00:00.500 day_after_bounding_box_snapshot.png

