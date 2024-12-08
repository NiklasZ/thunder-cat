# Play audio on a specific device by ID (e.g., device 1)
from libaudio import play, is_playing
import time

file_path = "/home/rae-rae/thunder-cat/sound/zoidberg.wav"
while True:
    if not is_playing():
        print("Attempting to play.")
        play(file_path, device_id=3)
    time.sleep(2)
