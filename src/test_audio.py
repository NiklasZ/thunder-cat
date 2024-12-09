# Play audio on a specific device by ID (e.g., device 1)
import time

from libaudio import is_playing, play_sound

file_path = "data/sound/zoidberg.wav"
while True:
    if not is_playing():
        print("Attempting to play.")
        play_sound(file_path, device_id=1)
    time.sleep(2)
