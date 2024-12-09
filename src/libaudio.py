import threading
from typing import Optional

import sounddevice as sd
import soundfile as sf

playing = False
playing_lock = threading.Lock()


def play_thread(file_name: str, device_id: Optional[int] = None) -> None:
    """Play a .wav file on the selected audio device."""
    global playing
    with playing_lock:
        playing = True  # Set playing to True when playback starts

    try:
        data, sample_rate = sf.read(file_name)
        # Device ID = None selects the default device
        device_info = sd.query_devices(device_id)
        device_sample_rate = device_info["default_samplerate"]

        # Resample if needed; may take a long time to complete...
        # if sample_rate != device_sample_rate:
        #    data = scipy.signal.resample(data, int(len(data) * device_sample_rate / sample_rate))
        #    sample_rate = device_sample_rate

        sd.play(data, samplerate=device_sample_rate, device=device_id)
        sd.wait()  # Wait until file finishes playing
    finally:
        with playing_lock:
            playing = False  # Reset playing to False when playback ends


def play_sound(file_name: str, device_id: Optional[int] = None) -> None:
    """Play audio in a thread."""
    playback_thread = threading.Thread(target=play_thread, args=(file_name, device_id))
    playback_thread.start()


def is_playing() -> bool:
    """Check if audio is currently playing."""
    with playing_lock:
        return playing


def get_device_id_matching(name: str) -> int:
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if name in device["name"]:
            return idx

    raise Exception(f"Could not find any audio device matching '{name}'\nFound: {devices}")
