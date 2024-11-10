import threading
from typing import Optional

import sounddevice as sd
import soundfile as sf


def play_thread(file_name: str, device_id: Optional[int] = None) -> None:
    """Play a .wav file on the selected audio device."""
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


def play(file_name: str, device_id: Optional[int] = None) -> None:
    """Play audio in a thread."""
    playback_thread = threading.Thread(target=play_thread, args=(file_name, device_id))
    playback_thread.start()

    playback_thread.join()
