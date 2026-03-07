"""
Audio Recorder
Records microphone input using sounddevice and saves as WAV.
"""

import os
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd

    AUDIO_AVAILABLE = True
except (ImportError, OSError):
    sd = None
    AUDIO_AVAILABLE = False


SAMPLE_RATE = 16000  # 16 kHz – ideal for speech
CHANNELS = 1
DTYPE = "int16"
CHUNK_SIZE = 1024


class AudioRecorder:
    def __init__(self, output_dir: str = "recordings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._frames: list[np.ndarray] = []
        self._recording = False
        self._thread: threading.Thread | None = None
        self.output_path: str = ""

    def _callback(self, indata, frames, time_info, status):
        if self._recording:
            self._frames.append(indata.copy())

    def start(self) -> str:
        if not AUDIO_AVAILABLE:
            raise RuntimeError(
                "Audio recording unavailable: PortAudio library not found. Install portaudio19-dev."
            )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = str(self.output_dir / f"encounter_{timestamp}.wav")
        self._frames = []
        self._recording = True

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=CHUNK_SIZE,
            callback=self._callback,
        )
        self._stream.start()
        return self.output_path

    def stop(self) -> str:
        self._recording = False
        if hasattr(self, "_stream"):
            self._stream.stop()
            self._stream.close()

        if self._frames:
            audio_data = np.concatenate(self._frames, axis=0)
            self._save_wav(audio_data)

        return self.output_path

    def _save_wav(self, audio_data: np.ndarray):
        with wave.open(self.output_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())

    def get_duration(self) -> float:
        if not self._frames:
            return 0.0
        total_samples = sum(f.shape[0] for f in self._frames)
        return total_samples / SAMPLE_RATE

    @staticmethod
    def list_devices():
        if not AUDIO_AVAILABLE:
            return "Audio not available"
        return sd.query_devices()
