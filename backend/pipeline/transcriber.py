"""
Audio transcriber using OpenAI Whisper base model.
Runs on CPU, ~74MB model.
Converts non-wav files to wav via ffmpeg before transcription.
"""
import os
import shutil
import subprocess
import tempfile
from typing import Optional


class Transcriber:
    """Wraps Whisper transcription for audio files and byte streams."""

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self.model = None

    def _load(self):
        """Lazy-load the Whisper model."""
        if self.model is None:
            import whisper
            self.model = whisper.load_model(self.model_name)
            print(f"[Transcriber] Loaded whisper-{self.model_name} on CPU.")

    def _ensure_wav(self, audio_bytes: bytes, ext: str = ".wav") -> str:
        """Convert audio bytes to a temporary .wav file if needed."""
        # Always write wav to ensure whisper compatibility
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        wav_path = tmp.name

        # If already wav, just write directly
        if ext == ".wav":
            with open(wav_path, "wb") as f:
                f.write(audio_bytes)
            return wav_path

        # For non-wav, write to temp file then convert with ffmpeg
        if shutil.which("ffmpeg"):
            src = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            src.write(audio_bytes)
            src.close()
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", src.name,
                        "-acodec", "pcm_s16le", "-ar", "16000",
                        "-ac", "1", wav_path
                    ],
                    check=True,
                    capture_output=True,
                    timeout=120,
                )
                return wav_path
            finally:
                os.unlink(src.name)
        else:
            # No ffmpeg - write whatever we have (whisper may still handle it)
            with open(wav_path, "wb") as f:
                f.write(audio_bytes)
            return wav_path

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file to text."""
        self._load()
        result = self.model.transcribe(audio_path, language="en")
        return result.get("text", "").strip()

    def transcribe_bytes(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """Transcribe audio from a bytes object."""
        ext = os.path.splitext(filename)[1].lower() or ".wav"
        wav_path = self._ensure_wav(audio_bytes, ext)
        try:
            return self.transcribe(wav_path)
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)

    def transcribe_with_timestamps(self, audio_path: str) -> dict:
        """Transcribe with word-level timestamps."""
        self._load()
        result = self.model.transcribe(audio_path, language="en", word_timestamps=True)
        return {
            "text": result.get("text", "").strip(),
            "segments": result.get("segments", []),
        }
