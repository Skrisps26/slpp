"""
Audio transcriber using OpenAI Whisper base model.
Runs on CPU, ~74MB model.
"""
import os
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

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file to text."""
        self._load()
        result = self.model.transcribe(audio_path, language="en")
        return result.get("text", "").strip()

    def transcribe_bytes(self, audio_bytes: bytes) -> str:
        """Transcribe audio from a bytes object."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            return self.transcribe(tmp_path)
        finally:
            os.unlink(tmp_path)

    def transcribe_with_timestamps(self, audio_path: str) -> dict:
        """Transcribe with word-level timestamps."""
        self._load()
        result = self.model.transcribe(audio_path, language="en", word_timestamps=True)
        return {
            "text": result.get("text", "").strip(),
            "segments": result.get("segments", []),
        }
