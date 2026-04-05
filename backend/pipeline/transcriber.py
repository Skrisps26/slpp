"""
Audio transcriber using OpenAI Whisper base model.
Writes audio bytes to tempfile (uuid-named) because Whisper requires a file path.
"""
import whisper
import uuid
import os
import tempfile


class Transcriber:
    """Wraps Whisper transcription for audio files and byte streams."""

    def __init__(self):
        self.model = None

    def load(self):
        """Lazy-load the Whisper model."""
        self.model = whisper.load_model("base")

    def transcribe_bytes(self, audio_bytes: bytes) -> str:
        """
        Transcribe audio from a bytes object.
        Whisper requires a file path — write temp file, transcribe, always delete.
        Uses uuid4() for the temp filename to avoid collisions.
        """
        tmp_path = os.path.join(
            tempfile.gettempdir(), f"gcis_{uuid.uuid4().hex}.wav"
        )
        try:
            with open(tmp_path, "wb") as f:
                f.write(audio_bytes)
            result = self.model.transcribe(tmp_path, language="en")
            return result["text"].strip()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def transcribe_file(self, file_path: str) -> str:
        """Transcribe an audio file directly."""
        result = self.model.transcribe(file_path, language="en")
        return result["text"].strip()
