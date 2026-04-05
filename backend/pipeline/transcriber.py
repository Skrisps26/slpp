"""
Audio transcriber using OpenAI Whisper base model.
Writes audio bytes to tempfile (uuid-named).
Runs Whisper in a subprocess with CUDA hidden — GPU is exclusively for Ollama.
This is the ONLY reliable way to avoid GPU OOM conflicts.
"""
import os
import sys
import uuid
import tempfile
import subprocess
import textwrap


class Transcriber:
    def __init__(self):
        pass

    def load(self):
        """No-op — Whisper runs in isolated subprocesses with CUDA hidden."""
        print("[Transcriber] Will use subprocess for CPU-only Whisper (GPU reserved for Ollama)")

    def transcribe_bytes(self, audio_bytes: bytes) -> str:
        """Transcribe WAV audio bytes via subprocess (CPU only)."""
        tmp_path = os.path.join(tempfile.gettempdir(), f"gcis_{uuid.uuid4().hex}.wav")
        try:
            with open(tmp_path, "wb") as f:
                f.write(audio_bytes)
            return self._transcribe_path(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def transcribe_file(self, file_path: str) -> str:
        """Transcribe an audio file via subprocess (CPU only)."""
        return self._transcribe_path(file_path)

    def _transcribe_path(self, audio_path: str) -> str:
        """Run whisper in a subprocess with CUDA_VISIBLE_DEVICES='', then return text."""
        python = sys.executable
        # The script MUST hide GPU and use CPU before whisper is imported
        script = textwrap.dedent("""
            import os, sys
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

            import torch
            torch.set_default_device('cpu')

            import whisper
            model = whisper.load_model('base')
            result = model.transcribe(sys.argv[1], language='en')
            print(result['text'].strip(), end='')
        """)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""

        result = subprocess.run(
            [python, "-c", script, audio_path],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Whisper transcription failed (exit {result.returncode}): "
                f"{result.stderr.strip()}"
            )
        text = result.stdout.strip()
        if not text:
            raise RuntimeError("Whisper returned empty transcription.")
        return text
