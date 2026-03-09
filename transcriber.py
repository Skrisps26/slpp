"""
Transcriber
Uses OpenAI Whisper (local) if installed, otherwise prompts for manual transcript entry.
"""

import os
import sys
from pathlib import Path


def transcribe(audio_path: str, progress_callback=None) -> tuple[str, str]:
    """
    Attempt Whisper transcription first.
    Returns (transcript_text, method_used)
    """
    whisper_result = _try_whisper(audio_path, progress_callback)
    if whisper_result:
        return whisper_result, "whisper"

    # Fallback: manual entry
    return _manual_entry(audio_path), "manual"


def _try_whisper(audio_path: str, progress_callback=None) -> str | None:
    try:
        import whisper

        if progress_callback:
            progress_callback("Loading Whisper model (base)...")
        model = whisper.load_model("base")
        if progress_callback:
            progress_callback("Transcribing audio...")
        result = model.transcribe(audio_path, language="en", fp16=False)
        return result["text"].strip()
    except ImportError:
        return None
    except Exception as e:
        print(f"  [Whisper error: {e}]")
        return None


def _manual_entry(audio_path: str) -> str:
    print()
    print("=" * 60)
    print("  WHISPER NOT INSTALLED — MANUAL TRANSCRIPT ENTRY")
    print("=" * 60)
    print(f"  Audio saved to: {audio_path}")
    print()
    print("  Options:")
    print("  1. Type or paste the transcript below")
    print("  2. Install Whisper for automatic transcription:")
    print("       pip install openai-whisper")
    print()
    print("  Paste transcript (press Enter twice on an empty line when done):")
    print("-" * 60)

    lines = []
    blank_count = 0
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            blank_count += 1
            if blank_count >= 2:
                break
            lines.append(line)
        else:
            blank_count = 0
            lines.append(line)

    transcript = "\n".join(lines).strip()
    if not transcript:
        transcript = "[No transcript provided. Report generated from patient information and pre-existing data only.]"
    return transcript


def load_transcript_from_file(path: str) -> str:
    """Load a transcript from a .txt file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Transcript file not found: {path}")
    return p.read_text(encoding="utf-8").strip()
