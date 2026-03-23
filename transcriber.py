"""
Transcriber
Uses OpenAI Whisper (local) for transcription and pyannote/speaker-diarization-3.1
for speaker diarization. Produces a labelled Doctor/Patient transcript.

Speaker assignment heuristic:
  - The speaker who talks MORE total seconds = Doctor
  - The speaker who talks LESS total seconds = Patient
  (In a clinical encounter the doctor almost always speaks more.)

If diarization is unavailable the pipeline falls back to plain Whisper,
and if Whisper is also unavailable it falls back to manual entry.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_env():
    """Load .env from project root (best-effort)."""
    try:
        from dotenv import load_dotenv

        _root = Path(__file__).resolve().parent
        env_path = _root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass


def _get_hf_token() -> str | None:
    _load_env()
    return os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transcribe(audio_path: str, progress_callback=None) -> tuple[str, str]:
    """
    Transcribe *audio_path* and return (transcript_text, method_used).

    method_used is one of:
        "whisper+diarization"  – Whisper + pyannote speaker labels
        "whisper"              – plain Whisper, no speaker labels
        "manual"               – user-typed fallback
    """
    # 1. Try diarized transcription (best output)
    diarized = _try_diarized(audio_path, progress_callback)
    if diarized:
        return diarized, "whisper+diarization"

    # 2. Fall back to plain Whisper
    plain = _try_whisper(audio_path, progress_callback)
    if plain:
        return plain, "whisper"

    # 3. Last resort – manual entry
    return _manual_entry(audio_path), "manual"


def load_transcript_from_file(path: str) -> str:
    """Load a transcript from a .txt file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Transcript file not found: {path}")
    return p.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Diarized transcription  (pyannote + Whisper word timestamps)
# ---------------------------------------------------------------------------


def _try_diarized(audio_path: str, progress_callback=None) -> str | None:
    """
    Run pyannote speaker diarization then align Whisper word-level timestamps
    to produce a Doctor / Patient labelled transcript.

    Returns None if pyannote or Whisper are not available, or if the HF
    token is missing.
    """
    hf_token = _get_hf_token()
    if not hf_token:
        if progress_callback:
            progress_callback(
                "No HUGGINGFACE_TOKEN found in .env – skipping diarization."
            )
        return None

    # -- check imports -------------------------------------------------------
    try:
        import torch
        import torchaudio
        from pyannote.audio import Pipeline
    except ImportError:
        if progress_callback:
            progress_callback("pyannote.audio not installed – skipping diarization.")
        return None

    try:
        import whisper
    except ImportError:
        if progress_callback:
            progress_callback("openai-whisper not installed – skipping diarization.")
        return None

    # -- diarization ---------------------------------------------------------
    try:
        if progress_callback:
            progress_callback("Loading pyannote diarization pipeline...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        pipeline.to(device)

        if progress_callback:
            progress_callback("Running speaker diarization...")

        waveform, sample_rate = torchaudio.load(audio_path)
        diarization = pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            min_speakers=2,
            max_speakers=2,  # doctor + patient
        )

        # Build list of (start_sec, end_sec, speaker_label)
        segments: list[tuple[float, float, str]] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))

    except Exception as e:
        if progress_callback:
            progress_callback(
                f"Diarization failed ({e}) – falling back to plain Whisper."
            )
        return None

    if not segments:
        if progress_callback:
            progress_callback(
                "No speaker segments found – falling back to plain Whisper."
            )
        return None

    # -- whisper with word timestamps ----------------------------------------
    try:
        if progress_callback:
            progress_callback("Loading Whisper model (small)...")

        model = whisper.load_model("small")

        if progress_callback:
            progress_callback("Transcribing with word-level timestamps...")

        result = model.transcribe(
            audio_path,
            language="en",
            fp16=False,
            word_timestamps=True,
            initial_prompt=(
                "Medical consultation between a doctor and patient. "
                "The patient reports symptoms such as chest pain, shortness of breath, "
                "headache, nausea, fever, fatigue, and mentions medications and allergies."
            ),
        )
    except Exception as e:
        if progress_callback:
            progress_callback(f"Whisper failed ({e}) – falling back to plain Whisper.")
        return None

    # -- extract word-level timestamps from Whisper output -------------------
    # Each item: {"word": str, "start": float, "end": float}
    words: list[dict] = []
    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            word_text = w.get("word", "").strip()
            w_start = w.get("start")
            w_end = w.get("end")
            if word_text and w_start is not None and w_end is not None:
                words.append({"word": word_text, "start": w_start, "end": w_end})

    if not words:
        # word timestamps unavailable – fall back to segment-level alignment
        words = _words_from_segments(result)

    # -- assign each word to a speaker ---------------------------------------
    def _speaker_at(t: float) -> str | None:
        """Return the pyannote speaker label active at time t."""
        for seg_start, seg_end, spk in segments:
            if seg_start <= t <= seg_end:
                return spk
        # Find nearest segment if t falls in a gap
        best_spk = None
        best_dist = float("inf")
        for seg_start, seg_end, spk in segments:
            dist = min(abs(t - seg_start), abs(t - seg_end))
            if dist < best_dist:
                best_dist = dist
                best_spk = spk
        return best_spk

    # -- compute total speaking time per pyannote speaker label --------------
    speaker_durations: dict[str, float] = {}
    for seg_start, seg_end, spk in segments:
        speaker_durations[spk] = speaker_durations.get(spk, 0.0) + (seg_end - seg_start)

    if len(speaker_durations) < 2:
        # Only one speaker detected – can't distinguish
        if progress_callback:
            progress_callback(
                "Only one speaker detected – falling back to plain Whisper."
            )
        return None

    # Doctor = highest total speaking time
    sorted_speakers = sorted(
        speaker_durations, key=lambda spk: speaker_durations[spk], reverse=True
    )
    doctor_label = sorted_speakers[0]
    patient_label = sorted_speakers[1]

    def _role(spk: str | None) -> str:
        if spk == doctor_label:
            return "Doctor"
        return "Patient"

    # -- group consecutive words by speaker into utterances ------------------
    utterances: list[dict] = []  # {"role": str, "words": list[str]}

    for w in words:
        mid = (w["start"] + w["end"]) / 2.0
        spk = _speaker_at(mid)
        role = _role(spk)

        if utterances and utterances[-1]["role"] == role:
            utterances[-1]["words"].append(w["word"])
        else:
            utterances.append({"role": role, "words": [w["word"]]})

    # -- build final transcript string ---------------------------------------
    lines: list[str] = []
    for utt in utterances:
        text = " ".join(utt["words"]).strip()
        # Clean up spurious leading/trailing punctuation from word tokens
        text = text.strip(" ,;")
        if text:
            lines.append(f"{utt['role']}: {text}")

    transcript = "\n".join(lines).strip()

    if progress_callback:
        progress_callback(
            f"Diarization complete – "
            f"Doctor={doctor_label} ({speaker_durations[doctor_label]:.1f}s), "
            f"Patient={patient_label} ({speaker_durations[patient_label]:.1f}s)"
        )

    return transcript if transcript else None


def _words_from_segments(result: dict) -> list[dict]:
    """
    Fallback: build a word list from Whisper segment-level timestamps
    when word_timestamps are not available.
    Each word in the segment is assigned the segment's time range evenly.
    """
    words = []
    for seg in result.get("segments", []):
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", seg_start)
        seg_text = seg.get("text", "").strip()
        tokens = seg_text.split()
        if not tokens:
            continue
        dur = (seg_end - seg_start) / len(tokens)
        for i, token in enumerate(tokens):
            words.append(
                {
                    "word": token,
                    "start": seg_start + i * dur,
                    "end": seg_start + (i + 1) * dur,
                }
            )
    return words


# ---------------------------------------------------------------------------
# Plain Whisper fallback
# ---------------------------------------------------------------------------


def _try_whisper(audio_path: str, progress_callback=None) -> str | None:
    try:
        import whisper
    except ImportError:
        return None

    try:
        if progress_callback:
            progress_callback("Loading Whisper model (small)...")
        model = whisper.load_model("small")

        if progress_callback:
            progress_callback("Transcribing audio...")

        result = model.transcribe(
            audio_path,
            language="en",
            fp16=False,
            initial_prompt=(
                "Medical consultation between a doctor and patient. "
                "The patient reports symptoms such as chest pain, shortness of breath, "
                "headache, nausea, fever, fatigue, and mentions medications and allergies."
            ),
        )
        return result["text"].strip()

    except Exception as e:
        if progress_callback:
            progress_callback(f"Whisper error: {e}")
        return None


# ---------------------------------------------------------------------------
# Manual entry fallback
# ---------------------------------------------------------------------------


def _manual_entry(audio_path: str) -> str:
    print()
    print("=" * 60)
    print("  WHISPER / DIARIZATION NOT AVAILABLE — MANUAL TRANSCRIPT ENTRY")
    print("=" * 60)
    print(f"  Audio saved to: {audio_path}")
    print()
    print("  Options:")
    print("  1. Type or paste the transcript below")
    print("  2. Install dependencies for automatic transcription:")
    print("       pip install openai-whisper pyannote.audio torch torchaudio")
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
        transcript = (
            "[No transcript provided. "
            "Report generated from patient information and pre-existing data only.]"
        )
    return transcript
