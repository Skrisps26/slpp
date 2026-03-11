"""
nlp_engine/symptom_frames.py
Symptom frame extraction: converts clinical sentences into structured
{symptom, severity, duration, location, character, onset, progression} frames.
"""

import re
from typing import Optional


SEVERITY_RE = re.compile(
    r"\b(mild|moderate(?:ly)?|severe(?:ly)?|sharp|dull|aching|crushing|stabbing|burning|throbbing|"
    r"pressure[\-\s]?like|tight(?:ness)?|squeezing|crampy|constant|intermittent|"
    r"(\d+)\s*/?\s*10|(\d+)\s+out\s+of\s+10)\b",
    re.I,
)

DURATION_RE = re.compile(
    r"\bfor\s+(?:the\s+(?:past|last)\s+)?"
    r"((?:about|approximately|around|nearly)?\s*(?:\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven"
    r"|eight|nine|ten|few|several|couple\s+of)\s*"
    r"(?:minute|minutes|hour|hours|day|days|week|weeks|month|months|year|years))\b",
    re.I,
)

ONSET_RE = re.compile(
    r"\b(started|began|onset|noticed|woke up with|first noticed)\s+([^.]{3,60}?)"
    r"(?:ago|yesterday|this\s+\w+|last\s+\w+)?\b",
    re.I,
)

PROGRESSION_RE = re.compile(
    r"\b(getting\s+worse|worsening|worsened|progressing|spreading|improving|getting\s+better|"
    r"resolved|unchanged|stable|constant|comes\s+and\s+goes|intermittent)\b",
    re.I,
)

CHARACTER_RE = re.compile(
    r"\b(sharp|dull|aching|crushing|stabbing|burning|throbbing|pressure|tight|squeezing|"
    r"crampy|colicky|gnawing|tearing|electric|shooting|radiating)\b",
    re.I,
)

FREQUENCY_RE = re.compile(
    r"\b(constant|intermittent|comes\s+and\s+goes|episodic|continuous|every\s+\w+|\d+\s+times?\s+(?:a\s+)?(?:day|week|hour))\b",
    re.I,
)

RADIATION_RE = re.compile(
    r"\b(?:radiates?|spreads?|goes?|going)\s+(?:to|down|into|up|toward)\s+(?:my\s+)?([^.]{3,40})\b",
    re.I,
)


def extract_symptom_frame(symptom_name: str, sentence: str, context: str = "") -> dict:
    """
    Given a symptom and the sentence where it appears, extract a structured frame:
    {severity, duration, character, location, onset, progression, frequency, radiation}
    """
    combined = f"{sentence} {context}"
    frame = {}

    # Severity
    m = SEVERITY_RE.search(combined)
    if m:
        frame["severity"] = m.group(1)

    # Duration
    m = DURATION_RE.search(combined)
    if m:
        frame["duration"] = m.group(1).strip()

    # Character (texture/quality of pain)
    m = CHARACTER_RE.search(combined)
    if m:
        frame["character"] = m.group(1).lower()

    # Progression
    m = PROGRESSION_RE.search(combined)
    if m:
        frame["progression"] = m.group(1).lower()

    # Frequency
    m = FREQUENCY_RE.search(combined)
    if m:
        frame["frequency"] = m.group(1).lower()

    # Radiation
    m = RADIATION_RE.search(combined)
    if m:
        frame["radiation"] = m.group(1).strip()

    # Onset description
    m = ONSET_RE.search(combined)
    if m:
        frame["onset"] = m.group(0).strip()

    return frame


def extract_numeric_severity(text: str) -> Optional[str]:
    """
    Extract numeric pain scale like '7/10' or '7 out of 10'.
    Returns formatted string like '7/10' or None.
    """
    m = re.search(r"\b(\d+)\s*/?\s*10\b", text)
    if m:
        return f"{m.group(1)}/10"
    m = re.search(r"\b(\d+)\s+out\s+of\s+10\b", text, re.I)
    if m:
        return f"{m.group(1)}/10"
    return None
