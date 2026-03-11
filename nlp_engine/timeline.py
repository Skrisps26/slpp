"""
nlp_engine/timeline.py
Extracts a clinical temporal timeline from the transcript.
"""

import re
from typing import List
from .entities import ClinicalTimeline


# Temporal anchor patterns
_TEMPORAL_PATTERNS = [
    # "two days ago", "three weeks ago", "a year ago"
    (re.compile(
        r"(\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        r"(?:day|days|week|weeks|month|months|year|years)\s+ago)\b", re.I),
     "relative_past"),
    # "since yesterday", "since Monday"
    (re.compile(r"\bsince\s+(yesterday|last\s+\w+|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I),
     "relative_past"),
    # "this morning", "today", "tonight"
    (re.compile(r"\b(this morning|today|tonight|this afternoon|this evening|just now|an hour ago)\b", re.I),
     "today"),
    # "started X ago"
    (re.compile(r"\b(?:started|began|onset|noticed)\s+([^.]{3,60})ago\b", re.I),
     "onset"),
    # "worsened / improved today / this week"
    (re.compile(r"\b(worsened|got worse|improving|better|improving|stable|unchanged)\s+(?:over\s+)?(?:the\s+)?(\w+\s+\w+)\b", re.I),
     "progression"),
    # "for X days/weeks"
    (re.compile(r"\bfor\s+(the\s+(?:past|last)\s+)?(\d+\s*(?:day|days|week|weeks|month|months|year|years)|a\s+(?:day|week|month|year))\b", re.I),
     "duration"),
]

_EVENT_KEYWORDS = [
    (re.compile(r"\b(chest pain|chest pressure|shortness of breath|headache|dizziness|nausea|vomiting|fever|cough)\b", re.I), "symptom"),
    (re.compile(r"\b(took|taking|started|stopped|prescribed|given|increased|decreased)\s+\w+", re.I), "treatment"),
    (re.compile(r"\b(admitted|hospitalized|went to ER|emergency room|surgery|procedure)\b", re.I), "medical_event"),
    (re.compile(r"\b(lab results|blood work|x-ray|CT scan|MRI|ultrasound|ECG|tested positive|tested negative)\b", re.I), "test"),
]


def extract_timeline(text: str) -> ClinicalTimeline:
    """
    Extract temporal clinical events from the transcript.
    Returns a ClinicalTimeline with ordered events.
    """
    timeline = ClinicalTimeline()
    sentences = re.split(r"[.!?\n]", text)

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        time_ref = None
        event_type = "general"

        # Find temporal reference
        for pattern, label in _TEMPORAL_PATTERNS:
            m = pattern.search(sent)
            if m:
                time_ref = m.group(0).strip()
                event_type = label
                break

        # Find event type
        for pattern, etype in _EVENT_KEYWORDS:
            if pattern.search(sent):
                event_type = etype
                break

        if time_ref or any(p.search(sent) for p, _ in _EVENT_KEYWORDS):
            timeline.events.append({
                "time": time_ref or "present",
                "event": sent[:150],
                "type": event_type,
            })

    # Sort: put past events before present
    priority = {"onset": 0, "relative_past": 1, "duration": 2, "progression": 3, "today": 4, "treatment": 5, "general": 6, "test": 7, "medical_event": 8}
    timeline.events.sort(key=lambda e: priority.get(e["type"], 9))

    return timeline
