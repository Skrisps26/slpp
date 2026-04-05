"""
Temporal expression extractor using HeidelTime (rule-based, no training).
Wraps py-heideltime with graceful fallback if unavailable.
"""
import re
from typing import List, Dict


class TemporalExtractor:
    """Extracts temporal expressions from clinical text."""

    # Fallback regex patterns if py-heideltime is unavailable
    _TEMPORAL_PATTERNS = [
        (r"\d+\s+(days?|weeks?|months?|years?|hours?|minutes?)\s+ago", "DURATION"),
        (r"(yesterday|today|tomorrow|last\s+\w+|next\s+\w+)", "DATE"),
        (r"\d{1,2}/\d{1,2}/\d{2,4}", "DATE"),
        (r"\d{4}-\d{2}-\d{2}", "DATE"),
        (r"(every|each|daily|weekly|monthly)", "FREQUENCY"),
        (r"for\s+\d+\s+(days?|weeks?|months?|years?)", "DURATION"),
    ]

    def __init__(self):
        self.heideltime_available = False
        try:
            from py_heideltime import PyHeidelTime
            self.heideltime = None  # lazy init
            self.heideltime_available = True
        except (ImportError, ModuleNotFoundError):
            self.heideltime_available = False

    def extract(self, text: str) -> list:
        """Extract temporal expressions from text.

        Returns list of dicts:
          {"text": "3 days ago", "type": "DURATION", "normalized": "P3D", "start": 12, "end": 22}
        """
        if self.heideltime_available:
            return self._extract_with_heideltime(text)
        return self._extract_with_regex(text)

    def _extract_with_heideltime(self, text: str) -> list:
        """Extract temporals using py-heideltime."""
        try:
            from py_heideltime import PyHeidelTime
            ht = PyHeidelTime()
            results = ht.extract(text)
            events = []
            for r in results:
                events.append({
                    "text": r.get("text", ""),
                    "type": r.get("type", "DATE"),
                    "normalized": r.get("timex3", r.get("normalized", "")),
                    "start": r.get("start", 0),
                    "end": r.get("end", 0),
                })
            return events
        except Exception:
            return self._extract_with_regex(text)

    def _extract_with_regex(self, text: str) -> list:
        """Fallback regex-based temporal extraction."""
        events = []
        for pattern, event_type in self._TEMPORAL_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Heuristic normalization
                matched_text = match.group(0)
                normalized = self._heuristic_normalize(matched_text, event_type)
                events.append({
                    "text": matched_text,
                    "type": event_type,
                    "normalized": normalized,
                    "start": match.start(),
                    "end": match.end(),
                })
        # Sort by start position
        events.sort(key=lambda e: e["start"])
        return events

    def _heuristic_normalize(self, text: str, event_type: str) -> str:
        """Heuristically normalize a temporal expression."""
        import re as _re
        # Duration patterns
        m = _re.search(r"(\d+)\s+(days?|weeks?|months?|years?|hours?|minutes?)", text, _re.IGNORECASE)
        if m:
            num = int(m.group(1))
            unit = m.group(2).lower()
            if unit.startswith("day"):
                return f"P{num}D"
            elif unit.startswith("week"):
                return f"P{num * 7}D"
            elif unit.startswith("month"):
                return f"P{num * 30}D"
            elif unit.startswith("year"):
                return f"P{num * 365}D"
            elif unit.startswith("hour"):
                return f"PT{num}H"
            elif unit.startswith("minute"):
                return f"PT{num}M"

        import datetime
        today = datetime.date.today()
        text_lower = text.lower()
        if "yesterday" in text_lower:
            return (today - datetime.timedelta(days=1)).isoformat()
        elif "today" in text_lower:
            return today.isoformat()
        elif "tomorrow" in text_lower:
            return (today + datetime.timedelta(days=1)).isoformat()

        # Date patterns
        date_match = _re.match(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", text)
        if date_match:
            m2, d2, y2 = date_match.groups()
            year = int(y2) if int(y2) > 100 else 2000 + int(y2)
            return f"{year:04d}-{int(m2):02d}-{int(d2):02d}"

        return text
