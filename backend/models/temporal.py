"""
Temporal expression extractor using HeidelTime (rule-based, no training).
Wraps py-heideltime with graceful fallback if unavailable.
"""
import re
from typing import List, Dict


class TemporalExtractor:
    """Extracts temporal expressions from clinical text."""

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
            self.heideltime_available = True
        except (ImportError, ModuleNotFoundError):
            pass

    def extract(self, text: str) -> list:
        """Extract temporal expressions from text."""
        if self.heideltime_available:
            return self._extract_with_heideltime(text)
        return self._extract_with_regex(text)

    def _extract_with_heideltime(self, text: str) -> list:
        try:
            from py_heideltime import PyHeidelTime
            ht = PyHeidelTime()
            results = ht.extract(text)
            return [
                {
                    "text": r.get("text", ""),
                    "type": r.get("type", "DATE"),
                    "normalized": r.get("timex3", r.get("normalized", "")),
                    "start": r.get("start", 0),
                    "end": r.get("end", 0),
                }
                for r in results
            ]
        except Exception:
            return self._extract_with_regex(text)

    def _extract_with_regex(self, text: str) -> list:
        """Fallback regex-based temporal extraction."""
        events = []
        for pattern, event_type in self._TEMPORAL_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matched_text = match.group(0)
                normalized = self._normalize(matched_text, event_type)
                events.append({
                    "text": matched_text, "type": event_type,
                    "normalized": normalized,
                    "start": match.start(), "end": match.end(),
                })
        events.sort(key=lambda e: e["start"])
        return events

    @staticmethod
    def _normalize(text: str, event_type: str) -> str:
        import re as _re
        m = _re.search(r"(\d+)\s+(days?|weeks?|months?|years?|hours?|minutes?)", text, _re.IGNORECASE)
        if m:
            num = int(m.group(1))
            unit = m.group(2).lower()
            if "day" in unit: return f"P{num}D"
            if "week" in unit: return f"P{num*7}D"
            if "month" in unit: return f"P{num*30}D"
            if "year" in unit: return f"P{num*365}D"
            if "hour" in unit: return f"PT{num}H"
            if "minute" in unit: return f"PT{num}M"

        import datetime
        today = datetime.date.today()
        tl = text.lower()
        if "yesterday" in tl: return (today - datetime.timedelta(days=1)).isoformat()
        if "today" in tl: return today.isoformat()
        if "tomorrow" in tl: return (today + datetime.timedelta(days=1)).isoformat()

        return text
