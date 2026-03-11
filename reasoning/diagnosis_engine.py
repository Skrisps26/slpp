"""
reasoning/diagnosis_engine.py
Differential diagnosis ranking engine.
Loads diseases.json and scores each disease by symptom overlap.
"""

import json
import sys
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from nlp_engine.entities import ClinicalEntities, Diagnosis

_KB = _ROOT / "knowledge_base" / "diseases.json"


def _load_diseases() -> list:
    if _KB.exists():
        with open(_KB, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("diseases", [])
    return []


_DISEASES = _load_diseases()


class DiagnosisEngine:
    """
    Ranks differential diagnoses based on matched symptoms.
    Algorithm:
        base_score = matched_required × 3 + matched_optional × 1
        confidence = base_score / max_possible_score (capped at 0.95)
    """

    def __init__(self):
        self.diseases = _DISEASES

    def rank(
        self,
        entities: ClinicalEntities,
        top_n: int = 8,
    ) -> List[Diagnosis]:
        """
        Returns ranked list of Diagnosis objects with confidence scores.
        """
        active_symptoms = {
            s.name.lower()
            for s in entities.symptoms
            if not s.negated
        }

        # Also include colloquial names and partial matches
        expanded = set(active_symptoms)
        for sym in active_symptoms:
            parts = sym.split("/")
            expanded.update(p.strip() for p in parts)

        scored = []

        for disease in self.diseases:
            required = [r.lower() for r in disease.get("required_symptoms", [])]
            optional = [s.lower() for s in disease.get("symptoms", [])]

            # Check required symptoms
            required_matched = [r for r in required if self._matches(r, expanded)]
            if required and not required_matched:
                continue  # Skip if no required symptom matched

            # Score optional symptoms
            optional_matched = [s for s in optional if self._matches(s, expanded)]

            # Scoring
            n_req = len(required_matched) * 3
            n_opt = len(optional_matched)
            max_possible = len(required) * 3 + len(optional)

            if max_possible == 0 or (n_req + n_opt) == 0:
                continue

            raw_score = n_req + n_opt
            confidence = min(raw_score / max(max_possible, 1), 0.95)

            # Boost for exact chief complaint match
            if entities.symptoms:
                cc = entities.symptoms[0].name.lower()
                if any(cc in s or s in cc for s in optional):
                    confidence = min(confidence * 1.2, 0.95)

            icd10 = disease.get("icd10")
            dx = Diagnosis(
                name=disease["name"],
                icd10=icd10,
                certainty="possible",
                primary=False,
                confidence=round(confidence, 3),
                matched_symptoms=optional_matched[:5],
            )
            scored.append((confidence, dx, disease.get("urgency", "LOW")))

        # Sort by confidence desc, then urgency
        urgency_order = {"CRITICAL": 0, "URGENT": 1, "MODERATE": 2, "LOW": 3}
        scored.sort(key=lambda x: (-x[0], urgency_order.get(x[2], 4)))

        results = []
        for i, (conf, dx, urgency) in enumerate(scored[:top_n]):
            dx.primary = i == 0
            dx.certainty = (
                "confirmed" if conf >= 0.7
                else "possible" if conf >= 0.3
                else "possible"
            )
            results.append(dx)

        return results

    def _matches(self, disease_symptom: str, patient_symptoms: set) -> bool:
        """Fuzzy symptom matching: checks for substring overlap."""
        for ps in patient_symptoms:
            if (disease_symptom in ps) or (ps in disease_symptom):
                return True
            # Word-level match
            d_words = set(disease_symptom.split())
            p_words = set(ps.split())
            if d_words & p_words:
                return True
        return False

    def get_recommended_tests(self, diagnosis_name: str) -> List[str]:
        """Get recommended tests for a given diagnosis."""
        for disease in self.diseases:
            if disease["name"] == diagnosis_name:
                return disease.get("recommended_tests", [])
        return []

    def get_red_flags(self, diagnosis_name: str) -> List[str]:
        """Get red flags for a given diagnosis."""
        for disease in self.diseases:
            if disease["name"] == diagnosis_name:
                return disease.get("red_flags", [])
        return []
