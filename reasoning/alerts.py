"""
reasoning/alerts.py
Enhanced clinical alert detection:
  - Critical vital sign combinations
  - Drug allergy conflicts
  - High-risk symptom patterns
  - Emergency criteria
"""

import re
from typing import List

from nlp_engine.entities import ClinicalEntities


# ── Vital sign alert thresholds ───────────────────────────────────────────────

VITAL_ALERTS = {
    "Blood Pressure": [
        (lambda v: _bp_sys(v) >= 180 or _bp_dia(v) >= 120,
         "🚨 CRITICAL: BP {v} mmHg — Hypertensive emergency; end-organ damage risk"),
        (lambda v: _bp_sys(v) >= 160 or _bp_dia(v) >= 100,
         "⚠️ ALERT: BP {v} mmHg — Stage 2 hypertension; requires urgent management"),
        (lambda v: _bp_sys(v) < 90,
         "🚨 CRITICAL: BP {v} mmHg — Hypotension; hemodynamic instability"),
    ],
    "Heart Rate": [
        (lambda v: _num(v) > 150,
         "🚨 CRITICAL: HR {v} bpm — Severe tachycardia; evaluate for arrhythmia or hemodynamic compromise"),
        (lambda v: _num(v) > 120,
         "⚠️ ALERT: HR {v} bpm — Tachycardia; evaluate for underlying cause"),
        (lambda v: _num(v) < 40,
         "🚨 CRITICAL: HR {v} bpm — Severe bradycardia"),
        (lambda v: _num(v) < 50,
         "⚠️ ALERT: HR {v} bpm — Bradycardia"),
    ],
    "O2 Saturation": [
        (lambda v: _fnum(v) < 88,
         "🚨 CRITICAL: SpO₂ {v}% — Severe hypoxia; immediate supplemental O₂ required"),
        (lambda v: _fnum(v) < 92,
         "⚠️ ALERT: SpO₂ {v}% — Hypoxia; supplemental O₂ indicated"),
        (lambda v: _fnum(v) < 95,
         "📋 NOTE: SpO₂ {v}% — Low-normal saturation; monitor closely"),
    ],
    "Temperature": [
        (lambda v: _fnum(v) >= 104.0,
         "🚨 CRITICAL: Temp {v}°F — High-grade fever; evaluate for serious infection"),
        (lambda v: _fnum(v) >= 103.0,
         "⚠️ ALERT: Temp {v}°F — High fever; antipyretics and workup warranted"),
        (lambda v: _fnum(v) >= 100.4,
         "📋 NOTE: Temp {v}°F — Fever; clinical monitoring required"),
        (lambda v: _fnum(v) < 96.0,
         "⚠️ ALERT: Temp {v}°F — Hypothermia"),
    ],
    "Blood Glucose": [
        (lambda v: _fnum(v) > 500,
         "🚨 CRITICAL: Glucose {v} mg/dL — Severe hyperglycemia; evaluate for DKA/HHS"),
        (lambda v: _fnum(v) > 250,
         "⚠️ ALERT: Glucose {v} mg/dL — Hyperglycemia; insulin management required"),
        (lambda v: _fnum(v) < 54,
         "🚨 CRITICAL: Glucose {v} mg/dL — Severe hypoglycemia; immediate treatment required"),
        (lambda v: _fnum(v) < 70,
         "⚠️ ALERT: Glucose {v} mg/dL — Hypoglycemia; treat and monitor"),
    ],
}

# High-risk symptom combinations
CRITICAL_COMBINATIONS = [
    (
        {"chest pain", "dyspnea"},
        "⚠️ ALERT: Chest pain + Dyspnea — ACS/PE pattern; urgent cardiac/pulmonary workup",
    ),
    (
        {"chest pain", "diaphoresis"},
        "🚨 CRITICAL: Chest pain + Diaphoresis — High-suspicion ACS; consider emergent ECG + troponin",
    ),
    (
        {"fever", "altered mentation", "tachycardia"},
        "🚨 CRITICAL: Fever + Altered mentation + Tachycardia — Sepsis criteria; immediate workup",
    ),
    (
        {"headache", "fever", "stiff neck"},
        "🚨 CRITICAL: Headache + Fever + Meningism — Bacterial meningitis possible; LP urgently",
    ),
    (
        {"suicidal ideation"},
        "🚨 CRITICAL: Suicidal ideation documented — Immediate psychiatric safety assessment required",
    ),
    (
        {"hemoptysis", "chest pain"},
        "⚠️ ALERT: Hemoptysis + Chest pain — PE and malignancy must be excluded",
    ),
    (
        {"weakness", "confusion"},
        "⚠️ ALERT: Focal weakness + Confusion — Stroke protocol; brain imaging urgently",
    ),
]

# High-alert medications
HIGH_ALERT_MEDS = {
    "warfarin": "⚠️ HIGH-ALERT: Warfarin — INR monitoring required; bleeding risk; drug interactions",
    "coumadin": "⚠️ HIGH-ALERT: Warfarin (Coumadin) — INR monitoring required",
    "insulin": "⚠️ HIGH-ALERT: Insulin — Hypoglycemia risk; dose verification required",
    "digoxin": "⚠️ HIGH-ALERT: Digoxin — Narrow therapeutic index; toxicity monitoring required",
    "amiodarone": "⚠️ HIGH-ALERT: Amiodarone — Thyroid/pulmonary toxicity monitoring required",
    "lithium": "⚠️ HIGH-ALERT: Lithium — Narrow therapeutic index; lithium level monitoring required",
    "methotrexate": "⚠️ HIGH-ALERT: Methotrexate — Weekly dosing only; hepatotoxicity + bone marrow suppression risk",
    "heparin": "⚠️ HIGH-ALERT: Heparin — Monitor for HIT; aPTT/anti-Xa monitoring",
    "clozapine": "⚠️ HIGH-ALERT: Clozapine — Agranulocytosis risk; mandatory WBC monitoring",
}

# Drug class allergy cross-reactivity
ALLERGY_CROSS_REACTIVITY = {
    "penicillin": ["amoxicillin", "ampicillin", "augmentin", "dicloxacillin", "nafcillin", "oxacillin", "piperacillin"],
    "sulfa": ["bactrim", "sulfamethoxazole", "hydrochlorothiazide", "furosemide"],
    "nsaid": ["ibuprofen", "naproxen", "aspirin", "celecoxib", "indomethacin", "ketorolac"],
    "cephalosporin": ["cephalexin", "cefazolin", "ceftriaxone", "cefdinir"],
}


# ── Numeric helpers ───────────────────────────────────────────────────────────

def _bp_sys(value: str) -> int:
    try:
        return int(value.split("/")[0])
    except (ValueError, IndexError, AttributeError):
        return 0


def _bp_dia(value: str) -> int:
    try:
        return int(value.split("/")[1])
    except (ValueError, IndexError, AttributeError):
        return 0


def _num(value: str) -> int:
    try:
        return int(re.sub(r"[^\d]", "", str(value))[:4])
    except (ValueError, TypeError):
        return 0


def _fnum(value: str) -> float:
    try:
        return float(re.sub(r"[^\d.]", "", str(value))[:6])
    except (ValueError, TypeError):
        return 0.0


# ── Alert engine ──────────────────────────────────────────────────────────────

class AlertEngine:
    """Generates clinical alerts."""

    def generate(self, entities: ClinicalEntities, patient_info: dict) -> List[str]:
        alerts = []

        alerts += self._vital_alerts(entities.vitals)
        alerts += self._combination_alerts(entities)
        alerts += self._medication_alerts(entities)
        alerts += self._allergy_conflict_alerts(entities, patient_info)

        # Deduplicate
        seen = set()
        unique = []
        for a in alerts:
            if a not in seen:
                seen.add(a)
                unique.append(a)

        return unique

    def _vital_alerts(self, vitals: list) -> List[str]:
        alerts = []
        for vital in vitals:
            rules = VITAL_ALERTS.get(vital.name, [])
            for predicate, template in rules:
                try:
                    if predicate(vital.value):
                        alerts.append(template.format(v=vital.value))
                        break  # Only most severe alert per vital
                except Exception:
                    continue
        return alerts

    def _combination_alerts(self, entities: ClinicalEntities) -> List[str]:
        alerts = []
        active = {s.name.lower() for s in entities.symptoms if not s.negated}
        # Expand with partial matches
        expanded = set()
        for sym in active:
            expanded.add(sym)
            for part in sym.split("/"):
                expanded.add(part.strip())
            # Add key words
            for word in sym.split():
                expanded.add(word)

        for required_symptoms, alert_msg in CRITICAL_COMBINATIONS:
            matched = sum(
                1 for req in required_symptoms
                if any(req in exp or exp in req for exp in expanded)
            )
            if matched >= len(required_symptoms):
                alerts.append(alert_msg)

        return alerts

    def _medication_alerts(self, entities: ClinicalEntities) -> List[str]:
        alerts = []
        for med in entities.medications:
            med_lower = med.name.lower()
            for key, alert_msg in HIGH_ALERT_MEDS.items():
                if key in med_lower:
                    alerts.append(alert_msg)
                    break
        return alerts

    def _allergy_conflict_alerts(
        self, entities: ClinicalEntities, patient_info: dict
    ) -> List[str]:
        alerts = []
        all_allergies = list(entities.allergies)
        allergy_str = patient_info.get("allergies", "")
        if allergy_str and allergy_str.upper() != "NKDA":
            all_allergies += [a.strip() for a in allergy_str.split(",")]

        prescribed = [m.name.lower() for m in entities.medications if m.status == "prescribed"]
        current = [m.name.lower() for m in entities.medications]

        for allergy in all_allergies:
            allergy_lower = allergy.lower()
            cross = ALLERGY_CROSS_REACTIVITY.get(allergy_lower, [allergy_lower])

            for med_name in prescribed + current:
                for cr in cross:
                    if cr in med_name or med_name in cr:
                        alerts.append(
                            f"🚨 CRITICAL: DRUG ALLERGY CONFLICT — Patient allergic to {allergy}, "
                            f"medication {med_name} may be contraindicated"
                        )
                        break

        return alerts
