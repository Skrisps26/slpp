"""
reasoning/risk_scores.py
Standard clinical risk scoring models:
  - Wells Score (PE risk)
  - HEART Score (chest pain / MACE risk)
  - CHA2DS2-VASc (stroke risk in AF)
"""

import re
from dataclasses import dataclass
from typing import Optional

from nlp_engine.entities import ClinicalEntities, Vital


@dataclass
class RiskScore:
    name: str
    score: int
    max_score: int
    interpretation: str
    risk_level: str          # "LOW" | "MODERATE" | "HIGH" | "VERY_HIGH"
    recommendation: str
    components: dict         # component → points awarded


class RiskScoreEngine:
    """Evaluates standard clinical risk scoring models."""

    def compute_all(self, entities: ClinicalEntities, patient_info: dict) -> list:
        scores = []

        symptoms_lower = {s.name.lower() for s in entities.symptoms if not s.negated}
        vitals_map = {v.name: v.value for v in entities.vitals}

        # Wells Score — compute whenever PE is a possibility (dyspnea/chest pain)
        if any(kw in symptoms_lower for kw in ["dyspnea", "chest pain", "tachycardia", "calf pain", "leg swelling"]):
            s = self._wells_score(entities, patient_info, vitals_map)
            if s:
                scores.append(s)

        # HEART Score — compute for chest pain presentations
        if any(kw in symptoms_lower for kw in ["chest pain", "chest pressure", "chest tightness"]):
            s = self._heart_score(entities, patient_info, vitals_map)
            if s:
                scores.append(s)

        # CHA2DS2-VASc — compute whenever palpitations/AF is in context
        if any(kw in symptoms_lower for kw in ["palpitations", "atrial fibrillation", "dysrhythmia"]):
            s = self._chads_vasc(entities, patient_info)
            if s:
                scores.append(s)

        return scores

    # ── Wells Score ───────────────────────────────────────────────────────

    def _wells_score(
        self, entities: ClinicalEntities, info: dict, vitals_map: dict
    ) -> Optional[RiskScore]:
        components = {}
        total = 0

        symptoms_lower = {s.name.lower() for s in entities.symptoms if not s.negated}
        meds_lower = {m.name.lower() for m in entities.medications}
        known = (info.get("known_conditions", "") + " " + info.get("current_medications", "")).lower()

        # Clinical signs of DVT (+3)
        if any(k in symptoms_lower for k in ["calf pain", "leg swelling", "ankle edema"]):
            components["Clinical signs of DVT"] = 3
            total += 3

        # PE more likely than alternative (+3)
        # (heuristic: chest pain + dyspnea with no better explanation)
        if "chest pain" in symptoms_lower and "dyspnea" in symptoms_lower:
            components["PE most likely diagnosis"] = 3
            total += 3

        # Heart rate > 100 (+1.5)
        hr_val = vitals_map.get("Heart Rate")
        if hr_val:
            try:
                hr = int(re.sub(r"[^\d]", "", hr_val)[:3])
                if hr > 100:
                    components["Heart rate > 100 bpm"] = 1
                    total += 1  # simplified to integer
            except ValueError:
                pass

        # Immobilization / surgery in past 4 weeks (+1.5)
        if any(k in known for k in ["immobiliz", "surgery", "hospitalized", "bedrest", "fracture"]):
            components["Immobilization or surgery ≤4 weeks"] = 2
            total += 2

        # Prior DVT/PE (+1.5)
        if any(k in known for k in ["dvt", "pe", "pulmonary embolism", "deep vein thrombosis"]):
            components["Previous DVT/PE"] = 2
            total += 2

        # Hemoptysis (+1)
        if "hemoptysis" in symptoms_lower:
            components["Hemoptysis"] = 1
            total += 1

        # Active malignancy (+1)
        if any(k in known for k in ["cancer", "malignancy", "tumor", "oncology", "chemotherapy"]):
            components["Active malignancy"] = 1
            total += 1

        if total == 0:
            return None

        if total <= 1:
            interp, risk = "Low probability (PE unlikely)", "LOW"
            rec = "Consider D-dimer to rule out PE"
        elif total <= 6:
            interp, risk = "Moderate probability", "MODERATE"
            rec = "Consider CT Pulmonary Angiography or V/Q scan"
        else:
            interp, risk = "High probability (PE likely)", "HIGH"
            rec = "CT Pulmonary Angiography urgently; anticoagulation if no contraindications"

        return RiskScore(
            name="Wells Score (PE)",
            score=total,
            max_score=12,
            interpretation=interp,
            risk_level=risk,
            recommendation=rec,
            components=components,
        )

    # ── HEART Score ───────────────────────────────────────────────────────

    def _heart_score(
        self, entities: ClinicalEntities, info: dict, vitals_map: dict
    ) -> Optional[RiskScore]:
        components = {}
        total = 0
        known = (info.get("known_conditions", "") + " " + info.get("current_medications", "")).lower()
        symptoms_lower = {s.name.lower() for s in entities.symptoms if not s.negated}

        # H – History (0-2)
        if any(k in symptoms_lower for k in ["crushing", "pressure", "diaphoresis", "radiation", "nausea"]):
            components["History (highly suspicious)"] = 2
            total += 2
        elif "chest pain" in symptoms_lower:
            components["History (moderately suspicious)"] = 1
            total += 1
        else:
            components["History (slightly suspicious)"] = 0

        # E – ECG (approximate from context, 0-2)
        # We can only give 0 since we don't have ECG data
        components["ECG (no data)"] = 0

        # A – Age (0-2)
        try:
            from datetime import datetime
            dob_str = info.get("patient_dob", "")
            dob = datetime.strptime(dob_str, "%Y-%m-%d")
            age = (datetime.now() - dob).days // 365
            if age >= 65:
                components["Age ≥ 65"] = 2
                total += 2
            elif age >= 45:
                components["Age 45-64"] = 1
                total += 1
            else:
                components["Age < 45"] = 0
        except (ValueError, TypeError):
            components["Age (unknown)"] = 0

        # R – Risk Factors (0-2)
        risk_factors = ["hypertension", "diabetes", "hyperlipidemia", "smoking", "obesity", "family history", "atherosclerosis"]
        rf_count = sum(1 for rf in risk_factors if rf in known)
        if rf_count >= 3:
            components["≥3 Risk factors or known atherosclerosis"] = 2
            total += 2
        elif rf_count >= 1:
            components["1-2 Risk factors"] = 1
            total += 1
        else:
            components["No risk factors"] = 0

        # T – Troponin (0-2) — can't know without labs
        components["Troponin (lab not available)"] = 0

        if total == 0:
            return None

        if total <= 3:
            interp, risk = "Low risk - MACE 1.7%", "LOW"
            rec = "Safe for early discharge; follow-up in outpatient setting"
        elif total <= 6:
            interp, risk = "Moderate risk - MACE 12-16.6%", "MODERATE"
            rec = "Admit for serial troponins; further evaluation required"
        else:
            interp, risk = "High risk - MACE 50%+", "HIGH"
            rec = "Urgent cardiology consultation; rule out ACS aggressively"

        return RiskScore(
            name="HEART Score (Chest Pain)",
            score=total,
            max_score=10,
            interpretation=interp,
            risk_level=risk,
            recommendation=rec,
            components=components,
        )

    # ── CHA2DS2-VASc ──────────────────────────────────────────────────────

    def _chads_vasc(
        self, entities: ClinicalEntities, info: dict
    ) -> Optional[RiskScore]:
        components = {}
        total = 0
        known = (info.get("known_conditions", "") + " " + info.get("current_medications", "")).lower()

        # C – CHF (+1)
        if any(k in known for k in ["congestive heart failure", "chf", "heart failure"]):
            components["Congestive heart failure"] = 1
            total += 1

        # H – Hypertension (+1)
        if "hypertension" in known or "high blood pressure" in known:
            components["Hypertension"] = 1
            total += 1

        # A2 – Age ≥ 75 (+2) or 65–74 (+1)
        try:
            from datetime import datetime
            dob_str = info.get("patient_dob", "")
            dob = datetime.strptime(dob_str, "%Y-%m-%d")
            age = (datetime.now() - dob).days // 365
            if age >= 75:
                components["Age ≥ 75"] = 2
                total += 2
            elif age >= 65:
                components["Age 65-74"] = 1
                total += 1
        except (ValueError, TypeError):
            pass

        # D – Diabetes (+1)
        if "diabetes" in known or "diabetic" in known:
            components["Diabetes mellitus"] = 1
            total += 1

        # S2 – Stroke/TIA history (+2)
        if any(k in known for k in ["stroke", "tia", "transient ischemic"]):
            components["Prior stroke/TIA"] = 2
            total += 2

        # V – Vascular disease (+1)
        if any(k in known for k in ["coronary artery disease", "cad", "peripheral vascular", "myocardial infarction", "aortic plaque"]):
            components["Vascular disease"] = 1
            total += 1

        # Sc – Sex category female (+1)
        # (cannot determine from transcript without explicit mention)

        if total == 0:
            return None

        if total == 0:
            interp, risk, rec = "Low risk", "LOW", "May not require anticoagulation"
        elif total == 1:
            interp, risk, rec = "Low-moderate risk", "LOW", "Consider anticoagulation; weigh bleeding risk"
        elif total == 2:
            interp, risk, rec = "Moderate risk (estimated 2.2% annual stroke rate)", "MODERATE", "Anticoagulation recommended (DOAC preferred)"
        else:
            interp, risk, rec = f"High risk (score {total}, estimated {min(total*1.5, 15):.1f}%/yr stroke rate)", "HIGH", "Anticoagulation strongly recommended (DOAC preferred)"

        return RiskScore(
            name="CHA₂DS₂-VASc (Stroke Risk in AF)",
            score=total,
            max_score=9,
            interpretation=interp,
            risk_level=risk,
            recommendation=rec,
            components=components,
        )
