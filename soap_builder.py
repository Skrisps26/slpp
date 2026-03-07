"""
SOAP Note Builder
Structures clinical entities into Subjective / Objective / Assessment / Plan format.
Adds clinical reasoning, flags, and recommendations.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.nlp_engine import ClinicalEntities, Diagnosis, Medication, Symptom, Vital


@dataclass
class SOAPNote:
    encounter_date: str
    encounter_time: str

    # Patient metadata
    patient_name: str = "Unknown"
    patient_dob: str = "Unknown"
    patient_id: str = ""
    patient_age: Optional[str] = None

    # Clinician metadata
    physician_name: str = "Unknown"
    physician_specialty: str = "General Practice"
    facility: str = "Medical Facility"
    encounter_type: str = "Office Visit"

    # SOAP sections
    chief_complaint: str = ""
    hpi: str = ""  # History of Present Illness
    subjective_summary: str = ""

    vitals_summary: list[Vital] = field(default_factory=list)
    physical_exam_notes: str = ""
    objective_summary: str = ""

    primary_diagnosis: Optional[Diagnosis] = None
    secondary_diagnoses: list[Diagnosis] = field(default_factory=list)
    differential_diagnoses: list[str] = field(default_factory=list)
    assessment_narrative: str = ""

    plan_items: list[str] = field(default_factory=list)
    medications_prescribed: list[Medication] = field(default_factory=list)
    medications_current: list[Medication] = field(default_factory=list)
    medications_discontinued: list[Medication] = field(default_factory=list)
    follow_up: str = ""
    plan_narrative: str = ""

    # Additional
    allergies: list[str] = field(default_factory=list)
    family_history: list[str] = field(default_factory=list)
    social_history: dict = field(default_factory=dict)
    review_of_systems: dict = field(default_factory=dict)
    clinical_flags: list[str] = field(default_factory=list)
    entities: Optional[ClinicalEntities] = None

    # Raw
    raw_transcript: str = ""


CLINICAL_FLAGS = {
    # Vital sign alert thresholds
    "Blood Pressure": lambda v: _flag_bp(v),
    "Heart Rate": lambda v: _flag_hr(v),
    "Temperature": lambda v: _flag_temp(v),
    "O2 Saturation": lambda v: _flag_o2(v),
    "Blood Glucose": lambda v: _flag_glucose(v),
}

RED_FLAG_SYMPTOMS = {
    "Chest pain": "⚠️  RED FLAG: Chest pain – consider ACS, PE, aortic dissection",
    "Hemoptysis": "⚠️  RED FLAG: Hemoptysis – consider malignancy, TB, PE",
    "Dyspnea": "⚠️  ALERT: Dyspnea – evaluate for cardiopulmonary etiology",
    "Syncope": "⚠️  RED FLAG: Syncope – consider cardiac arrhythmia, structural disease",
    "Seizure": "⚠️  RED FLAG: New-onset seizure – neurological workup warranted",
    "Unintentional weight loss": "⚠️  RED FLAG: Unexplained weight loss – consider malignancy, endocrine, GI pathology",
    "Hematochezia": "⚠️  ALERT: Rectal bleeding – consider lower GI source",
    "Melena": "⚠️  RED FLAG: Melena – consider upper GI bleed",
}

HIGH_RISK_MEDS = {
    "Warfarin": "⚠️  HIGH-ALERT MED: Warfarin – INR monitoring required",
    "Coumadin": "⚠️  HIGH-ALERT MED: Warfarin – INR monitoring required",
    "Insulin": "⚠️  HIGH-ALERT MED: Insulin – hypoglycemia risk, dose verification required",
    "Opioid": "⚠️  HIGH-ALERT MED: Opioid analgesic – monitor for dependency, respiratory depression",
    "Digoxin": "⚠️  HIGH-ALERT MED: Digoxin – narrow therapeutic index, toxicity monitoring required",
    "Amiodarone": "⚠️  HIGH-ALERT MED: Amiodarone – thyroid/pulmonary toxicity monitoring required",
    "Lithium": "⚠️  HIGH-ALERT MED: Lithium – narrow therapeutic index, level monitoring required",
}


def _flag_bp(value_str: str) -> Optional[str]:
    try:
        parts = value_str.split("/")
        sys_bp = int(parts[0])
        dia_bp = int(parts[1]) if len(parts) > 1 else 0
        if sys_bp >= 180 or dia_bp >= 120:
            return f"🚨 CRITICAL: BP {value_str} mmHg – Hypertensive crisis"
        if sys_bp >= 140 or dia_bp >= 90:
            return f"⚠️  ALERT: BP {value_str} mmHg – Hypertensive range"
        if sys_bp < 90:
            return f"🚨 CRITICAL: BP {value_str} mmHg – Hypotensive"
    except (ValueError, IndexError):
        pass
    return None


def _flag_hr(value_str: str) -> Optional[str]:
    try:
        hr = int(value_str.split()[0])
        if hr > 120:
            return f"⚠️  ALERT: HR {hr} bpm – Tachycardia"
        if hr > 100:
            return f"📋  NOTE: HR {hr} bpm – Mildly elevated"
        if hr < 50:
            return f"⚠️  ALERT: HR {hr} bpm – Bradycardia"
    except ValueError:
        pass
    return None


def _flag_temp(value_str: str) -> Optional[str]:
    try:
        t = float(re.sub(r"[^\d.]", "", value_str))
        if t >= 103.0:
            return f"🚨 CRITICAL: Temp {t}°F – High fever"
        if t >= 100.4:
            return f"⚠️  ALERT: Temp {t}°F – Fever"
        if t < 96.0:
            return f"⚠️  ALERT: Temp {t}°F – Hypothermia"
    except ValueError:
        pass
    return None


def _flag_o2(value_str: str) -> Optional[str]:
    try:
        o2 = float(re.sub(r"[^\d.]", "", value_str))
        if o2 < 88:
            return f"🚨 CRITICAL: SpO2 {o2}% – Severe hypoxia"
        if o2 < 92:
            return f"⚠️  ALERT: SpO2 {o2}% – Hypoxia – supplemental O2 indicated"
        if o2 < 95:
            return f"📋  NOTE: SpO2 {o2}% – Low-normal"
    except ValueError:
        pass
    return None


def _flag_glucose(value_str: str) -> Optional[str]:
    try:
        g = float(re.sub(r"[^\d.]", "", value_str))
        if g > 500:
            return f"🚨 CRITICAL: Glucose {g} mg/dL – Severe hyperglycemia"
        if g > 250:
            return f"⚠️  ALERT: Glucose {g} mg/dL – Hyperglycemia"
        if g < 54:
            return f"🚨 CRITICAL: Glucose {g} mg/dL – Severe hypoglycemia"
        if g < 70:
            return f"⚠️  ALERT: Glucose {g} mg/dL – Hypoglycemia"
    except ValueError:
        pass
    return None


class SOAPBuilder:
    def build(
        self,
        entities: ClinicalEntities,
        transcript: str,
        patient_info: dict,
        pre_existing: dict,
    ) -> SOAPNote:
        now = datetime.now()
        note = SOAPNote(
            encounter_date=now.strftime("%B %d, %Y"),
            encounter_time=now.strftime("%I:%M %p"),
            patient_name=patient_info.get("patient_name", "Unknown"),
            patient_dob=patient_info.get("patient_dob", "Unknown"),
            patient_id=patient_info.get("patient_id", ""),
            physician_name=patient_info.get("doctor_name", "Unknown"),
            physician_specialty=patient_info.get(
                "doctor_specialty", "General Practice"
            ),
            facility=patient_info.get("facility", "Medical Facility"),
            encounter_type=patient_info.get("encounter_type", "Office Visit"),
            raw_transcript=transcript,
            entities=entities,
        )

        # Calculate age from DOB
        try:
            dob = datetime.strptime(patient_info.get("patient_dob", ""), "%Y-%m-%d")
            age = (now - dob).days // 365
            note.patient_age = str(age)
        except (ValueError, TypeError):
            note.patient_age = None

        # Chief complaint
        note.chief_complaint = self._extract_chief_complaint(
            transcript, patient_info, entities
        )

        # HPI
        note.hpi = self._build_hpi(entities, transcript, note.chief_complaint)

        # Subjective
        note.subjective_summary = self._build_subjective(entities, note)

        # Vitals
        note.vitals_summary = entities.vitals

        # Objective
        note.objective_summary = self._build_objective(entities)

        # Merge pre-existing medications with extracted
        self._merge_medications(note, entities, pre_existing)

        # Diagnoses
        if entities.diagnoses:
            note.primary_diagnosis = next(
                (d for d in entities.diagnoses if d.primary), entities.diagnoses[0]
            )
            note.secondary_diagnoses = [
                d for d in entities.diagnoses if d != note.primary_diagnosis
            ]

        # Differentials
        note.differential_diagnoses = self._build_differentials(
            entities, note.chief_complaint
        )

        # Assessment narrative
        note.assessment_narrative = self._build_assessment(note, entities)

        # Plan
        note.plan_items = self._build_plan(entities, note)
        note.plan_narrative = self._build_plan_narrative(note)

        # Ancillary
        note.allergies = entities.allergies or pre_existing.get("allergies_list", [])
        note.family_history = entities.family_history
        note.social_history = entities.social_history
        note.review_of_systems = entities.review_of_systems

        # Clinical flags
        note.clinical_flags = self._generate_flags(entities)

        # Follow-up
        note.follow_up = self._extract_follow_up(transcript, entities.plan_items)

        return note

    def _extract_chief_complaint(
        self, text: str, info: dict, entities: ClinicalEntities
    ) -> str:
        # From pre-existing info
        if info.get("chief_complaint"):
            return info["chief_complaint"]

        # From transcript
        cc_patterns = [
            re.compile(
                r"(?:chief complaint|presenting with|here (?:today )?(?:for|because|with))[:\s]+([^.!\n]+)",
                re.I,
            ),
            re.compile(
                r"(?:came in|presents?)[:\s]*(?:today\s*)?(?:with|for)\s+([^.!\n]+)",
                re.I,
            ),
            re.compile(
                r"patient (?:is )?(?:complaining of|reports?|has)\s+([^.!\n]+)", re.I
            ),
        ]
        for pat in cc_patterns:
            m = pat.search(text)
            if m:
                return m.group(1).strip()[:120]

        # Fallback: first non-negated symptom
        for s in entities.symptoms:
            if not s.negated:
                return s.name
        return "Unspecified complaint"

    def _build_hpi(self, entities: ClinicalEntities, text: str, cc: str) -> str:
        parts = []
        active = [s for s in entities.symptoms if not s.negated]
        if not active:
            return f"Patient presents with {cc}. Details of the history of present illness were documented during the encounter."

        primary = active[0]
        desc = f"Patient presents with {primary.name.lower()}"
        if primary.duration:
            desc += f" for {primary.duration}"
        if primary.character:
            desc += f", described as {primary.character}"
        if primary.severity:
            desc += f", rated as {primary.severity} in severity"
        if primary.location:
            desc += f", localized to the {primary.location}"
        parts.append(desc + ".")

        if len(active) > 1:
            assoc = ", ".join(s.name.lower() for s in active[1 : min(4, len(active))])
            parts.append(f"Associated symptoms include {assoc}.")

        negated = [s for s in entities.symptoms if s.negated]
        if negated:
            neg_list = ", ".join(s.name.lower() for s in negated[:4])
            parts.append(f"Patient denies {neg_list}.")

        return " ".join(parts)

    def _build_subjective(self, entities: ClinicalEntities, note: "SOAPNote") -> str:
        lines = []
        active = [s for s in entities.symptoms if not s.negated]
        if active:
            lines.append(
                f"The patient reports {len(active)} symptom(s) on presentation."
            )
        return " ".join(lines)

    def _build_objective(self, entities: ClinicalEntities) -> str:
        if entities.vitals:
            v_str = "; ".join(f"{v.name}: {v.value} {v.unit}" for v in entities.vitals)
            return f"Vital signs obtained: {v_str}."
        return "Vital signs not documented in transcript."

    def _merge_medications(
        self, note: "SOAPNote", entities: ClinicalEntities, pre_existing: dict
    ):
        prescribed = [m for m in entities.medications if m.status == "prescribed"]
        current = [
            m for m in entities.medications if m.status in ("current", "mentioned")
        ]
        discontinued = [m for m in entities.medications if m.status == "discontinued"]

        # Add pre-existing medications from form
        pre_meds = pre_existing.get("current_medications", "")
        if pre_meds:
            for med_name in re.split(r",|;|\n", pre_meds):
                med_name = med_name.strip()
                if med_name:
                    current.append(Medication(name=med_name.title(), status="current"))

        note.medications_prescribed = prescribed
        note.medications_current = current
        note.medications_discontinued = discontinued

    def _build_differentials(self, entities: ClinicalEntities, cc: str) -> list[str]:
        differentials = []
        cc_lower = cc.lower()

        # Symptom-based differential generation
        ddx_map = {
            "chest pain": [
                "Acute Coronary Syndrome (ACS)",
                "Musculoskeletal chest wall pain",
                "Gastroesophageal reflux disease",
                "Pulmonary embolism",
                "Pneumothorax",
                "Pericarditis",
                "Aortic dissection",
            ],
            "shortness of breath": [
                "Community-acquired pneumonia",
                "Acute asthma exacerbation",
                "COPD exacerbation",
                "Pulmonary embolism",
                "Congestive heart failure",
                "Pneumothorax",
                "Anemia",
            ],
            "abdominal pain": [
                "Acute appendicitis",
                "Peptic ulcer disease",
                "Gastroenteritis",
                "Cholecystitis",
                "Ovarian pathology",
                "Mesenteric ischemia",
                "Renal colic",
            ],
            "headache": [
                "Tension-type headache",
                "Migraine without aura",
                "Cluster headache",
                "Hypertensive headache",
                "Cervicogenic headache",
                "Intracranial hypertension",
            ],
            "fever": [
                "Viral upper respiratory infection",
                "Bacterial infection (unspecified)",
                "Urinary tract infection",
                "Community-acquired pneumonia",
                "Influenza",
                "COVID-19",
            ],
            "dizziness": [
                "Benign paroxysmal positional vertigo (BPPV)",
                "Orthostatic hypotension",
                "Labyrinthitis / vestibular neuritis",
                "Anemia",
                "Hypoglycemia",
                "Cardiac arrhythmia",
            ],
        }

        for keyword, ddx in ddx_map.items():
            if keyword in cc_lower:
                differentials.extend(ddx)
                break

        # Add from identified diagnoses as differentials
        for d in entities.diagnoses:
            if d.certainty == "possible" and d.name not in differentials:
                differentials.append(d.name)

        return differentials[:6]

    def _build_assessment(self, note: "SOAPNote", entities: ClinicalEntities) -> str:
        parts = []
        name = note.patient_name
        age = f"{note.patient_age}-year-old" if note.patient_age else ""

        primary_str = ""
        if note.primary_diagnosis:
            d = note.primary_diagnosis
            primary_str = f"Primary impression is {d.name} (ICD-10: {d.icd10})"
        else:
            primary_str = (
                f"The clinical presentation is consistent with {note.chief_complaint}"
            )

        parts.append(
            f"{name} is a {age} patient presenting with {note.chief_complaint}. {primary_str}."
        )

        if note.secondary_diagnoses:
            comorbidities = ", ".join(d.name for d in note.secondary_diagnoses[:3])
            parts.append(f"Comorbid conditions noted include: {comorbidities}.")

        if note.differential_diagnoses:
            ddx = ", ".join(note.differential_diagnoses[:3])
            parts.append(f"Differential diagnoses considered include: {ddx}.")

        active_symptoms = [s for s in entities.symptoms if not s.negated]
        if active_symptoms:
            sym_names = ", ".join(s.name for s in active_symptoms[:4])
            parts.append(
                f"The symptom constellation ({sym_names}) informs the clinical reasoning above."
            )

        return " ".join(parts)

    def _build_plan(self, entities: ClinicalEntities, note: "SOAPNote") -> list[str]:
        items = list(entities.plan_items)

        # Add medication plans
        for m in note.medications_prescribed:
            dose = f" {m.dose}" if m.dose else ""
            freq = f" {m.frequency}" if m.frequency else ""
            items.append(f"Prescribe {m.name}{dose}{freq}")

        for m in note.medications_discontinued:
            items.append(f"Discontinue {m.name}")

        # Add generic recommendations based on diagnoses
        dx_plans = {
            "Hypertension": "Monitor blood pressure; lifestyle modifications including dietary sodium restriction and regular aerobic exercise",
            "Type 2 diabetes mellitus": "Blood glucose monitoring; HbA1c in 3 months; dietary counseling",
            "Hyperlipidemia": "Fasting lipid panel; dietary modifications; statin therapy as appropriate",
            "Urinary tract infection": "Urine culture and sensitivity; ensure adequate hydration",
            "Pneumonia": "Consider chest X-ray; sputum culture if productive cough; oxygen saturation monitoring",
            "Acute pharyngitis": "Rapid strep test / throat culture; antibiotics only if bacterial etiology confirmed",
            "Migraine": "Identify and avoid triggers; acute abortive therapy; consider prophylaxis if frequent",
        }

        if note.primary_diagnosis:
            for dx_key, plan_text in dx_plans.items():
                if dx_key.lower() in note.primary_diagnosis.name.lower():
                    items.append(plan_text)

        if not items:
            items.append(
                "Clinical management per physician discretion based on encounter findings."
            )

        return list(dict.fromkeys(items))  # deduplicate

    def _build_plan_narrative(self, note: "SOAPNote") -> str:
        parts = []
        if note.medications_prescribed:
            meds = "; ".join(m.name for m in note.medications_prescribed)
            parts.append(f"The following medications were prescribed: {meds}.")
        if note.follow_up:
            parts.append(f"Follow-up: {note.follow_up}.")
        if not parts:
            parts.append(
                "Management plan detailed above. Patient instructed to return if symptoms worsen or new symptoms develop."
            )
        return " ".join(parts)

    def _extract_follow_up(self, text: str, plan_items: list[str]) -> str:
        patterns = [
            re.compile(
                r"(?:follow.?up|return|come back|see (?:me|you|us))\s+(?:in\s+)?([^.!\n]{3,50})",
                re.I,
            ),
            re.compile(
                r"(?:appointment|visit)\s+(?:in\s+)?(\d+\s*(?:day|days|week|weeks|month|months))",
                re.I,
            ),
        ]
        for pat in patterns:
            m = pat.search(text)
            if m:
                return m.group(1).strip()
        for item in plan_items:
            if "follow" in item.lower():
                return item
        return "As needed or per physician instruction"

    def _generate_flags(self, entities: ClinicalEntities) -> list[str]:
        flags = []

        # Vital sign flags
        for vital in entities.vitals:
            flag_fn = CLINICAL_FLAGS.get(vital.name)
            if flag_fn:
                flag = flag_fn(vital.value)
                if flag:
                    flags.append(flag)

        # Symptom red flags
        for symptom in entities.symptoms:
            if not symptom.negated:
                flag = RED_FLAG_SYMPTOMS.get(symptom.name)
                if flag:
                    flags.append(flag)

        # Medication flags
        for med in entities.medications:
            for high_risk_name, flag_msg in HIGH_RISK_MEDS.items():
                if high_risk_name.lower() in med.name.lower():
                    if flag_msg not in flags:
                        flags.append(flag_msg)

        return flags
