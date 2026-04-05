"""
Dialogue Act Classifier — hybrid zero-shot (regex rules + embedding fallback).
No training data required. Works immediately.
"""
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

LABELS = [
    "SYMPTOM_REPORT", "QUESTION", "DIAGNOSIS_STATEMENT",
    "TREATMENT_PLAN", "REASSURANCE", "HISTORY", "OTHER",
]

PROTOTYPES = {
    "SYMPTOM_REPORT": "Patient describing physical symptoms, pain, discomfort, or complaints.",
    "QUESTION": "Asking about health, treatment, medication, or history.",
    "DIAGNOSIS_STATEMENT": "Stating a diagnosis, test result, or clinical assessment.",
    "TREATMENT_PLAN": "Prescribing medication, ordering tests, recommending therapy.",
    "REASSURANCE": "Offering comfort, good news, or benign prognosis.",
    "HISTORY": "Discussing past medical history, family disease, or long-term medications.",
    "OTHER": "Administrative tasks, scheduling, consent forms, or closing remarks.",
}

# Regex rules evaluated IN ORDER. First match wins.
RULES = [
    # 1. QUESTIONS
    ("QUESTION", [
        r"\?$",
        r"^(how|what|when|where|why|who|which|do |does |did |is |are |was |were )"
        r"^(should |could |would |can |will |have |has |had )",
        r"^(could you|can you|do you|would you|should i|could i)",
    ]),
    # 2. OTHER (admin, signing, scheduling)
    ("OTHER", [
        r"\b(sign|signing|signed|form|consent|form|appointment|schedule|reception"
        r"|check (?:in|out)|waiting room|copay|insurance card|photo id|emergency contact"
        r"|next appointment|follow.?up appointment|front desk|receptionist)\b",
    ]),
    # 3. SYMPTOM REPORT (patient complaints — catch this BEFORE history)
    # Matches "I have/had/am having [symptom] for [duration]"
    ("SYMPTOM_REPORT", [
        r"\b(i[''][\s]?ve been |i have been |i am |i[''?]m )\s+\w*\s*(having|experiencing|feeling)\s+.*?\b",
        r"\b(patients?\s+(?:have|has|present(?:s|ed)?|complain(?:s|ed)?|report(?:s|ed)?)\s+)\b",
        # Symptom keywords (includes plurals with s?)
        r"\b(bad\s+headaches?\b|terrible\s+headaches?\b|severe\s+pain\b|chest\s+pain\b|shortness\s+of\s+breath)",
        r"\b(headaches?|migraines?|coughs?|fever|nausea|dizzines?s?|fatigue|tiredness|sore\s+throat|"
          r"ach(es|es?)|hurt(s|ing)?|swell(en|ing|ed|s)?|stiff|itch(ing|y)?|numb(ness)?|tingling?|"
          r"burn(ing|ed|s)?|palpitation(s)?|flush(ing|es)?|sweat(ing|s)?|stomach\s+(pain|ache)|"
          r"abdomen\s+(pain|ache)|back\s+(pain|ache)|joint\s+(pain|ache)|neck\s+pain|shoulder\s+pain|"
          r"knee\s+pain|leg\s+pain|arm\s+pain|hand\s+pain|foot\s+pain|toe\s+pain|"
          r"eye\s+(pain|sore)|ear\s+(pain|ache)|mouth\s+(pain|sore)|throat\s+(sore|pain)|"
          r"skin\s+(rash|itch)|facial\s+(pain|pressure))\b",
    ]),
    # 4. TREATMENT PLAN (imperative + prescriptions)
    ("TREATMENT_PLAN", [
        r"^(take |apply |use |wear |call |schedule |see |follow |start |stop )",
        r"\b(mg |ml |tablets|capsul|prescri|dosage|dose |twice daily|every \d+|q\w\b|as needed)",
    ]),
    # 5. REASSURANCE (comfort, prognosis)
    ("REASSURANCE", [
        r"\b(normal|fine|nothing serious|nothing to worry|don.?[a-zA-Z]*t worry"
        r"|no need to worry|significant improvement|ahead of schedule"
        r"|much better|typically will pass|benign|good news|reassuring|nothing concerning"
        r"|nothing alarming|nothing to be concerned)\b",
    ]),
    # 6. DIAGNOSIS STATEMENT (test results + diagnoses)
    ("DIAGNOSIS_STATEMENT", [
        r"\b(x[- ]?ray|mri|ct|scan|blood work|blood test|labs|lab results|"
         r"test result|ultrasound|echo|ekg|ecg|biopsy|exam) "
        r"(shows?|confirms|indicates|reveals|suggest|demonstrate)",
    ]),
    # 7. HISTORY (past events, chronic meds, family — but NOT current symptom complaints)
    ("HISTORY", [
        r"\b(family history|allergic|quit |smok|drink|surge|procedure|years ago|months ago"
        r"|diagnosed .* (?:year|month|since)|for \d+ (?:year|month|day)|been taking"
        r"|history of hypertension|history of diabetes|known .* allergies)\b",
    ]),
]


class DialogueActModel:
    """Hybrid zero-shot dialogue act classifier."""

    def __init__(self):
        self.embedder = None
        self.label_names = list(PROTOTYPES.keys())
        self.prototype_embeddings = None

    @classmethod
    def load(cls, models_dir: str = "models/dialogue_act") -> "DialogueActModel":
        instance = cls()
        from models.embedder import EmbedderModel
        instance.embedder = EmbedderModel.get_instance()
        if instance.embedder.model:
            instance.prototype_embeddings = instance.embedder.model.encode(list(PROTOTYPES.values()))
            print("[DialogueAct] Loaded hybrid zero-shot classifier")
        return instance

    def classify(self, text: str) -> dict:
        """Classify using rules first, embeddings as fallback."""
        label = self._structural_classify(text)
        if label:
            return {"label": label, "confidence": 0.85}

        # Fallback to embedding similarity
        if self.prototype_embeddings is None or self.embedder.model is None:
            return {"label": "OTHER", "confidence": 0.0}

        t_lower = text.strip().lower()
        text_emb = self.embedder.model.encode([text.strip()])
        sims = cosine_similarity(text_emb, self.prototype_embeddings)[0]

        boost = 5.0 if any(q in t_lower for q in ("?", "do you", "have you", "should i", "could it")) else 1.0
        if boost > 1.0:
            sims[LABELS.index("QUESTION")] *= boost

        exp_sims = np.exp(sims * 8)
        probs = exp_sims / exp_sims.sum()
        best = int(np.argmax(probs))
        return {"label": self.label_names[best], "confidence": round(float(probs[best]), 4)}

    @staticmethod
    def _structural_classify(text: str) -> str | None:
        for label, patterns in RULES:
            for pat in patterns:
                if re.search(pat, text, re.IGNORECASE):
                    return label
        return None
