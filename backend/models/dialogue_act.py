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


class DialogueActModel:
    """Hybrid zero-shot dialogue act classifier."""

    def __init__(self):
        self.embedder = None
        self.label_names = list(PROTOTYPES.keys())
        self.prototype_embeddings = None

    @classmethod
    def load(cls) -> "DialogueActModel":
        instance = cls()
        from models.embedder import EmbedderModel
        instance.embedder = EmbedderModel.get_instance()
        if instance.embedder.model:
            instance.prototype_embeddings = instance.embedder.model.encode(list(PROTOTYPES.values()))
            print("[DialogueAct] Loaded hybrid zero-shot classifier")
        return instance

    def classify(self, text: str) -> dict:
        """Classify a sentence using rules first, embeddings as fallback."""
        label = self._structural_classify(text)
        if label:
            return {"label": label, "confidence": 0.85}

        # Fallback to embedding similarity
        if self.prototype_embeddings is None or self.embedder.model is None:
            return {"label": "OTHER", "confidence": 0.0}

        text_emb = self.embedder.model.encode([text.strip()])
        sims = cosine_similarity(text_emb, self.prototype_embeddings)[0]
        boost = 5.0 if text.strip().endswith("?") else 1.0
        if boost > 1.0:
            sims[LABELS.index("QUESTION")] *= boost

        exp_sims = np.exp(sims * 8)
        probs = exp_sims / exp_sims.sum()
        best = int(np.argmax(probs))
        return {"label": self.label_names[best], "confidence": round(float(probs[best]), 4)}

    @staticmethod
    def _structural_classify(text: str) -> str | None:
        """Regex-based structural classification."""
        t = text.strip().lower()

        # 1. QUESTIONS
        if t.endswith("?") or re.match(
            r"^(how|what|when|where|why|who|which|do |does |did |is |are |"
            r"was |were |should |could |would |can |will |have |has |had |"
            r"could you|can you|do you|would you|should i)", t
        ):
            return "QUESTION"

        # 2. OTHER (admin, consent, scheduling — must be high priority)
        if re.search(r"\b(sign|signing|signed|consent|form|paperwork|reception|check.?in|check.?out|"
                     r"waiting|copay|insurance|phone number|address|email |contact info|"
                     r"next visit|next appointment|follow.?up appointment|front desk)\b", t):
            return "OTHER"

        # 3. TREATMENT PLAN (imperative prescriptions)
        if re.match(r"^(take |apply |use |wear |call |schedule |see a |follow |start |stop )", t):
            return "TREATMENT_PLAN"
        # Prescription dosing keywords
        if re.search(r"\b(\d+\s*mg|\d+\s*ml|tablets?|capsules?|prescri|dosage|dose of|"
                     r"twice daily|three times|every \d+.*hours|q\w\s|as needed|start .* for|begin .*)\b", t):
            return "TREATMENT_PLAN"

        # 4. REASSURANCE (comfort, prognosis)
        if re.search(r"\b(completely normal|fine|nothing serious|don'?t? worry|"
                     r"no need to worry|significant improvement|ahead of schedule|"
                     r"very reassuring|good news|nothing concerning|nothing alarming)\b", t):
            return "REASSURANCE"

        # 5. DIAGNOSIS (test results + clinical statements)
        if re.search(r"\b(blood work|labs|ultrasound|mri|ct scan|x[- ]?ray|biopsy|test results|culture|pathology|exam)\b.*?\b(shows|confirms|suggests|reveals|indicates|demonstrates|consistent with)\b", t):
            return "DIAGNOSIS_STATEMENT"

        # 6. HISTORY (past events, chronic meds, family)
        # "on [medication]" patterns
        if re.search(r"\b(on\s+(?:ibuprofen|lisinopril|metformin|omeprazole|atorvastatin|aspirin|tylenol|amoxicillin|multivitamin))\b", t):
            return "HISTORY"
        # "for [duration]" patterns (indicating duration of medication use)
        if re.search(r"\b(for\s+(?:the past|over|about|almost|nearly)\s+\d+\s+(?:day|week|month|year|decade)s?)\b", t):
            return "HISTORY"
        # General history keywords
        if re.search(r"\b(family history|allergic to|quit\s+\w+|smok(?:e|ing|ed)|drink(s?|ing|ing)|"
                     r"(?:surge|procedur)e\s+"
                     r"years\s+ago|months\s+ago|diagnosed\s+(?:with|at|in)|"
                     r"history\s+of|allergies\s+(?:include|to))\b", t):
            return "HISTORY"
        # "have been [verb]ing" or "have had" (past/ongoing condition)
        if re.search(r"\b(have\s+been\s+\w+ing|have\s+had\s+\w+ed?|has\s+been\s+\w+ing|has\s+had)\b", t):
            return "HISTORY"

        # 7. SYMPTOM REPORT (patient complaints — broad catch-all)
        if re.search(r"\b(i['']?\s+have\s+\w+.*(?:pain|ache|hurt|cough|fever|nausea|dizziness|"
                     r"headache|fatigue|tired|swell|swollen|stiff|itch|numb|tingle|burn|burning|"
                     r"sore|throat|shortness of breath|breathing|chest|palpitation|flush|sweat))|"
                     r"\b(patients?\s+(?:have|has|present(?:s|ed)?|complain(?:s|ed)?|report(?:s|ed)?)\s+)|"
                     r"\b(my\s+(?:chest|head|throat|back|stomach|abdomen|arm|leg|hand|finger|foot|toe|"
                     r"joint|knee|shoulder|neck|eye|ear|mouth|nose|skin|face).*?(?:hurt|pain|ache|feel|sore))+",
                     t):
            return "SYMPTOM_REPORT"
        if re.search(r"\b(pain|ache|hurts|hurt|cough|fever|nausea|dizz|fatigue|sore throat)\b", t):
            return "SYMPTOM_REPORT"

        return None
