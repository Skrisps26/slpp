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
    "SYMPTOM_REPORT": (
        "Patient describes physical symptoms, pain, discomfort, or abnormal sensations."
    ),
    "QUESTION": (
        "Any question asking about health, treatment, medication, or history."
    ),
    "DIAGNOSIS_STATEMENT": (
        "Doctor stating a diagnosis, test result, clinical assessment, or conclusion."
    ),
    "TREATMENT_PLAN": (
        "Doctor prescribing medication, ordering tests, recommending therapy, or giving care instructions."
    ),
    "REASSURANCE": (
        "Doctor offering comfort, good news, or benign prognosis."
    ),
    "HISTORY": (
        "Discussion of past medical history, family disease, lifestyle, medications, or allergies."
    ),
    "OTHER": (
        "Administrative, scheduling, greetings, closing remarks."
    ),
}


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
        """Classify a sentence using rules first, embeddings as fallback."""
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
        """Fast regex-based structural classification."""
        t = text.strip().lower()

        # Questions
        if t.endswith("?") or re.match(
            r"^(how|what|when|where|why|who|which|do |does |did |is |are |was |were |"
            r"should |could |would |can |will |have |has |had |could you|can you|do you)", t
        ):
            return "QUESTION"

        # Imperative / prescription
        if re.match(r"^(take |apply |use |wear |call |schedule |see |follow |start |stop )", t):
            return "TREATMENT_PLAN"
        if re.search(r"\b(mg |ml |tablets|capsul|prescri|dosage|dose |twice daily|every \d+|q\w\b|as needed)", t):
            return "TREATMENT_PLAN"

        # Reassurance
        if re.search(r"\b(normal|fine|nothing (?:serious|to worry|concerning)|"
                     r"no need to worry|significant improvement|ahead of schedule|"
                     r"much better|don.?t worry)\b", t):
            return "REASSURANCE"

        # Diagnosis
        if re.search(r"\b(x[- ]?ray|mri|ct|scan|blood work|test|labs|ultrasound|biopsy|exam)\b.*?\b(shows|confirms|suggests|reveals|indicates|consistent)", t):
            return "DIAGNOSIS_STATEMENT"

        # History
        if re.search(r"\b(family history|years ago|since |started |diagnosed |on .*(?:mg|daily)|"
                     r"allergic|quit |smok|drink|surge|procedure)\b", t):
            return "HISTORY"

        return None
