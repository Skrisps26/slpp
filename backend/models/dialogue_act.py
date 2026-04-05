"""
Hybrid Dialogue Act Classifier.
Uses regex rules for structural features + MiniLM embedding similarity
for semantic content classification.
"""
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from models.embedder import EmbedderModel

LABELS = [
    "SYMPTOM_REPORT", "QUESTION", "DIAGNOSIS_STATEMENT",
    "TREATMENT_PLAN", "REASSURANCE", "HISTORY", "OTHER",
]


class DialogueActModel:
    """Hybrid dialogue act classifier using rules + embedding similarity."""

    def __init__(self):
        self.embedder = None
        self.label_names = LABELS
        self.prototype_embeddings = None
        self.prototypes = {
            "SYMPTOM_REPORT": (
                "Patient describes symptoms, pain, discomfort, illness, "
                "abnormal physical sensations, complaints about their body."
            ),
            "DIAGNOSIS_STATEMENT": (
                "Doctor states a diagnosis, test result, clinical assessment "
                "or medical conclusion about what condition the patient has."
            ),
            "TREATMENT_PLAN": (
                "Doctor tells patient what to do: prescribing medication, "
                "ordering tests, scheduling procedures, giving medical instructions."
            ),
            "REASSURANCE": (
                "Doctor gives good news, comforts patient, says results are normal, "
                "reduces worry, expresses favorable prognosis."
            ),
            "HISTORY": (
                "Discussion of past medical events, family disease history, "
                "lifestyle, past surgeries, allergies, long-term medication use."
            ),
            "OTHER": (
                "Administrative, scheduling, greetings, closing remarks, "
                "signing forms, insurance, logistics."
            ),
        }

    @classmethod
    def load(cls, models_dir: str = "models/dialogue_act"):
        instance = cls()
        from models.embedder import EmbedderModel
        instance.embedder = EmbedderModel.get_instance()
        try:
            instance.embedder.load()
        except Exception as e:
            print(f"[DialogueAct] Embedder failed to load: {e}")
            return instance

        if instance.embedder.model:
            instance.prototype_embeddings = instance.embedder.model.encode(
                list(instance.prototypes.values())
            )
            print(f"[DialogueAct] Loaded hybrid classifier with {len(instance.prototypes)} prototypes")

        return instance

    def _structural_classify(self, text: str) -> str | None:
        """Fast structural classification using regex rules."""
        t = text.strip()
        t_lower = t.lower()

        # Questions: obvious structural markers
        if (t.endswith('?') or
            re.match(r'^(how|what|when|where|why|who|which|do |does |did |is |are |was |were |'
                     r'should |could |would |can |will |have |has |had |'
                     r'is there |are there |do you |have you |has |will |'
                     r'could |should |would |can you )', t_lower)):
            return "QUESTION"

        # "Your X-ray/blood work/labs/results/test/imaging show/confirms/indicates/reveals" → DIAGNOSIS
        if re.search(r'\b(x-ray|mri|ct|scan|blood work|blood test|labs|lab results|'
                     r'test result|ultrasound|echo|ekg|ecg|biopsy|exam) '
                     r'(shows?|confirms|indicates|reveals|suggest|demonstrate)', t_lower):
            return "DIAGNOSIS_STATEMENT"

        # Other patterns that strongly indicate specific classes
        # Imperative/instruction → TREATMENT_PLAN
        if re.match(r'^(take |apply |use |wear |call |schedule |see |follow |start |begin |stop )', t_lower):
            return "TREATMENT_PLAN"

        # Prescription language → TREATMENT_PLAN
        if re.search(r'(prescrib|dosing |dosage |\d+ ?mg|\d+ ml|\d+ tablets|twice daily|three times|every \d+|as needed|per day|q[.hds]\b)', t_lower):
            return "TREATMENT_PLAN"

        # "I am prescribing" → TREATMENT_PLAN
        if re.search(r'i am (?:starting |recommending |prescribing |ordering |referring )', t_lower):
            return "TREATMENT_PLAN"

        # Reassurance markers
        if re.search(r'\b(normal|fine|nothing (?:serious|to worry about|to be concerned)|'
                     r'no (?:need to worry|cause for concern|reason to worry)|'
                     r'significant improvement|excellent|great news|ahead of schedule|'
                     r'much better|improved|responding well|common and )\b', t_lower):
            return "REASSURANCE"

        # Reassurance: "not concerning" / "not serious"
        if re.search(r"(not|no|nothing|isn't|aren't) .* (serious|concerning|worrisome|abnormal)", t_lower):
            return "REASSURANCE"

        # History: family history, personal history, lifestyle
        if re.search(r'\b(family history|my (?:mother|father|brother|sister|wife|husband|family|parents?|grandmother|grandfather|grandma|grandpa|uncle|aunt)|'
                     r'years ago|since (?:i |i )|i (?:used to|quit|started|stopped|began )|'
                     r'when i was |i (?:am|was|have been|had|took) .*(?:years|months|day|age|since))\b', t_lower):
            return "HISTORY"

        # "Have you ever" / "Have you been" → HISTORY when context is about past
        if re.match(r'^have (?:you|they) (?:ever |been )', t_lower):
            return "HISTORY"

        # Administrative → OTHER
        if re.search(r'\b(sign |form |appointment |insurance |copay|reception|check in|'
                     r'fill out |update |confirm your |portal |fax )\b', t_lower):
            return "OTHER"

        # "I will" / "We will" scheduling → OTHER
        if re.search(r'\b(i |we )will (?:send|schedule|call|fax|write )', t_lower):
            return "OTHER"

        return None  # No structural match, use embeddings

    def classify(self, text: str) -> dict:
        """Classify a sentence into a dialogue act label."""
        # Step 1: Try structural classification
        structural = self._structural_classify(text)
        if structural:
            return {"label": structural, "confidence": 0.85}

        # Step 2: Fall back to embedding similarity
        if self.prototype_embeddings is None or self.embedder.model is None:
            return {"label": "OTHER", "confidence": 0.0}

        text_emb = self.embedder.model.encode([text])
        sims = cosine_similarity(text_emb, self.prototype_embeddings)[0]

        # "I have" + symptom keywords → SYMPTOM_REPORT
        if re.match(r'^i (?:have |am |feel |have been |experiencing )', text.lower()):
            s_idx = self.label_names.index("SYMPTOM_REPORT")
            sims[s_idx] *= 3.0

        # Softmax
        exp_sims = np.exp(sims * 8)
        probs = exp_sims / exp_sims.sum()

        best_idx = int(np.argmax(probs))
        return {
            "label": self.label_names[best_idx],
            "confidence": round(float(probs[best_idx]), 4),
        }
