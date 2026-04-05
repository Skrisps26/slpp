"""
NLI hallucination detector with automatic label-order detection.
Uses cross-encoder/nli-deberta-v3-small (CPU).
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"


class NLIModel:
    """Natural Language Inference model for hallucination detection."""

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.label_map = {}

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.eval()

        # Auto-detect label order from model config — never hardcode
        raw = self.model.config.id2label
        self.label_map = {}
        for idx, label in raw.items():
            u = label.upper()
            if "CONTRADICT" in u or u == "CONTRADICTION":
                self.label_map[idx] = "CONTRADICTED"
            elif "ENTAIL" in u or u == "ENTAILMENT":
                self.label_map[idx] = "ENTAILED"
            elif "NEUTRAL" in u:
                self.label_map[idx] = "NEUTRAL"
            else:
                self.label_map[idx] = u

    def score(self, premise: str, hypothesis: str) -> dict:
        if self.model is None:
            self.load()

        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt", truncation=True, max_length=512,
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = int(probs.argmax().item())
        return {
            "label": self.label_map.get(pred_idx, "NEUTRAL"),
            "confidence": round(float(probs[pred_idx].item()), 4),
        }
