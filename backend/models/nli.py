"""
NLI (Natural Language Inference) model for hallucination detection.
Uses cross-encoder/nli-deberta-v3-small (180MB, CPU).
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class NLIModel:
    """
    Natural Language Inference model.
    Labels: CONTRADICTION=0, NEUTRAL=1, ENTAILMENT=2
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load(self):
        """Load the NLI model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()

    def score(self, premise: str, hypothesis: str) -> dict:
        """
        Score whether the hypothesis is entailed by the premise.

        Returns:
            dict with "label" (ENTAILED/NEUTRAL/CONTRADICTED) and "confidence"
        """
        if self.model is None:
            self.load()

        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]

        # DeBERTa NLI label order: 0=contradiction, 1=neutral, 2=entailment
        label_map = {0: "CONTRADICTED", 1: "NEUTRAL", 2: "ENTAILED"}
        pred = probs.argmax().item()

        return {
            "label": label_map[pred],
            "confidence": round(probs[pred].item(), 4),
        }
