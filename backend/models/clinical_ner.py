"""
Clinical NER + Negation model using BioBERT backbone with multi-task heads.
Falls back to rule-based extraction when the fine-tuned model is not available.
"""
import os
import json
import re
import torch
import numpy as np
from dataclasses import dataclass, asdict
from typing import List

NER_LABELS = ["O", "B-SYMPTOM", "I-SYMPTOM", "B-MEDICATION", "I-MEDICATION",
              "B-DIAGNOSIS", "I-DIAGNOSIS", "B-VITAL", "I-VITAL"]

NEG_LABELS = ["O", "B-NEG", "I-NEG", "B-UNCERTAIN", "I-UNCERTAIN"]


@dataclass
class NEREntity:
    """A named entity extracted from clinical text."""
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float = 1.0
    negated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class ClinicalNERModel:
    """
    Clinical NER using BioBERT.
    Falls back to regex when no fine-tuned checkpoint exists.
    """

    def __init__(self):
        self.tokenizer = None
        self.ner_model = None
        self.neg_head = None
        self.model_finetuned = False

    @classmethod
    def load(cls, ner_dir: str = "models/clinical_ner",
             neg_dir: str = "models/negation"):
        """Load the NER model and optionally the negation head."""
        instance = cls()
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        full_ner = ner_dir if os.path.isabs(ner_dir) else os.path.join(base, ner_dir)
        full_neg = neg_dir if os.path.isabs(neg_dir) else os.path.join(base, neg_dir)

        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            instance.tokenizer = AutoTokenizer.from_pretrained(
                "dmis-lab/biobert-base-cased-v1.2")

            has_fine = (
                os.path.exists(os.path.join(full_ner, "pytorch_model.bin"))
                or os.path.exists(os.path.join(full_ner, "model.safetensors"))
            )

            if has_fine:
                instance.ner_model = AutoModelForTokenClassification.from_pretrained(full_ner)
                instance.model_finetuned = True
            else:
                instance.ner_model = AutoModelForTokenClassification.from_pretrained(
                    "dmis-lab/biobert-base-cased-v1.2",
                    num_labels=len(NER_LABELS),
                    id2label={i: l for i, l in enumerate(NER_LABELS)},
                    label2id={l: i for i, l in enumerate(NER_LABELS)},
                )
                print("[ClinicalNER] Fine-tuned weights not found, using rule-based NER fallback")

            instance.ner_model.eval()

            # Load negation head
            neg_head_path = os.path.join(full_neg, "neg_head.pt")
            if os.path.exists(neg_head_path):
                hidden = instance.ner_model.config.hidden_size
                instance.neg_head = torch.nn.Linear(hidden, len(NEG_LABELS))
                instance.neg_head.load_state_dict(torch.load(neg_head_path, map_location="cpu"))
                instance.neg_head.eval()

        except Exception as e:
            print(f"[ClinicalNER] Model loading error: {e}")

        return instance

    def extract_entities(self, text: str) -> List[NEREntity]:
        """Extract clinical entities from text."""
        if self.ner_model is None or self.tokenizer is None:
            return self._rule_based_fallback(text)
        if not self.model_finetuned:
            return self._rule_based_fallback(text)

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.ner_model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        return self._decode_bio(tokens, predictions, NER_LABELS, text)

    def detect_negation(self, text: str) -> List[dict]:
        """Detect negation scopes."""
        if self.neg_head is not None and self.tokenizer is not None:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                hidden = self.ner_model(inputs["input_ids"]).last_hidden_state
                logits = self.neg_head(hidden)
            predictions = torch.argmax(logits, dim=-1)[0].tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            return self._decode_bio_tokens(tokens, predictions, NEG_LABELS, text)
        return self._rule_based_negation(text)

    def _decode_bio(self, tokens, predictions, labels, original_text):
        entities, cur = [], None
        for token, pred in zip(tokens, predictions):
            label = labels[pred] if pred < len(labels) else "O"
            if token in ("[CLS]", "[SEP]", "[PAD]"):
                continue
            if label.startswith("B-"):
                if cur: entities.append(cur)
                cur = {"text": token, "type": label[2:], "tokens": [token]}
            elif label.startswith("I-") and cur:
                cur["tokens"].append(token)
            else:
                if cur: entities.append(cur)
                cur = None
        if cur: entities.append(cur)

        result = []
        for ent in entities:
            t = "".join([tok.replace("##", "") for tok in ent["tokens"]])
            start = max(0, original_text.lower().find(t.lower()))
            end = start + len(t)
            result.append(NEREntity(text=t, entity_type=ent["type"], start=start, end=end))
        return result

    def _decode_bio_tokens(self, tokens, predictions, labels, original_text):
        scopes, cur = [], None
        for token, pred in zip(tokens, predictions):
            label = labels[pred] if pred < len(labels) else "O"
            if token in ("[CLS]", "[SEP]", "[PAD]"):
                continue
            if label.startswith("B-"):
                if cur: scopes.append(cur)
                cur = {"text": token, "type": label[2:], "tokens": [token]}
            elif label.startswith("I-") and cur:
                cur["tokens"].append(token)
            else:
                if cur: scopes.append(cur)
                cur = None
        if cur: scopes.append(cur)

        result = []
        for scope in scopes:
            t = "".join([tok.replace("##", "") for tok in scope["tokens"]])
            start = max(0, original_text.lower().find(t.lower()))
            result.append({"text": t, "type": scope["type"], "start": start, "end": start + len(t)})
        return result

    # ── Rule-based fallback ──

    def _rule_based_fallback(self, text: str) -> List[NEREntity]:
        entities = []
        for pat, etype in [
            (r'fever|cough|headache|nausea|fatigue|dizziness|'
             r'chest pain|shortness of breath|vomiting|diarrhea|'
             r'swelling|rash|chills|body aches|sore throat|'
             r'dry cough|mild headache|tight chest|burning sensation', "SYMPTOM"),
            (r'ibuprofen|aspirin|tylenol|metformin|lisinopril|'
             r'amoxicillin|acetaminophen|omeprazole|atorvastatin|multivitamin', "MEDICATION"),
            (r'diabetes|hypertension|pneumonia|asthma|bronchitis|'
             r'infection|viral infection|upper respiratory infection', "DIAGNOSIS"),
            (r'(?:temperature\s+(?:is\s+)?)\d+(?:\.\d+)?\s*F|'
             r'(?:heart rate|HR)\s*(?:is\s+|of\s+)?\d+\s*bpm|'
             r'(?:blood pressure|BP)\s*(?:is\s+|of\s+)?\d+/\d+', "VITAL"),
        ]:
            for m in re.finditer(pat, text, re.IGNORECASE):
                entities.append(NEREntity(
                    text=m.group(0), entity_type=etype,
                    start=m.start(), end=m.end(),
                ))
        return entities

    def _rule_based_negation(self, text: str) -> List[dict]:
        negation_triggers = [
            r"denies\s+([\w\s]+?)(?:\.|,|and|or|$)",
            r"no\s+(?:history\s+of\s+|significant\s+)?([\w\s]+?)(?:\.|,|and|or|$)",
            r"negative\s+for\s+([\w\s]+?)(?:\.|,|and|or|$)",
            r"negates?\s+([\w\s]+?)(?:\.|,|and|or|$)",
            r"does not have\s+([\w\s]+?)(?:\.|,|and|or|$)",
        ]
        scopes = []
        for pat in negation_triggers:
            for m in re.finditer(pat, text, re.IGNORECASE):
                end = min(m.end(), len(text))
                full = text[m.start():end].strip()
                scopes.append({"text": full, "type": "NEG", "start": m.start(), "end": end})
        return scopes
