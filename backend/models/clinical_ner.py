"""
Clinical NER + Negation model using BioBERT backbone with multi-task heads.
Backbone: dmis-lab/biobert-base-cased-v1.2
Runs on CPU during inference, GPU during training.
"""
import json
import os
import torch
import numpy as np
import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict


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


NER_LABELS = ["O", "B-SYMPTOM", "I-SYMPTOM", "B-MEDICATION", "I-MEDICATION",
              "B-DIAGNOSIS", "I-DIAGNOSIS", "B-VITAL", "I-VITAL"]

NEG_LABELS = ["O", "B-NEG", "I-NEG", "B-UNCERTAIN", "I-UNCERTAIN"]


class ClinicalNERModel:
    """
    Multi-task clinical NER and negation detection.
    Uses a shared BioBERT backbone with separate classification heads.
    Falls back to rule-based extraction if fine-tuned model not available.
    """

    def __init__(self):
        self.tokenizer = None
        self.ner_model = None
        self.neg_head = None
        self.model_finetuned = False
        self.ner_labels = NER_LABELS
        self.neg_labels = NEG_LABELS

    @classmethod
    def load(cls, ner_model_dir: str = "models/clinical_ner",
             negation_dir: str = "models/negation") -> "ClinicalNERModel":
        """Load the NER model and optionally the negation head."""
        instance = cls()

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        full_ner_dir = ner_model_dir if os.path.isabs(ner_model_dir) else os.path.join(base_dir, ner_model_dir)
        full_neg_dir = negation_dir if os.path.isabs(negation_dir) else os.path.join(base_dir, negation_dir)

        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification

            instance.tokenizer = AutoTokenizer.from_pretrained(
                "dmis-lab/biobert-base-cased-v1.2"
            )

            # Check if fine-tuned model exists
            has_finetuned = os.path.exists(full_ner_dir) and any(
                f in ['pytorch_model.bin', 'model.safetensors']
                for f in os.listdir(full_ner_dir)
            )

            if has_finetuned:
                instance.ner_model = AutoModelForTokenClassification.from_pretrained(full_ner_dir)
                instance.model_finetuned = True
                print(f"[ClinicalNER] Loaded fine-tuned NER model from {full_ner_dir}")
            else:
                # Use raw pretrained model with proper head
                instance.ner_model = AutoModelForTokenClassification.from_pretrained(
                    "dmis-lab/biobert-base-cased-v1.2",
                    num_labels=len(NER_LABELS),
                    id2label={i: l for i, l in enumerate(NER_LABELS)},
                    label2id={l: i for i, l in enumerate(NER_LABELS)},
                )
                print("[ClinicalNER] Fine-tuned NER model not found, skipping model-based extraction")

            instance.ner_model.eval()

            # Load negation head if available
            neg_head_path = os.path.join(full_neg_dir, "neg_head.pt")
            if os.path.exists(neg_head_path):
                hidden_size = instance.ner_model.config.hidden_size
                instance.neg_head = torch.nn.Linear(hidden_size, len(NEG_LABELS))
                instance.neg_head.load_state_dict(torch.load(neg_head_path, map_location="cpu"))
                instance.neg_head.eval()
                print("[ClinicalNER] Negation head loaded")
            else:
                print("[ClinicalNER] Negation head not found, will use rule-based fallback")

        except Exception as e:
            print(f"[ClinicalNER] Model loading error: {e}")

        return instance

    def extract_entities(self, text: str) -> List[NEREntity]:
        """Extract clinical entities from text."""
        # Use rule-based if model not loaded or not fine-tuned
        if self.ner_model is None or self.tokenizer is None or not self.model_finetuned:
            return self._rule_based_fallback(text)

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )

        with torch.no_grad():
            outputs = self.ner_model(**inputs)
            logits = outputs.logits

        predictions = torch.argmax(logits, dim=-1)[0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        entities = self._decode_predictions(tokens, predictions, self.ner_labels, text)
        return entities

    def detect_negation(self, text: str) -> List[dict]:
        """Detect negation scopes in text."""
        if self.neg_head is not None and self.tokenizer is not None:
            # Use the negation head on the shared backbone
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                backbone_out = self.ner_model(inputs["input_ids"], output_hidden_states=True)
                hidden_states = backbone_out.hidden_states[-1]
                neg_logits = self.neg_head(hidden_states)
            predictions = torch.argmax(neg_logits, dim=-1)[0].tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            return self._decode_negation_predictions(tokens, predictions, self.neg_labels, text)

        return self._rule_based_negation(text)

    def _decode_predictions(self, tokens, predictions, labels, original_text):
        """Decode token-level predictions into entity spans (BIO format)."""
        entities = []
        current_entity = None

        for i, (token, pred) in enumerate(zip(tokens, predictions)):
            label = labels[pred] if pred < len(labels) else "O"

            if token in ("[CLS]", "[SEP]", "[PAD]"):
                continue

            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "text": token,
                    "type": label[2:],
                    "start": None,
                    "end": None,
                    "tokens": [token],
                }
            elif label.startswith("I-") and current_entity:
                current_entity["tokens"].append(token)
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        result = []
        for ent in entities:
            text = "".join([t.replace("##", "") for t in ent["tokens"]])
            start = original_text.lower().find(text.lower())
            if start == -1:
                start = 0
            end = start + len(text)
            result.append(NEREntity(
                text=text, entity_type=ent["type"],
                start=start, end=end,
            ))
        return result

    def _decode_negation_predictions(self, tokens, predictions, labels, original_text):
        """Decode negation predictions into scope annotations."""
        scopes = []
        current_scope = None

        for i, (token, pred) in enumerate(zip(tokens, predictions)):
            label = labels[pred] if pred < len(labels) else "O"
            if token in ("[CLS]", "[SEP]", "[PAD]"):
                continue

            if label.startswith("B-"):
                if current_scope:
                    scopes.append(current_scope)
                current_scope = {"text": token, "type": label[2:], "tokens": [token]}
            elif label.startswith("I-") and current_scope:
                current_scope["tokens"].append(token)
            else:
                if current_scope:
                    scopes.append(current_scope)
                    current_scope = None

        if current_scope:
            scopes.append(current_scope)

        result = []
        for scope in scopes:
            text = "".join([t.replace("##", "") for t in scope["tokens"]])
            start = original_text.lower().find(text.lower())
            if start == -1:
                start = 0
            result.append({"text": text, "type": scope["type"], "start": start, "end": start + len(text)})
        return result

    def _rule_based_fallback(self, text: str) -> List[NEREntity]:
        """Rule-based entity extraction as fallback when model not available."""
        entities = []

        # Symptom patterns (word boundaries required)
        symptom_patterns = [
            "fever", "cough", "headache", "nausea", "fatigue", "dizziness",
            "swelling", "rash", "chills", "vomiting", "diarrhea",
            "shortness of breath", "chest pain", "body aches",
            "back pain", "abdominal pain", "joint pain",
            "sore throat", "runny nose", "muscle cramps",
            "heart palpitations", "night sweats", "burning sensation",
            "numbness", "blurred vision", "tingling",
        ]
        for pat in symptom_patterns:
            for m in re.finditer(r'\b' + re.escape(pat) + r'\b', text, re.IGNORECASE):
                entities.append(NEREntity(
                    text=m.group(0), entity_type="SYMPTOM",
                    start=m.start(), end=m.end(),
                ))

        medication_patterns = [
            "ibuprofen", "aspirin", "tylenol", "metformin", "lisinopril",
            "amoxicillin", "acetaminophen", "omeprazole", "atorvastatin",
            "metoprolol", "amlodipine", "levothyroxine", "prednisone",
            "gabapentin", "hydrochlorothiazide",
        ]
        for pat in medication_patterns:
            for m in re.finditer(r'\b' + re.escape(pat) + r'\b', text, re.IGNORECASE):
                entities.append(NEREntity(
                    text=m.group(0), entity_type="MEDICATION",
                    start=m.start(), end=m.end(),
                ))

        diagnosis_patterns = [
            "diabetes", "hypertension", "pneumonia", "asthma",
            "bronchitis", "infection", "arthritis", "anemia",
            "heart disease", "kidney disease", "thyroid",
        ]
        for pat in diagnosis_patterns:
            for m in re.finditer(r'\b' + re.escape(pat) + r'\b', text, re.IGNORECASE):
                entities.append(NEREntity(
                    text=m.group(0), entity_type="DIAGNOSIS",
                    start=m.start(), end=m.end(),
                ))

        # Vital patterns
        vital_patterns = [
            r"blood pressure\s*(?:is\s*)?\d+/\d+",
            r"heart rate\s*(?:is\s*)?\d+\s*bpm",
            r"temperature\s*(?:is\s*)?[\d.]+\s*°?F",
            r"\b\d+/\d+\s*mm?\s*Hg\b",
        ]
        for pat in vital_patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                entities.append(NEREntity(
                    text=m.group(0), entity_type="VITAL",
                    start=m.start(), end=m.end(),
                ))

        return entities

    def _rule_based_negation(self, text: str) -> List[dict]:
        """Rule-based negation detection as fallback (NegEx-style patterns)."""
        negation_triggers = [
            r"(?i)denies\s+([\w\s]+?)(?:\.|,|and|or|$)",
            r"(?i)no\s+(?:history\s+of\s+|significant\s+|evidence\s+of\s+)?([\w\s]+?)(?:\.|,|and|or|$)",
            r"(?i)negative\s+for\s+([\w\s]+?)(?:\.|,|and|or|$)",
            r"(?i)negates?\s+([\w\s]+?)(?:\.|,|and|or|$)",
            r"(?i)ruled\s+out\s+([\w\s]+?)(?:\.|,|and|or|$)",
            r"(?i)denied\s+([\w\s]+?)(?:\.|,|and|or|$)",
        ]

        scopes = []
        for trigger_pat in negation_triggers:
            for m in re.finditer(trigger_pat, text):
                start = m.start()
                # Scope extends to match end or sentence boundary
                end = min(m.end(), len(text))
                # Find actual sentence end
                for boundary_char in ['.!', '?']:
                    boundary_pos = text.find(boundary_char, start)
                    if boundary_pos != -1 and boundary_pos < end:
                        end = boundary_pos + 1
                        break
                
                scope_text = text[start:end].strip()
                if scope_text:
                    scopes.append({
                        "text": scope_text,
                        "type": "NEG",
                        "start": start,
                        "end": end,
                    })

        return scopes
