"""
Dialogue Act Classifier using MiniLM-L6-v2 + a classification head.
Labels: SYMPTOM_REPORT, QUESTION, DIAGNOSIS_STATEMENT, TREATMENT_PLAN, REASSURANCE, HISTORY, OTHER
"""
import json
import os
import torch
from torch import nn
from .embedder import EmbedderModel

LABELS = [
    "SYMPTOM_REPORT", "QUESTION", "DIAGNOSIS_STATEMENT",
    "TREATMENT_PLAN", "REASSURANCE", "HISTORY", "OTHER",
]


class DialogueActModel:
    """Dialogue act classifier using frozen MiniLM embeddings + MLP head."""

    def __init__(self):
        self.classifier: nn.Module = None
        self.labels = LABELS

    @classmethod
    def load(cls, models_dir: str = "models/dialogue_act") -> "DialogueActModel":
        """Load pretrained classifier head."""
        instance = cls()
        embedder = EmbedderModel.get_instance()
        embedder.load()

        # Build classifier architecture
        instance.classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, len(LABELS)),
        )

        # Load trained weights if available
        head_path = os.path.join(models_dir, "classifier_head.pt")
        if os.path.exists(head_path):
            instance.classifier.load_state_dict(torch.load(head_path, map_location="cpu"))
        instance.classifier.eval()

        # Load label config if available
        config_path = os.path.join(models_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            instance.labels = config.get("labels", LABELS)

        return instance

    def classify(self, text: str) -> dict:
        """Classify a single sentence into a dialogue act label."""
        embedder = EmbedderModel.get_instance()
        embedding = embedder.encode([text])[0]
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = self.classifier(embedding_tensor)
            probs = torch.softmax(logits, dim=-1)[0]

        pred_label = probs.argmax().item()
        confidence = probs[pred_label].item()

        return {
            "label": self.labels[pred_label],
            "confidence": round(confidence, 4),
        }
