"""
Dialogue act classifier training.
Uses frozen MiniLM-L6-v2 embeddings + MLP head.
Run: python -m backend.training.train_dialogue_acts
Expected time: 20 minutes on CPU
Expected disk: 80 MB
"""
import os
import json
import torch
from torch import nn, optim

LABELS = ["SYMPTOM_REPORT", "QUESTION", "DIAGNOSIS_STATEMENT",
          "TREATMENT_PLAN", "REASSURANCE", "HISTORY", "OTHER"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
OUTPUT_DIR = "models/dialogue_act"


def train():
    try:
        from sentence_transformers import SentenceTransformer

        # Load data
        data_path = "data/dialogue_acts/train.json"
        if not os.path.exists(data_path):
            print("[train_dialogue_acts] Training data not found.")
            print("[train_dialogue_acts] Run generate_synthetic_data.py first.")
            return

        with open(data_path) as f:
            data = json.load(f)

        labels = [LABEL2ID[item["label"]] for item in data]
        texts = [item["text"] for item in data]

        # Load frozen embedder
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        for param in embedder.parameters():
            param.requires_grad = False

        embeddings = embedder.encode(texts)
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.long)

        # Build classifier head
        classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, len(LABELS)),
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

        # Train
        max_epochs = 10
        batch_size = 32
        n_samples = len(texts)

        for epoch in range(max_epochs):
            classifier.train()
            total_loss = 0
            n_batches = 0

            # Shuffle
            perm = torch.randperm(n_samples)
            for i in range(0, n_samples, batch_size):
                batch_idx = perm[i:i+batch_size]
                X = embeddings[batch_idx]
                y = label_tensor[batch_idx]

                optimizer.zero_grad()
                logits = classifier(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches
            print(f"[train_dialogue_acts] Epoch {epoch+1}/{max_epochs}, loss: {avg_loss:.4f}")

        # Evaluate
        classifier.eval()
        with torch.no_grad():
            logits = classifier(embeddings)
            preds = logits.argmax(dim=1)
            accuracy = (preds == label_tensor).float().mean().item()
        print(f"[train_dialogue_acts] Training accuracy: {accuracy:.4f}")

        # Save classifier head
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        torch.save(classifier.state_dict(), os.path.join(OUTPUT_DIR, "classifier_head.pt"))
        with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
            json.dump({"labels": LABELS}, f)

        print(f"[train_dialogue_acts] Model saved to {OUTPUT_DIR}")

    except Exception as e:
        print(f"[train_dialogue_acts] Error: {e}")


if __name__ == "__main__":
    train()
