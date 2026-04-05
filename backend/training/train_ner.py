"""
Clinical NER fine-tuning script for BioBERT on i2b2 2010 dataset.
Run: python -m backend.training.train_ner
Expected time: ~90 minutes on 4GB VRAM GPU
"""
import os
import sys

LABELS = ["O", "B-SYMPTOM", "I-SYMPTOM", "B-MEDICATION", "I-MEDICATION",
          "B-DIAGNOSIS", "I-DIAGNOSIS", "B-VITAL", "I-VITAL"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
OUTPUT_DIR = "models/clinical_ner"


def train():
    try:
        from transformers import (
            AutoTokenizer, AutoModelForTokenClassification,
            TrainingArguments, Trainer, DataCollatorForTokenClassification
        )
        from datasets import load_dataset, Dataset
        import torch

        # Check for dataset
        dataset_path = "data/ner/i2b2_train"
        dataset_exists = os.path.exists(dataset_path)

        if not dataset_exists:
            print("[train_ner] i2b2 dataset not found. Creating mock dataset for structure test.")
            print("[train_ner] Download i2b2 2010 from https://www.i2b2.org/NLP/DataSets/Main.php")
            # Create minimal mock dataset
            mock_tokens = ["Patient", "reports", "chest", "pain", "and", "fever", "."]
            mock_tags = [0, 0, 1, 2, 0, 1, 2]  # O, O, B-SYMPTOM, I-SYMPTOM, O, B-SYMPTOM, I-SYMPTOM
            data = {"tokens": [mock_tokens] * 10, "ner_tags": [mock_tags] * 10}
            dataset = Dataset.from_dict(data)
            dataset = dataset.train_test_split(test_size=0.2)
        else:
            dataset = load_dataset("json", data_files={
                "train": os.path.join(dataset_path, "train.json"),
                "validation": os.path.join(dataset_path, "validation.json"),
            })

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        def tokenize_and_align_labels(examples):
            tokenized = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True)
            labels = []
            for i, label in enumerate(examples["ner_tags"]):
                word_ids = tokenized.word_ids(batch_index=i)
                prev_word_idx = None
                current_labels = []
                for word_idx in word_ids:
                    if word_idx is None:
                        current_labels.append(-100)
                    elif word_idx != prev_word_idx:
                        current_labels.append(label[word_idx])
                    else:
                        current_labels.append(-100)
                    prev_word_idx = word_idx
                labels.append(current_labels)
            tokenized["labels"] = labels
            return tokenized

        tokenized = dataset.map(tokenize_and_align_labels, batched=True)

        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
        )

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=4,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_dir="./logs/ner",
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=True,
            gradient_checkpointing=True,
            dataloader_num_workers=2,
            report_to="none",
        )

        collator = DataCollatorForTokenClassification(tokenizer)

        # Compute metrics
        def compute_metrics(p):
            import numpy as np
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)
            true_labels = [[LABELS[l] for l in label if l != -100] for label in labels]
            pred_labels = [[LABELS[p] for p, l in zip(prediction, label) if l != -100]
                          for prediction, label in zip(predictions, labels)]
            # Simple accuracy
            correct = sum(1 for true, pred in zip(true_labels, pred_labels)
                         for t, pp in zip(true, pred) if t == pp)
            total = sum(len(t) for t in true_labels)
            return {"accuracy": correct / total if total > 0 else 0.0}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        # Clean up checkpoints except best
        import glob
        for ckpt in glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")):
            import shutil
            shutil.rmtree(ckpt)

        print(f"[train_ner] Model saved to {OUTPUT_DIR}")

    except Exception as e:
        print(f"[train_ner] Training error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    train()
