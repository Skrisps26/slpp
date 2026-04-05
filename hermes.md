# AGENTS.md — MedScribe → GCIS Refactor
## Grounded Clinical Intelligence System (GCIS)
### Instructions for autonomous execution by Hermes

---

## 0. READ THIS ENTIRE FILE BEFORE WRITING A SINGLE LINE OF CODE

You are refactoring the MedScribe project (`github.com/Skrisps26/slpp`) into a research-grade
clinical NLP system called **GCIS — Grounded Clinical Intelligence System**.

The architecture has one central idea: **Extract → Generate → Verify → Refine (EGV-R)**.
- Small trained NLP models extract structured information (fast, deterministic, interpretable)
- A local LLM generates fluent clinical prose from those structured entities
- An NLI model validates every sentence in the output against the source transcript
- A self-correction loop re-prompts the LLM to fix any sentence scored as hallucinated

This is not a cosmetic refactor. You are replacing the Streamlit frontend with Next.js,
restructuring the entire Python backend into a clean modular API, training or downloading
fine-tuned models, and wiring everything together.

Do not skip steps. Do not approximate. Complete each phase fully before moving to the next.

---

## 1. HARDWARE CONSTRAINTS — ABSOLUTE LIMITS

You must respect these at all times. Never recommend or use anything outside these bounds.

```
GPU VRAM:         4 GB (maximum — assume GTX 1650 / RTX 3050 class)
System RAM:       8 GB (safe assumption)
Disk storage:     12 GB total budget for ALL models + code + data combined
Training budget:  6 GB disk for training artifacts (datasets + checkpoints)
Model storage:    5 GB disk for final inference models
Code + frontend:  1 GB

VRAM allocation during training (must not exceed 4GB):
  - Use batch_size=8 for BERT-base models
  - Use gradient_accumulation_steps=4 to simulate batch_size=32
  - Use fp16=True always
  - Use gradient_checkpointing=True for anything over 110M params
  - NEVER load two neural models into GPU simultaneously during training
  - During inference: NLI and embedding models run on CPU. Only LLM uses GPU via Ollama.

TRAINING TIME TARGET: Each fine-tuning job must complete in under 3 hours on the target GPU.
```

---

## 2. THE ARCHITECTURE — EGV-R LOOP

```
                    ┌─────────────────────────────────────────────┐
                    │              GCIS PIPELINE                   │
                    └─────────────────────────────────────────────┘

[Audio / Text Input]
        │
        ▼
┌──────────────┐
│   Whisper    │  (keep existing — use whisper "base" model, 74MB, runs on CPU)
│  Transcriber │
└──────┬───────┘
       │  raw transcript (str)
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: EXTRACTION LAYER                    │
│                  (all models run on CPU in inference)           │
│                                                                 │
│  ┌──────────────────┐   ┌──────────────────┐                   │
│  │  Dialogue Act    │   │  Clinical NER    │                   │
│  │  Classifier      │   │  (fine-tuned     │                   │
│  │  (MiniLM-L6 +    │   │   BioBERT)       │                   │
│  │   linear head)   │   │                  │                   │
│  └────────┬─────────┘   └────────┬─────────┘                   │
│           │                      │                             │
│  ┌────────▼─────────┐   ┌────────▼─────────┐                   │
│  │  Negation Scope  │   │  Temporal        │                   │
│  │  Detector        │   │  Extractor       │                   │
│  │  (BioBERT BIO    │   │  (HeidelTime     │                   │
│  │   tagger)        │   │   via py-heidel) │                   │
│  └────────┬─────────┘   └────────┬─────────┘                   │
│           └──────────┬───────────┘                             │
│                      │                                         │
│              ClinicalEntities (dataclass)                       │
│              + DialogueActSequence                              │
│              + NegationScopes                                   │
│              + TemporalTimeline                                 │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STAGE 2: GENERATION LAYER                     │
│                                                                 │
│  RAG: embed knowledge_base/ with FAISS                         │
│  retrieve top-3 relevant disease docs for differential dx       │
│                                                                 │
│  Ollama (mistral:7b-instruct-q4_K_M)                           │
│  Input: structured entities + retrieved KB docs + prompt        │
│  Output: SOAP note in strict JSON schema                        │
│  Constrained decoding: JSON schema enforced via Ollama format   │
└──────────────────────┬──────────────────────────────────────────┘
                       │  generated SOAP (JSON)
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                 STAGE 3: VERIFICATION LAYER                     │
│                                                                 │
│  NLI Model: cross-encoder/nli-deberta-v3-small (CPU)           │
│  For each SOAP sentence:                                        │
│    premise   = full transcript                                  │
│    hypothesis = SOAP sentence                                   │
│    score     = {entailed, neutral, contradicted}               │
│                                                                 │
│  Output:                                                        │
│    faithfulness_score = entailed_count / total_sentences        │
│    flagged_sentences  = list of contradicted sentences          │
│    attribution_map    = SOAP sentence → source transcript line  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
  faithfulness >= 0.85        faithfulness < 0.85
          │                         │
          ▼                         ▼
   VERIFIED SOAP            STAGE 4: REFINEMENT
   (pass to frontend)            │
                                  │  re-prompt LLM with:
                                  │  - original entities
                                  │  - flagged sentence
                                  │  - "this claim is not in the transcript, revise"
                                  │
                                  ▼
                            regenerate flagged sentences only
                            re-run NLI on new sentences
                            (max 2 refinement iterations)
                                  │
                                  ▼
                           VERIFIED SOAP
                        (with faithfulness score)
```

---

## 3. MODEL CATALOG — EXACT MODELS TO USE

Do not deviate from this list. Every model was selected for the 4GB VRAM constraint.

### 3.1 Whisper (keep existing)
```
model:    openai/whisper-base
size:     74 MB
device:   CPU
reason:   already integrated, no change needed
```

### 3.2 Clinical NER Model
```
base:     dmis-lab/biobert-base-cased-v1.2
size:     400 MB (base) → ~420 MB after fine-tuning
device:   CPU for inference, GPU for training
task:     token classification (NER)
labels:   B-SYMPTOM, I-SYMPTOM, B-MEDICATION, I-MEDICATION,
          B-DIAGNOSIS, I-DIAGNOSIS, B-VITAL, I-VITAL, O
dataset:  i2b2 2010 (medication/problem/treatment)
          download: https://www.i2b2.org/NLP/DataSets/Main.php (free academic)
          fallback: n2c2 2018 shared task (same site)
training: see Section 5.1
disk:     420 MB
```

### 3.3 Negation Scope Detector
```
base:     dmis-lab/biobert-base-cased-v1.2  (SHARE weights with NER — different head only)
size:     +50 MB for the extra classification head
device:   CPU for inference
task:     token classification (BIO span tagging for negation/uncertainty scope)
labels:   B-NEG, I-NEG, B-UNCERTAIN, I-UNCERTAIN, O
dataset:  NegEx corpus
          download: https://github.com/chapmanbe/negex (free, no registration)
          additional: SFU Review Corpus for uncertainty
training: see Section 5.2
disk:     50 MB extra (shared backbone with NER model)
IMPORTANT: load the SAME BioBERT backbone, attach two separate heads.
           This is multi-task fine-tuning. Do not train two separate 400MB models.
```

### 3.4 Dialogue Act Classifier
```
base:     sentence-transformers/all-MiniLM-L6-v2
size:     80 MB
device:   CPU always
task:     sentence classification
labels:   SYMPTOM_REPORT, QUESTION, DIAGNOSIS_STATEMENT,
          TREATMENT_PLAN, REASSURANCE, HISTORY, OTHER
dataset:  Use GPT-4 or any LLM to synthetically generate 500 labeled examples
          of doctor-patient dialogue sentences (one-time generation, free with any API).
          Format: {"text": "...", "label": "SYMPTOM_REPORT"}
          Store in data/dialogue_acts/train.json
training: see Section 5.3
disk:     80 MB
```

### 3.5 Temporal Extractor
```
library:  py-heideltime (Python wrapper for HeidelTime)
          pip install py-heideltime
          also requires: java (JRE 8+) — add to Dockerfile
size:     ~200 MB (Java + HeidelTime resources)
device:   CPU, rule-based (no GPU needed, no training needed)
task:     temporal expression detection + normalization to ISO 8601
output:   [{"text": "3 days ago", "type": "DURATION", "normalized": "P3D", "start": 12, "end": 22}]
training: NONE — rule-based system, works out of the box
disk:     200 MB
```

### 3.6 LLM — Generation Layer
```
model:    mistral:3b-instruct-q4_K_M  (via Ollama)
size:     4.1 GB
device:   GPU (uses full 4GB VRAM — do NOT run any other GPU model simultaneously)
task:     SOAP note generation from structured entities
          differential diagnosis reasoning over retrieved KB docs
format:   Ollama JSON mode (constrained output)
disk:     4.1 GB
pull cmd: ollama pull mistral:3b-instruct-q4_K_M
```

### 3.7 NLI Hallucination Detector
```
model:    cross-encoder/nli-deberta-v3-small
size:     180 MB
device:   CPU always (do not put on GPU — Ollama owns the GPU)
task:     premise-hypothesis NLI for hallucination detection
          + sentence attribution via cosine similarity
library:  from transformers import AutoTokenizer, AutoModelForSequenceClassification
disk:     180 MB
```

### 3.8 RAG Embedding Model
```
model:    sentence-transformers/all-MiniLM-L6-v2  (SAME as 3.4 — reuse instance)
task:     embed knowledge_base/ documents + embed queries for retrieval
index:    FAISS flat index (faiss-cpu)
disk:     index < 10 MB for typical KB size
```

### TOTAL DISK BUDGET CHECK
```
whisper-base:          74 MB
biobert (shared):     420 MB   ← NER head + negation head on same backbone
MiniLM (shared):       80 MB   ← dialogue acts + RAG embeddings reuse same instance
heideltime:           200 MB
mistral q4_K_M:      4100 MB
deberta-v3-small:     180 MB
training data:        800 MB   ← i2b2 + NegEx + synthetic dialogue acts
training checkpoints: 900 MB   ← delete after training, keep only best checkpoint
─────────────────────────────
TOTAL:               ~6754 MB ≈ 6.6 GB  ✓ within 12 GB budget
```

---

## 4. REPOSITORY STRUCTURE — TARGET STATE

After refactoring, the repo must look exactly like this:

```
slpp/
├── AGENTS.md                        ← this file
├── README.md                        ← rewrite completely
├── docker-compose.yml               ← updated (add next.js service)
├── .env.example                     ← updated
│
├── backend/                         ← Python FastAPI backend
│   ├── main.py                      ← FastAPI app entry point
│   ├── requirements.txt
│   ├── Dockerfile
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── orchestrator.py          ← EGV-R loop coordinator
│   │   ├── transcriber.py           ← keep existing whisper logic, wrap in class
│   │   ├── extractor.py             ← Stage 1: all 4 extraction models
│   │   ├── generator.py             ← Stage 2: Ollama + RAG
│   │   ├── verifier.py              ← Stage 3: NLI + attribution
│   │   └── refiner.py               ← Stage 4: self-correction loop
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── clinical_ner.py          ← BioBERT NER + negation (multi-task heads)
│   │   ├── dialogue_act.py          ← MiniLM classifier
│   │   ├── temporal.py              ← HeidelTime wrapper
│   │   ├── nli.py                   ← DeBERTa NLI wrapper
│   │   └── embedder.py              ← MiniLM embedder (shared instance with dialogue_act)
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── indexer.py               ← build FAISS index from knowledge_base/
│   │   └── retriever.py             ← query FAISS, return top-k docs
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── entities.py              ← ClinicalEntities, DialogueAct, TemporalEvent dataclasses
│   │   ├── soap.py                  ← SOAPNote, SOAPSection dataclasses
│   │   └── verification.py          ← VerificationResult, AttributionMap dataclasses
│   │
│   ├── training/                    ← training scripts, run once, not part of inference
│   │   ├── train_ner.py
│   │   ├── train_negation.py
│   │   ├── train_dialogue_acts.py
│   │   ├── generate_synthetic_data.py
│   │   └── evaluate_all.py          ← ablation study runner
│   │
│   └── knowledge_base/              ← move existing knowledge_base/ here
│
├── frontend/                        ← Next.js 14 app
│   ├── package.json
│   ├── next.config.js
│   ├── tsconfig.json
│   ├── Dockerfile
│   │
│   └── src/
│       ├── app/
│       │   ├── layout.tsx
│       │   ├── page.tsx             ← main dashboard
│       │   └── globals.css
│       │
│       ├── components/
│       │   ├── AudioRecorder.tsx    ← browser mic recording
│       │   ├── TranscriptViewer.tsx ← highlighted transcript with attribution
│       │   ├── SOAPNote.tsx         ← SOAP display with faithfulness badges
│       │   ├── SymptomTimeline.tsx  ← temporal visualization
│       │   ├── FaithfulnessScore.tsx← big score card at top
│       │   ├── DialogueActs.tsx     ← color-coded utterance labels
│       │   └── DifferentialDx.tsx  ← RAG-sourced differentials with evidence
│       │
│       └── lib/
│           ├── api.ts               ← typed fetch wrappers to backend
│           └── types.ts             ← TypeScript types mirroring backend schemas
│
└── models/                          ← saved model weights (gitignored)
    ├── clinical_ner/                ← fine-tuned BioBERT checkpoint
    ├── negation/                    ← negation head weights only
    ├── dialogue_act/                ← fine-tuned MiniLM checkpoint
    └── faiss_index/                 ← FAISS index files
```

---

## 5. TRAINING INSTRUCTIONS

### 5.1 Clinical NER Fine-tuning

```python
# backend/training/train_ner.py
# Run: python -m backend.training.train_ner
# Expected time: ~90 minutes on 4GB VRAM GPU
# Expected disk: 420 MB final checkpoint

from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from datasets import load_dataset
import torch

LABELS = ["O", "B-SYMPTOM", "I-SYMPTOM", "B-MEDICATION", "I-MEDICATION",
          "B-DIAGNOSIS", "I-DIAGNOSIS", "B-VITAL", "I-VITAL"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
OUTPUT_DIR = "models/clinical_ner"

# Load i2b2 2010 dataset
# You must convert i2b2 XML to HuggingFace NER format first.
# Use the conversion script in training/convert_i2b2.py (write this script).
# Expected format: datasets with columns [tokens, ner_tags]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=8,        # 4GB VRAM safe
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,        # effective batch = 32
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs/ner",
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,                            # REQUIRED for 4GB VRAM
    gradient_checkpointing=True,          # REQUIRED for 4GB VRAM
    dataloader_num_workers=2,
    report_to="none",
)

# After training, delete all checkpoints except best:
# rm -rf models/clinical_ner/checkpoint-*/
# This saves ~600 MB
```

### 5.2 Negation Scope Fine-tuning (SHARES BioBERT backbone)

```python
# backend/training/train_negation.py
# IMPORTANT: Load the NER-fine-tuned BioBERT as the base, then add negation head.
# This is transfer learning on top of your own fine-tuned model.
# Expected time: ~45 minutes (smaller dataset)
# Expected disk: 50 MB (head weights only — backbone already saved)

# The trick: save only the classification head weights, not the full model.
# At inference time, load the NER model backbone + swap heads.

from transformers import AutoModelForTokenClassification
import torch

NEG_LABELS = ["O", "B-NEG", "I-NEG", "B-UNCERTAIN", "I-UNCERTAIN"]

# Load NER backbone
ner_model = AutoModelForTokenClassification.from_pretrained("models/clinical_ner")

# Add new head for negation (do not remove NER head)
neg_head = torch.nn.Linear(ner_model.config.hidden_size, len(NEG_LABELS))
# Train only neg_head while freezing all other params:
for param in ner_model.parameters():
    param.requires_grad = False
neg_head.requires_grad = True

# Dataset: NegEx corpus from https://github.com/chapmanbe/negex
# Convert to BIO format: annotate negation trigger spans + scope spans
# Training loop: standard token classification, 3 epochs

# Save ONLY the head:
torch.save(neg_head.state_dict(), "models/negation/neg_head.pt")
# Save label map:
import json
json.dump({"labels": NEG_LABELS}, open("models/negation/config.json", "w"))
```

### 5.3 Dialogue Act Classifier

```python
# backend/training/generate_synthetic_data.py
# Run FIRST to generate training data via any LLM API you have access to.
# This generates 500 examples — takes ~5 minutes and costs <$0.50 on any API.

PROMPT = """Generate 10 realistic doctor-patient dialogue sentences.
For each, provide the label from: SYMPTOM_REPORT, QUESTION, DIAGNOSIS_STATEMENT,
TREATMENT_PLAN, REASSURANCE, HISTORY, OTHER.

Return JSON array: [{"text": "...", "label": "..."}]

Make them varied and realistic. Include medical terminology."""

# Call your preferred LLM API 50 times to get 500 examples.
# Save to data/dialogue_acts/train.json

# backend/training/train_dialogue_acts.py
# Expected time: 20 minutes on CPU (MiniLM is tiny)
# Expected disk: 80 MB

from sentence_transformers import SentenceTransformer
from torch import nn
import torch, json

LABELS = ["SYMPTOM_REPORT", "QUESTION", "DIAGNOSIS_STATEMENT",
          "TREATMENT_PLAN", "REASSURANCE", "HISTORY", "OTHER"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Freeze embedder, train only the classification head
for param in embedder.parameters():
    param.requires_grad = False

classifier = nn.Sequential(
    nn.Linear(384, 128),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128, len(LABELS))
)

# Standard cross-entropy training, 10 epochs, batch_size=32, lr=1e-3
# Save: embedder stays as-is (reuse from HuggingFace), save only classifier head
torch.save(classifier.state_dict(), "models/dialogue_act/classifier_head.pt")
json.dump({"labels": LABELS}, open("models/dialogue_act/config.json", "w"))
```

### 5.4 Ablation Study (run after all training)

```python
# backend/training/evaluate_all.py
# This is the table that will blow the faculty's mind.
# Run on 30 held-out synthetic transcripts with gold annotations.

# For each configuration, compute:
# - Entity-level F1 (precision, recall, F1 per entity type)
# - Negation accuracy (correct scope / total negated entities)
# - Dialogue act accuracy
# - SOAP faithfulness score (NLI-based)

# Print as a markdown table:
# | Component          | Baseline (regex/scispaCy) | GCIS  | Delta |
# |--------------------|--------------------------|-------|-------|
# | NER F1             | 0.61                     | 0.79  | +0.18 |
# | Negation Accuracy  | 0.52                     | 0.84  | +0.32 |
# | Dialogue Act Acc.  | N/A                      | 0.87  | —     |
# | SOAP Faithfulness  | N/A                      | 0.91  | —     |
```

---

## 6. BACKEND — FASTAPI REFACTOR

### 6.1 Entry Point

```python
# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pipeline.orchestrator import GCISOrchestrator
import uvicorn

app = FastAPI(title="GCIS API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = GCISOrchestrator()  # loads all models once at startup

class TranscriptRequest(BaseModel):
    transcript: str
    patient_name: str
    patient_age: int
    patient_id: str

@app.post("/api/process/audio")
async def process_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    result = await orchestrator.process_audio(audio_bytes)
    return result

@app.post("/api/process/transcript")
async def process_transcript(req: TranscriptRequest):
    result = await orchestrator.process_text(req.transcript, req.dict())
    return result

@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": orchestrator.models_ready}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 6.2 Orchestrator

```python
# backend/pipeline/orchestrator.py
# This is the EGV-R loop. Every stage is called in sequence.
# Max refinement iterations: 2 (prevent infinite loops)

from pipeline.extractor import ExtractionLayer
from pipeline.generator import GenerationLayer
from pipeline.verifier import VerificationLayer
from pipeline.refiner import RefinementLayer
from pipeline.transcriber import Transcriber
from schemas.soap import SOAPNote
import asyncio

FAITHFULNESS_THRESHOLD = 0.85
MAX_REFINEMENT_ITERATIONS = 2

class GCISOrchestrator:
    def __init__(self):
        self.transcriber = Transcriber()
        self.extractor = ExtractionLayer()
        self.generator = GenerationLayer()
        self.verifier = VerificationLayer()
        self.refiner = RefinementLayer(self.generator, self.verifier)
        self.models_ready = False
        self._load_models()

    def _load_models(self):
        # Load all models at startup. Log each one.
        # Do NOT lazy-load — fail fast if any model is missing.
        self.extractor.load()
        self.verifier.load()
        self.models_ready = True

    async def process_text(self, transcript: str, patient_info: dict) -> dict:
        # Stage 1: Extract
        entities = self.extractor.extract(transcript)

        # Stage 2: Generate
        soap_draft = await self.generator.generate(transcript, entities, patient_info)

        # Stage 3: Verify
        verification = self.verifier.verify(transcript, soap_draft)

        # Stage 4: Refine if needed
        final_soap = soap_draft
        iterations = 0
        while (verification.faithfulness_score < FAITHFULNESS_THRESHOLD
               and iterations < MAX_REFINEMENT_ITERATIONS):
            final_soap = await self.refiner.refine(transcript, final_soap, verification)
            verification = self.verifier.verify(transcript, final_soap)
            iterations += 1

        return {
            "transcript": transcript,
            "entities": entities.to_dict(),
            "soap": final_soap.to_dict(),
            "verification": verification.to_dict(),
            "refinement_iterations": iterations,
        }

    async def process_audio(self, audio_bytes: bytes) -> dict:
        transcript = self.transcriber.transcribe_bytes(audio_bytes)
        return await self.process_text(transcript, {})
```

### 6.3 Extractor

```python
# backend/pipeline/extractor.py
# Runs all 4 extraction models. Returns ClinicalEntities dataclass.
# ALL models run on CPU during inference.

from models.clinical_ner import ClinicalNERModel
from models.dialogue_act import DialogueActModel
from models.temporal import TemporalExtractor
from schemas.entities import ClinicalEntities

class ExtractionLayer:
    def load(self):
        self.ner = ClinicalNERModel.load("models/clinical_ner", "models/negation")
        self.dialogue = DialogueActModel.load("models/dialogue_act")
        self.temporal = TemporalExtractor()

    def extract(self, transcript: str) -> ClinicalEntities:
        sentences = self._split_sentences(transcript)

        # Run NER + negation on each sentence
        raw_entities = self.ner.extract_entities(transcript)
        negation_scopes = self.ner.detect_negation(transcript)

        # Apply negation: mark entities within negation scope as negated=True
        entities = self._apply_negation(raw_entities, negation_scopes)

        # Classify each sentence's dialogue act
        dialogue_acts = [self.dialogue.classify(s) for s in sentences]

        # Extract temporal expressions
        temporal_events = self.temporal.extract(transcript)

        return ClinicalEntities(
            symptoms=[e for e in entities if e.type == "SYMPTOM"],
            medications=[e for e in entities if e.type == "MEDICATION"],
            diagnoses=[e for e in entities if e.type == "DIAGNOSIS"],
            vitals=[e for e in entities if e.type == "VITAL"],
            negation_scopes=negation_scopes,
            dialogue_acts=dialogue_acts,
            temporal_events=temporal_events,
            sentences=sentences,
        )

    def _split_sentences(self, text: str) -> list[str]:
        import spacy
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        doc = nlp(text)
        return [str(s).strip() for s in doc.sents]

    def _apply_negation(self, entities, negation_scopes):
        # For each entity, check if its char span falls within any negation scope
        # Mark entity.negated = True if so
        for entity in entities:
            for scope in negation_scopes:
                if scope.start <= entity.start < scope.end:
                    entity.negated = True
        return entities
```

### 6.4 Generator

```python
# backend/pipeline/generator.py
# Calls Ollama with structured entities + RAG context.
# Forces JSON output via Ollama's format parameter.

import httpx, json
from rag.retriever import RAGRetriever
from schemas.soap import SOAPNote

OLLAMA_URL = "http://ollama:11434/api/generate"
MODEL = "mistral:7b-instruct-q4_K_M"

SOAP_SCHEMA = {
    "type": "object",
    "properties": {
        "subjective": {"type": "string"},
        "objective": {"type": "string"},
        "assessment": {"type": "string"},
        "plan": {"type": "string"},
        "differentials": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "diagnosis": {"type": "string"},
                    "evidence": {"type": "string"},
                    "likelihood": {"type": "string"}
                }
            }
        }
    }
}

class GenerationLayer:
    def __init__(self):
        self.rag = RAGRetriever("backend/knowledge_base")

    async def generate(self, transcript: str, entities, patient_info: dict) -> SOAPNote:
        # RAG: retrieve relevant KB docs
        query = " ".join([e.text for e in entities.symptoms + entities.diagnoses])
        retrieved_docs = self.rag.retrieve(query, top_k=3)

        prompt = self._build_prompt(transcript, entities, retrieved_docs, patient_info)

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(OLLAMA_URL, json={
                "model": MODEL,
                "prompt": prompt,
                "format": "json",   # Ollama JSON mode — enforces valid JSON output
                "stream": False,
                "options": {
                    "temperature": 0.2,    # Low temperature for clinical accuracy
                    "top_p": 0.9,
                    "num_predict": 1024,
                }
            })

        raw = response.json()["response"]
        data = json.loads(raw)
        return SOAPNote.from_dict(data, entities, retrieved_docs)

    def _build_prompt(self, transcript, entities, docs, patient_info) -> str:
        confirmed_symptoms = [e.text for e in entities.symptoms if not e.negated]
        denied_symptoms = [e.text for e in entities.symptoms if e.negated]
        medications = [e.text for e in entities.medications]
        temporal = [f"{t.text} ({t.normalized})" for t in entities.temporal_events]

        kb_context = "\n\n".join([f"[KB: {d.title}]\n{d.content[:300]}" for d in docs])

        return f"""You are a clinical documentation assistant. Generate a SOAP note.

CONFIRMED SYMPTOMS: {', '.join(confirmed_symptoms) or 'none'}
DENIED SYMPTOMS: {', '.join(denied_symptoms) or 'none'}
MEDICATIONS: {', '.join(medications) or 'none'}
TEMPORAL CONTEXT: {', '.join(temporal) or 'none'}
DIAGNOSES MENTIONED: {', '.join([e.text for e in entities.diagnoses]) or 'none'}

CLINICAL REFERENCE (use for differential diagnosis only):
{kb_context}

ORIGINAL TRANSCRIPT:
{transcript}

Generate a SOAP note as valid JSON. Every claim MUST be supported by something in the transcript.
Do not invent symptoms, medications, or diagnoses not present in the transcript.
Return ONLY valid JSON matching this structure:
{json.dumps(SOAP_SCHEMA, indent=2)}"""
```

### 6.5 Verifier

```python
# backend/pipeline/verifier.py
# NLI-based hallucination detection + sentence attribution.
# Runs on CPU. DeBERTa-v3-small is fast enough.

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from schemas.verification import VerificationResult, SentenceVerification
from models.embedder import EmbedderModel
import re

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
# Labels: CONTRADICTION=0, NEUTRAL=1, ENTAILMENT=2  (model-specific, verify this)

class VerificationLayer:
    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        self.model.eval()
        self.embedder = EmbedderModel.get_instance()  # shared MiniLM instance

    def verify(self, transcript: str, soap: "SOAPNote") -> VerificationResult:
        soap_sentences = self._extract_sentences(soap)
        transcript_sentences = transcript.split(". ")

        verified = []
        for soap_sent in soap_sentences:
            # NLI: transcript as premise, SOAP sentence as hypothesis
            nli_result = self._nli_score(transcript, soap_sent.text)

            # Attribution: find closest transcript sentence via cosine similarity
            source_sentence = self._find_source(soap_sent.text, transcript_sentences)

            verified.append(SentenceVerification(
                soap_sentence=soap_sent.text,
                soap_section=soap_sent.section,
                label=nli_result["label"],          # ENTAILED / NEUTRAL / CONTRADICTED
                confidence=nli_result["confidence"],
                source_transcript_sentence=source_sentence,
                is_hallucinated=(nli_result["label"] == "CONTRADICTED"),
            ))

        entailed = sum(1 for v in verified if v.label == "ENTAILED")
        faithfulness_score = entailed / len(verified) if verified else 0.0

        return VerificationResult(
            sentence_results=verified,
            faithfulness_score=faithfulness_score,
            hallucinated_sentences=[v for v in verified if v.is_hallucinated],
        )

    def _nli_score(self, premise: str, hypothesis: str) -> dict:
        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        # DeBERTa NLI label order: verify from model card
        # cross-encoder/nli-deberta-v3-small: 0=contradiction, 1=neutral, 2=entailment
        label_map = {0: "CONTRADICTED", 1: "NEUTRAL", 2: "ENTAILED"}
        pred = probs.argmax().item()
        return {"label": label_map[pred], "confidence": probs[pred].item()}

    def _find_source(self, soap_sentence: str, transcript_sentences: list[str]) -> str:
        soap_emb = self.embedder.encode([soap_sentence])
        transcript_embs = self.embedder.encode(transcript_sentences)
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        sims = cosine_similarity(soap_emb, transcript_embs)[0]
        return transcript_sentences[np.argmax(sims)]

    def _extract_sentences(self, soap: "SOAPNote") -> list:
        # Flatten SOAP into (section, sentence) pairs
        results = []
        for section_name, text in [
            ("subjective", soap.subjective),
            ("objective", soap.objective),
            ("assessment", soap.assessment),
            ("plan", soap.plan),
        ]:
            for sent in re.split(r'[.!?]\s+', text):
                if sent.strip():
                    results.append(type('S', (), {'text': sent, 'section': section_name})())
        return results
```

---

## 7. FRONTEND — NEXT.JS 14 MIGRATION

### 7.1 Initialize Project

```bash
cd frontend
npx create-next-app@14 . --typescript --tailwind --eslint --app --no-src-dir
# Then reorganize to match the structure in Section 4
```

### 7.2 package.json dependencies to add

```json
{
  "dependencies": {
    "next": "14.x",
    "react": "^18",
    "react-dom": "^18",
    "typescript": "^5",
    "@types/node": "^20",
    "@types/react": "^18",
    "tailwindcss": "^3",
    "recharts": "^2.8",
    "framer-motion": "^10",
    "lucide-react": "^0.380",
    "clsx": "^2",
    "@radix-ui/react-tabs": "^1",
    "@radix-ui/react-badge": "^1",
    "@radix-ui/react-progress": "^1"
  }
}
```

### 7.3 Key Components

#### TranscriptViewer.tsx
```
Purpose: Display the raw transcript with sentence-level highlighting.
- Each sentence is a <span> with a data attribute for its sentence index.
- Hovering a SOAP note bullet highlights the source sentence in amber.
- Sentences classified as SYMPTOM_REPORT → light green background
- Sentences classified as QUESTION → light blue background
- Negated entity spans → strikethrough red text
- Hover any highlighted sentence → show tooltip with dialogue act label + confidence
```

#### FaithfulnessScore.tsx
```
Purpose: Large score card at the top of the report.
- Big circular gauge (SVG, not image) showing faithfulness_score as a percentage
- Color: green if >= 0.85, amber if 0.70-0.84, red if < 0.70
- Below the gauge: "X of Y sentences verified against transcript"
- Below that: list of hallucinated sentences with a warning icon
- If refinement_iterations > 0: show "Auto-corrected X times"
```

#### SymptomTimeline.tsx
```
Purpose: Horizontal timeline of temporal events.
- Use recharts LineChart or a custom SVG timeline
- X axis: time (relative to "today" or "visit date")
- Events: dots labeled with symptom + temporal expression
  e.g.: "chest pain" at "3 days ago", "fever" at "last Tuesday"
- Color events by entity type: symptoms=coral, medications=blue
- Only render if temporal_events.length > 0
```

#### SOAPNote.tsx
```
Purpose: Display the SOAP sections with per-sentence verification badges.
- Each sentence in the SOAP note is inline with a small colored dot:
  green dot = ENTAILED, amber dot = NEUTRAL, red dot = CONTRADICTED
- Clicking a sentence highlights the source sentence in TranscriptViewer
- Hallucinated sentences have a red background + "⚠ Not found in transcript" tooltip
- Differentials section: each differential has evidence text and a likelihood badge
```

#### DialogueActs.tsx
```
Purpose: Show the dialogue act sequence as a color-coded strip.
- A horizontal strip of sentence pills: "Dr: QUESTION" | "Pt: SYMPTOM_REPORT" | ...
- Color by label: SYMPTOM_REPORT=green, QUESTION=blue, DIAGNOSIS=purple,
  TREATMENT_PLAN=teal, REASSURANCE=gray, HISTORY=amber, OTHER=light gray
- Small label + confidence percentage inside each pill
- Clicking a pill scrolls to that sentence in the transcript
```

### 7.4 API Integration

```typescript
// frontend/src/lib/api.ts
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export async function processTranscript(
  transcript: string,
  patientInfo: PatientInfo
): Promise<GCISResponse> {
  const res = await fetch(`${BACKEND_URL}/api/process/transcript`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ transcript, ...patientInfo }),
  });
  if (!res.ok) throw new Error(`Backend error: ${res.status}`);
  return res.json();
}

export async function processAudio(audioBlob: Blob): Promise<GCISResponse> {
  const form = new FormData();
  form.append("file", audioBlob, "recording.wav");
  const res = await fetch(`${BACKEND_URL}/api/process/audio`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(`Backend error: ${res.status}`);
  return res.json();
}
```

---

## 8. DOCKER COMPOSE — UPDATED

```yaml
# docker-compose.yml
version: "3.9"

services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    entrypoint: >
      /bin/sh -c "ollama serve &
      sleep 5 &&
      ollama pull mistral:7b-instruct-q4_K_M &&
      wait"

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./backend/knowledge_base:/app/knowledge_base
    environment:
      - OLLAMA_URL=http://ollama:11434
      - MODELS_DIR=/app/models
    depends_on:
      - ollama
    command: python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
    depends_on:
      - backend
    command: npm run dev

volumes:
  ollama_data:
```

---

## 9. BACKEND DOCKERFILE

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim

# Install Java for HeidelTime
RUN apt-get update && apt-get install -y \
    default-jre-headless \
    ffmpeg \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy sentencizer model (tiny, no GPU)
RUN python -m spacy download en_core_web_sm

COPY . .

# Pre-download NLI model at build time (so container starts fast)
RUN python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-small')
AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-small')
"

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 10. BACKEND REQUIREMENTS.TXT

```
fastapi==0.111.0
uvicorn[standard]==0.30.0
httpx==0.27.0
pydantic==2.7.0

# NLP
transformers==4.41.0
torch==2.3.0
accelerate==0.30.0
datasets==2.19.0
tokenizers==0.19.0
sentence-transformers==2.7.0
spacy==3.7.4
scikit-learn==1.5.0
faiss-cpu==1.8.0
numpy==1.26.4

# Temporal
py-heideltime==1.1.0

# Audio
openai-whisper==20231117
sounddevice==0.4.7

# PDF (keep existing functionality)
reportlab==4.2.0

# Utils
python-multipart==0.0.9
aiofiles==23.2.1
```

---

## 11. .ENV.EXAMPLE — UPDATED

```bash
# .env.example
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b-instruct-q4_K_M
MODELS_DIR=./models
KNOWLEDGE_BASE_DIR=./backend/knowledge_base
FAITHFULNESS_THRESHOLD=0.85
MAX_REFINEMENT_ITERATIONS=2
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

---

## 12. EXECUTION ORDER FOR HERMES

Execute in this exact order. Do not parallelize. Each step depends on the previous.

```
PHASE 1 — REPOSITORY RESTRUCTURE
  [ ] Clone repo
  [ ] Create new directory structure from Section 4
  [ ] Move existing files to new locations:
        nlp_engine.py        → backend/models/clinical_ner.py (refactor class)
        soap_builder.py      → backend/schemas/soap.py + backend/pipeline/generator.py (split)
        pdf_builder.py       → backend/pipeline/pdf_builder.py (keep, minor refactor)
        recorder.py          → backend/pipeline/transcriber.py (merge with transcriber.py)
        transcriber.py       → merge into backend/pipeline/transcriber.py
        knowledge_base/      → backend/knowledge_base/
        streamlit_app/       → DELETE (replaced by Next.js)
        app.py               → DELETE (replaced by FastAPI)

PHASE 2 — BACKEND IMPLEMENTATION
  [ ] Write backend/schemas/ (entities.py, soap.py, verification.py)
  [ ] Write backend/models/ (clinical_ner.py, dialogue_act.py, temporal.py, nli.py, embedder.py)
  [ ] Write backend/rag/ (indexer.py, retriever.py)
  [ ] Write backend/pipeline/ (orchestrator.py, extractor.py, generator.py, verifier.py, refiner.py)
  [ ] Write backend/main.py
  [ ] Write backend/requirements.txt
  [ ] Write backend/Dockerfile

PHASE 3 — TRAINING DATA + MODEL TRAINING
  [ ] Write training/generate_synthetic_data.py and run it
  [ ] Write training/convert_i2b2.py (i2b2 XML → HuggingFace format)
  [ ] Download i2b2 2010 dataset (requires free academic registration)
  [ ] Download NegEx corpus from GitHub
  [ ] Run train_ner.py — verify F1 > 0.70 before proceeding
  [ ] Run train_negation.py — verify negation accuracy > 0.75
  [ ] Run train_dialogue_acts.py — verify accuracy > 0.80
  [ ] Build FAISS index (run rag/indexer.py)
  [ ] Run evaluate_all.py — save the results table

PHASE 4 — FRONTEND IMPLEMENTATION
  [ ] Initialize Next.js 14 project in frontend/
  [ ] Write frontend/src/lib/types.ts
  [ ] Write frontend/src/lib/api.ts
  [ ] Write all components from Section 7.3
  [ ] Write frontend/src/app/page.tsx (main layout)
  [ ] Write frontend/Dockerfile

PHASE 5 — INTEGRATION
  [ ] Update docker-compose.yml
  [ ] Update .env.example
  [ ] Test: docker-compose up --build
  [ ] Test: POST /api/process/transcript with sample transcript
  [ ] Test: Verify faithfulness score appears in frontend
  [ ] Test: Verify attribution highlighting works (hover SOAP → highlight transcript)

PHASE 6 — EVALUATION + DOCUMENTATION
  [ ] Run evaluate_all.py on 30 held-out examples
  [ ] Save ablation table to README.md
  [ ] Rewrite README.md with:
        - Architecture diagram (ASCII or Mermaid)
        - Model cards (what each model does, size, training data)
        - Ablation results table
        - Setup instructions
  [ ] Delete all training checkpoints except best (save ~600MB)
```

---

## 13. THINGS HERMES MUST NOT DO

```
✗ Do not use GPT-4 API or any external API for inference (only for synthetic data generation)
✗ Do not use any model larger than 4GB on GPU
✗ Do not load two GPU models at the same time
✗ Do not use the OpenAI Whisper API — use the local whisper library
✗ Do not use LangChain (write the RAG pipeline directly — fewer abstractions = cleaner code)
✗ Do not use any paid datasets
✗ Do not keep training checkpoints after best model is saved
✗ Do not use Streamlit for anything — it is completely replaced by Next.js
✗ Do not use any CSS framework other than Tailwind
✗ Do not use class components in React — hooks only
✗ Do not hallucinate library APIs — if unsure, check the library's documentation first
✗ Do not skip the ablation study — it is the most important academic deliverable
```

---

## 14. SUCCESS CRITERIA

The refactor is complete when ALL of the following are true:

```
[ ] docker-compose up --build completes without errors
[ ] POST /api/process/transcript returns a response with:
      - entities (symptoms, medications, diagnoses, vitals)
      - soap (subjective, objective, assessment, plan, differentials)
      - verification (faithfulness_score, sentence_results, hallucinated_sentences)
      - refinement_iterations (integer 0-2)
[ ] Next.js frontend renders at http://localhost:3000
[ ] FaithfulnessScore gauge shows correct percentage
[ ] SymptomTimeline renders for transcripts with temporal expressions
[ ] Hovering a SOAP bullet highlights the source transcript sentence
[ ] Dialogue acts are color-coded correctly in the transcript strip
[ ] evaluate_all.py produces an ablation table with all 4 metrics
[ ] Total disk usage of models/ directory is under 6GB
[ ] Backend startup time is under 30 seconds (models pre-loaded at init)
[ ] /api/process/transcript responds in under 60 seconds end-to-end
```
