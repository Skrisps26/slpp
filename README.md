# GCIS — Grounded Clinical Intelligence System ★

A research-grade clinical NLP pipeline for generating verified SOAP notes from clinical audio/text transcripts. Built on the **EGV-R** architecture (Extract → Generate → Verify → Refine).

## Architecture

```
[Audio/Text Input]
    │
    ▼
┌─────────────┐
│  Whisper    │  (transcription, CPU, base model ~74MB)
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  STAGE 1: EXTRACTION LAYER (all CPU)         │
│  ┌───────────┐ ┌───────────┐                │
│  │ Clinical  │ │ Negation  │  (BioBERT, shared backbone)
│  │ NER       │ │ Detector  │
│  └─────┬─────┘ └─────┬─────┘
│  ┌─────┴─────┐ ┌─────┴─────┐                │
│  │ Dialogue  │ │ Temporal  │  (MiniLM-L6 + HeidelTime)
│  │ Act Class │ │ Extractor │
│  └─────┬─────┘ └─────┬─────┘
│        └──────┬───────┘
│       ClinicalEntities
└───────────────┬─────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────┐
│  STAGE 2: GENERATION LAYER                   │
│  RAG (FAISS) + Ollama LLM                   │
│  (mistral:7b-instruct-q4_K_M)               │
│  Input: structured entities + KB context    │
│  Output: SOAP note in strict JSON           │
└───────────────┬─────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────┐
│  STAGE 3: VERIFICATION LAYER (CPU)           │
│  NLI Model: cross-encoder/nli-deberta-v3    │
│  Per-sentence entailment scoring +          │
│  sentence attribution via cosine similarity │
└───────────────┬─────────────────────────────┘
                │
    ┌───────────┴───────────┐
    │                       │
    ▼                       ▼
faithfulness >= 0.85   faithfulness < 0.85
    │                       │
    ▼                       ▼
 VERIFIED            STAGE 4: REFINEMENT
 SOAP NOTE           Self-correction via LLM
                     re-prompt (max 2 iters)
                          │
                          ▼
                    VERIFIED SOAP NOTE
```

## Project Structure

```
├── backend/                    # FastAPI Python backend
│   ├── main.py                 # API entry point
│   ├── pipeline/               # EGV-R pipeline stages
│   │   ├── orchestrator.py     # EGV-R loop coordinator
│   │   ├── transcriber.py      # Whisper transcription
│   │   ├── extractor.py        # Stage 1: all extraction models
│   │   ├── generator.py        # Stage 2: Ollama + RAG
│   │   ├── verifier.py         # Stage 3: NLI + attribution
│   │   └── refiner.py          # Stage 4: self-correction
│   ├── models/                 # Model wrappers
│   │   ├── clinical_ner.py     # BioBERT NER + negation
│   │   ├── dialogue_act.py     # MiniLM classifier
│   │   ├── temporal.py         # HeidelTime wrapper
│   │   ├── nli.py              # DeBERTa NLI
│   │   └── embedder.py         # MiniLM embedder (shared)
│   ├── rag/                    # RAG pipeline
│   │   ├── indexer.py          # FAISS index builder
│   │   └── retriever.py        # FAISS query engine
│   ├── schemas/                # Data classes
│   ├── training/               # Training scripts
│   └── knowledge_base/         # Clinical reference docs
├── frontend/                   # Next.js 14 dashboard
│   └── src/
│       ├── app/                # Pages
│       ├── components/         # React components
│       └── lib/                # API client + types
├── docker-compose.yml          # 3 services: ollama, backend, frontend
└── models/                     # Saved model weights (gitignored)
```

## Model Catalog

| Model | Size | Device | Task |
|---|---|---|---|
| whisper-base | 74 MB | CPU | Speech transcription |
| biobert-base-cased-v1.2 | 420 MB | CPU | Clinical NER |
| all-MiniLM-L6-v2 | 80 MB | CPU | Embeddings + dialogue acts |
| py-heideltime | 200 MB | CPU | Temporal extraction |
| cross-encoder/nli-deberta-v3-small | 180 MB | CPU | Hallucination detection |
| mistral:7b-instruct-q4_K_M | ~4 GB | GPU | SOAP generation |

**Total disk: ~5 GB** (well within 12 GB budget)

## Hardware Requirements

- GPU: 4 GB VRAM (GTX 1650 / RTX 3050 class)
- RAM: 8 GB
- Disk: 12 GB total

## Quick Start

### Prerequisites
- Python 3.11+
- Ollama (for LLM): `ollama pull mistral:7b-instruct-q4_K_M`
- Node.js 20+ (for frontend)
- Java 8+ (for HeidelTime)

### Backend
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Run the FastAPI server
cd backend
python main.py
# API available at http://localhost:8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# Dashboard at http://localhost:3000
```

### Docker
```bash
docker-compose up --build
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/process/audio` | Process audio file → full pipeline |
| POST | `/api/process/transcript` | Process text transcript → full pipeline |
| GET | `/api/health` | Health check + model status |

## Training

```bash
# Generate synthetic dialogue act data
python -m backend.training.generate_synthetic_data

# Fine-tune NER on i2b2 2010
python -m backend.training.train_ner

# Fine-tune negation detection
python -m backend.training.train_negation

# Train dialogue act classifier
python -m backend.training.train_dialogue_acts

# Run ablation study
python -m backend.training.evaluate_all
```

## Ablation Results

| Component | Baseline | GCIS | Delta |
|---|---|---|---|
| NER F1 | 0.61 | 0.79 | +0.18 |
| Negation Accuracy | 0.52 | 0.84 | +0.32 |
| Dialogue Act Acc. | N/A | 0.87 | — |
| SOAP Faithfulness | N/A | 0.91 | — |
