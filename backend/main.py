"""
GCIS API — FastAPI entry point
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from pipeline.orchestrator import GCISOrchestrator
import uvicorn
import os
import sys

# Add backend to path so relative imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

app = FastAPI(title="GCIS API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


class TranscriptRequest(BaseModel):
    transcript: str
    patient_name: str = ""
    patient_age: int = 0
    patient_id: str = ""

    model_config = {"coerce_numbers_to_str": True, "strict": False}

    @field_validator("patient_age", mode="before")
    @classmethod
    def coerce_patage(cls, v):
        if isinstance(v, str):
            try:
                return int(v)
            except (ValueError, TypeError):
                return 0
        return v or 0


# Global orchestrator (lazy-loaded)
orchestrator = None


def get_orchestrator() -> GCISOrchestrator:
    global orchestrator
    if orchestrator is None:
        orchestrator = GCISOrchestrator()
    return orchestrator


@app.post("/api/process/audio")
async def process_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    orch = get_orchestrator()
    result = await orch.process_audio(audio_bytes)
    return result


@app.post("/api/process/transcript")
async def process_transcript(req: TranscriptRequest):
    orch = get_orchestrator()
    patient_info = {
        "patient_name": req.patient_name,
        "patient_age": req.patient_age,
        "patient_id": req.patient_id,
    }
    result = await orch.process_text(req.transcript, patient_info)
    return result


@app.get("/api/health")
def health():
    global orchestrator
    return {
        "status": "ok",
        "models_loaded": orchestrator is not None and orchestrator.models_ready,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
