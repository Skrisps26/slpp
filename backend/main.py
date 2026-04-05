"""
GCIS API — FastAPI entry point
"""
import asyncio
import io
import json
import os
import sys
from pydantic import BaseModel, field_validator
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pipeline.orchestrator import GCISOrchestrator
import uvicorn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="GCIS API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscriptRequest(BaseModel):
    transcript: str
    patient_name: str = ""
    patient_age: int = 0
    patient_id: str = ""

    model_config = {"strict": False}

    @field_validator("patient_age", mode="before")
    @classmethod
    def coerce_age(cls, v):
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
async def process_audio(
    file: UploadFile = File(...),
    patient_info: str = Form(default="{}"),
):
    """Process audio file (.wav) through the full EGV-R pipeline."""
    # Validate file extension
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".wav", ".mp3", ".mp4", ".m4a", ".ogg", ".flac"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Please upload .wav, .mp3, .mp4, .m4a, .ogg, or .flac.",
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    orch = get_orchestrator()
    result = await orch.process_audio(audio_bytes, filename=filename)
    return result


@app.post("/api/process/transcript")
async def process_transcript(req: TranscriptRequest):
    """Process text transcript through the full EGV-R pipeline."""
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
