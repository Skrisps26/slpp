"""
FastAPI main with SSE streaming endpoints and PDF export.
"""
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from pipeline.orchestrator import GCISOrchestrator
from pipeline.pdf_exporter import PDFExporter

app = FastAPI(title="GCIS API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = GCISOrchestrator()
pdf_exporter = PDFExporter()


class TranscriptRequest(BaseModel):
    transcript: str
    patient_name: str = ""
    patient_age: int = 0
    patient_id: str = ""


@app.post("/api/process/transcript")
async def process_transcript(req: TranscriptRequest):
    async def event_stream():
        async for event in orchestrator.process_text_streaming(req.transcript, req.dict()):
            yield f"data: {json.dumps(event)}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/process/audio")
async def process_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    async def event_stream():
        async for event in orchestrator.process_audio_streaming(audio_bytes):
            yield f"data: {json.dumps(event)}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/export/pdf")
async def export_pdf(payload: dict):
    """Accepts full GCISResponse dict, returns PDF binary."""
    from schemas.soap import SOAPNote, Differential
    from schemas.verification import VerificationResult, SentenceVerification

    sd = payload["soap"]
    soap = SOAPNote(
        subjective=sd["subjective"], objective=sd["objective"],
        assessment=sd["assessment"], plan=sd["plan"],
        differentials=[Differential(**d) for d in sd.get("differentials", [])],
    )

    vd = payload["verification"]
    verification = VerificationResult(
        sentence_results=[SentenceVerification(**s) for s in vd["sentence_results"]],
        faithfulness_score=vd["faithfulness_score"],
        hallucinated_sentences=[SentenceVerification(**s) for s in vd["hallucinated_sentences"]],
    )

    pdf_bytes = pdf_exporter.export(
        soap, verification, payload.get("patient_info", {}), payload.get("transcript", "")
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=soap_note.pdf"},
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": orchestrator.models_ready}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
