"""
Stage 2: Generation Layer.
Calls Ollama LLM with structured entities + RAG context.
Returns a structured SOAP note in JSON format.
"""
import json
import os
import httpx

from schemas.soap import SOAPNote
from rag.retriever import RAGRetriever

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b-instruct")

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
                    "likelihood": {"type": "string"},
                },
            },
        },
    },
    "required": ["subjective", "objective", "assessment", "plan"],
}


def _build_prompt_template():
    lines = [
        "You are a clinical documentation assistant. Generate a SOAP note.",
        "CRITICAL RULES:",
        "1. Symptoms marked as DENIED must NEVER appear as present symptoms.",
        "2. Use 'Patient denies X' or 'No history of X' for denied items.",
        "3. Every claim MUST be supported by the transcript.",
        "4. Return ONLY valid JSON matching the required structure.",
        "",
        "CONFIRMED SYMPTOMS (actually present): {symptoms}",
        "DENIED SYMPTOMS (NOT present): {denied}",
        "MEDICATIONS: {medications}",
        "TEMPORAL CONTEXT: {temporal}",
        "DIAGNOSES MENTIONED: {diagnoses}",
        "",
        "CLINICAL REFERENCE (use for differential diagnosis only):",
        "{kb_context}",
        "",
        "ORIGINAL TRANSCRIPT:",
        "{transcript}",
        "",
        "Generate a SOAP note as valid JSON. Every claim MUST be supported by the transcript.",
        "Return ONLY valid JSON matching this structure:",
        "{schema}",
    ]
    return "\n".join(lines)


PROMPT_TEMPLATE = _build_prompt_template()


class GenerationLayer:
    """Generates SOAP notes using Ollama LLM + RAG context."""

    def __init__(self):
        self.rag = RAGRetriever()

    async def generate(self, transcript, entities, patient_info: dict) -> SOAPNote:
        """Generate a SOAP note from structured entities and transcript."""
        # Build query from confirmed symptoms + diagnoses
        query_parts = [e.text for e in entities.confirmed_symptoms()]
        query_parts += [e.text for e in entities.diagnoses]
        query = " ".join(query_parts) if query_parts else transcript[:200]

        retrieved_docs = self.rag.retrieve(query, top_k=3)
        prompt = self._build_prompt(transcript, entities, retrieved_docs)

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(OLLAMA_URL, json={
                    "model": MODEL,
                    "prompt": prompt,
                    "format": SOAP_SCHEMA,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "num_predict": 1024,
                    },
                })
                response.raise_for_status()
                raw = response.json()
                if "message" in raw:
                    data = json.loads(raw["message"]["content"])
                elif "response" in raw:
                    data = json.loads(raw["response"])
                else:
                    data = raw
        except Exception as e:
            print(f"[GenerationLayer] Ollama error: {e}, using fallback SOAP note")
            data = self._fallback_soap(entities, transcript)

        return SOAPNote.from_dict(data, entities, retrieved_docs)

    def _build_prompt(self, transcript, entities, docs) -> str:
        confirmed = [e.text for e in entities.confirmed_symptoms()]
        denied = [e.text for e in entities.denied_symptoms()]
        meds = [e.text for e in entities.medications]
        temporal = [f"{t.text} ({t.normalized})" for t in entities.temporal_events]
        diag = [e.text for e in entities.diagnoses]
        kb_context = "\n\n".join([f"[KB: {d.title}]\n{d.content[:300]}" for d in docs])

        return PROMPT_TEMPLATE.format(
            symptoms=", ".join(confirmed) or "none",
            denied=", ".join(denied) or "none",
            medications=", ".join(meds) or "none",
            temporal=", ".join(temporal) or "none",
            diagnoses=", ".join(diag) or "none",
            kb_context=kb_context,
            transcript=transcript,
            schema=json.dumps(SOAP_SCHEMA, indent=2),
        )

    @staticmethod
    def _fallback_soap(entities, transcript: str) -> dict:
        symptoms = [e.text for e in entities.confirmed_symptoms()]
        denied = [e.text for e in entities.denied_symptoms()]
        medications = [e.text for e in entities.medications]
        return {
            "subjective": f"Patient reports: {', '.join(symptoms)}. "
                          f"Denies: {', '.join(denied)}. Medications: {', '.join(medications)}.",
            "objective": f"Transcript reviewed. Key mentions: {', '.join(symptoms)}.",
            "assessment": f"Assessment pending based on patient-reported symptoms.",
            "plan": "Further evaluation recommended. Follow-up as clinically indicated.",
            "differentials": [],
        }
