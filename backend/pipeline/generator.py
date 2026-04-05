"""
Stage 2: Generation Layer.
Calls Ollama LLM with structured entities + RAG context.
Returns a structured SOAP note in JSON format.
"""
import json
import os
import httpx
from typing import Dict, Any

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
    "required": ["subjective", "objective", "assessment", "plan", "differentials"],
}


def _build_prompt_template():
    lines = [
        "You are a clinical documentation assistant. Generate a SOAP note.",
        "CRITICAL RULES:",
        "1. Symptoms marked as DENIED must NEVER appear as present in your note.",
        "2. Use 'Patient denies X' or 'No history of X' for denied symptoms.",
        "3. Do not add any symptom, medication, or diagnosis not explicitly in the transcript.",
        "4. Return ONLY valid JSON matching the required structure.",
        "",
        "CONFIRMED SYMPTOMS (actually present): {symptoms}",
        "DENIED SYMPTOMS (NOT present - do NOT list as present!): {denied}",
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
        self.rag = RAGRetriever("backend/knowledge_base")

    async def generate(self, transcript, entities, patient_info: Dict) -> SOAPNote:
        """Generate a SOAP note from structured entities and transcript."""
        query_parts = [e.text for e in entities.symptoms] + [e.text for e in entities.diagnoses]
        query = " ".join(query_parts) if query_parts else transcript[:200]
        retrieved_docs = self.rag.retrieve(query, top_k=3)
        prompt = self._build_prompt(transcript, entities, retrieved_docs, patient_info)

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
                raw = response.json()["response"]
                data = json.loads(raw)
        except Exception as e:
            print(f"[GenerationLayer] Ollama error: {e}, using fallback SOAP note")
            data = self._fallback_soap(entities, transcript)

        return SOAPNote.from_dict(data, entities, retrieved_docs)

    def _build_prompt(self, transcript, entities, docs, patient_info: Dict) -> str:
        confirmed = [e.text for e in entities.symptoms if not e.negated]
        denied = [e.text for e in entities.symptoms if e.negated]
        meds = [e.text for e in entities.medications]
        temporal = [f"{t.text} ({t.normalized})" for t in entities.temporal_events]
        diag = [e.text for e in entities.diagnoses]
        kb_context = "\n\n".join([f"[KB: {d.get('title', d.get('__title__', ''))}]\n{d.get('content', d.get('__content__', ''))[:300]}" for d in docs])

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
        symptoms = [e.text for e in entities.symptoms if not e.negated]
        medications = [e.text for e in entities.medications]
        diagnoses = [e.text for e in entities.diagnoses]
        return {
            "subjective": f"Patient reports: {', '.join(symptoms)}. Medications: {', '.join(medications)}.",
            "objective": f"Transcript reviewed. Key mentions: {', '.join(symptoms + diagnoses)}.",
            "assessment": f"Assessment pending based on patient-reported symptoms: {', '.join(symptoms or ['not specified'])}.",
            "plan": "Further evaluation recommended. Follow-up as clinically indicated.",
            "differentials": [
                {"diagnosis": d, "evidence": "Mentioned in transcript", "likelihood": "Moderate"}
                for d in (diagnoses or ["Pending evaluation"])
            ],
        }
