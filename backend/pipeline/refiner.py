"""
Stage 4: Refinement Layer.
Self-correction loop: re-prompts LLM to fix hallucinated sentences.
Max 2 refinement iterations.
"""
import json
import httpx
import os
from typing import List

from schemas.soap import SOAPNote
from schemas.verification import VerificationResult

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b-instruct")


def _build_refinement_prompt_template():
    lines = [
        "You are revising a SOAP note. The following sentence(s) were flagged as NOT "
        "supported by the original transcript:",
        "",
        "FLAGGED SENTENCES:",
        "{flagged}",
        "",
        "ORIGINAL TRANSCRIPT:",
        "{transcript}",
        "",
        "Please revise ONLY the flagged sentences so they are fully supported by the transcript. "
        "Do not change any other sentences. Return the REVISED sections as valid JSON "
        "with keys matching the section names (subjective, objective, assessment, plan).",
    ]
    return "\n".join(lines)


REFINEMENT_PROMPT_TEMPLATE = _build_refinement_prompt_template()


class RefinementLayer:
    """Refines hallucinated SOAP sentences via LLM self-correction."""

    def __init__(self, generator, verifier):
        self.generator = generator
        self.verifier = verifier

    async def refine(self, transcript: str, current_soap: SOAPNote,
                     verification: VerificationResult) -> SOAPNote:
        """Refine flagged sentences and return improved SOAP note."""
        flagged = verification.hallucinated_sentences
        if not flagged:
            return current_soap

        flagged_text = "\n".join([
            f"- [{s.soap_section}] {s.soap_sentence}" for s in flagged
        ])

        prompt = REFINEMENT_PROMPT_TEMPLATE.format(
            flagged=flagged_text,
            transcript=transcript,
        )

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(OLLAMA_URL, json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 512,
                    },
                })
                response.raise_for_status()
                raw = response.json()["response"]

                # Try to parse JSON
                revised = json.loads(raw)

                # Merge revised sections into current SOAP
                for section in ["subjective", "objective", "assessment", "plan"]:
                    if section in revised:
                        setattr(current_soap, section, revised[section])

        except Exception as e:
            print(f"[RefinementLayer] Refinement error: {e}, keeping current SOAP")

        return current_soap
