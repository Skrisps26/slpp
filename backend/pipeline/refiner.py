"""
Stage 4: Refinement Layer.
Self-correction loop: re-prompts LLM to fix hallucinated sentences.
Max 2 refinement iterations.
"""
import httpx
import os

from schemas.soap import SOAPNote
from schemas.verification import VerificationResult

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b-instruct")


class RefinementLayer:
    """Refines hallucinated SOAP sentences via LLM self-correction."""

    def __init__(self, generator, verifier):
        self.generator = generator
        self.verifier = verifier

    async def refine(self, transcript: str, soap: SOAPNote,
                     verification: VerificationResult) -> SOAPNote:
        """
        For each hallucinated sentence: re-prompt LLM to rewrite it.
        Replace hallucinated sentences in-place. Return patched SOAPNote.
        """
        soap_dict = {
            "subjective": soap.subjective,
            "objective": soap.objective,
            "assessment": soap.assessment,
            "plan": soap.plan,
        }

        for bad in verification.hallucinated_sentences:
            section = bad.soap_section
            current_text = soap_dict.get(section, "")

            if bad.soap_sentence not in current_text:
                continue   # already replaced in a previous pass

            rewritten = await self._rewrite_sentence(
                transcript=transcript,
                soap_section=section,
                bad_sentence=bad.soap_sentence,
                full_section_text=current_text,
            )

            if rewritten:
                soap_dict[section] = current_text.replace(bad.soap_sentence, rewritten, 1)
            else:
                # LLM said nothing supports this sentence — remove it
                soap_dict[section] = current_text.replace(bad.soap_sentence, "", 1).strip()

        return SOAPNote(
            subjective=soap_dict["subjective"],
            objective=soap_dict["objective"],
            assessment=soap_dict["assessment"],
            plan=soap_dict["plan"],
            differentials=soap.differentials,   # differentials not re-verified here
        )

    async def _rewrite_sentence(self, transcript: str, soap_section: str,
                                 bad_sentence: str, full_section_text: str) -> str:
        """Re-prompt LLM to rewrite a hallucinated sentence."""
        prompt = f"""You are a clinical documentation assistant performing a correction task.

The following sentence in the {soap_section.upper()} section of a SOAP note is NOT supported
by the patient-doctor transcript. You must rewrite it using only information from the transcript,
or return an empty string if nothing in the transcript supports it.

TRANSCRIPT:
{transcript}

PROBLEMATIC SENTENCE (not in transcript):
"{bad_sentence}"

SECTION CONTEXT:
{full_section_text}

Rules:
1. Use ONLY information present in the transcript above.
2. Maintain clinical documentation tone.
3. If nothing in the transcript supports this sentence, return exactly: REMOVE
4. Return ONLY the rewritten sentence or REMOVE. No explanation, no preamble.

REWRITTEN:"""

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(OLLAMA_URL, json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 150},
            })

        raw = resp.json().get("response", "").strip().strip('"').strip("'")
        if raw.upper() == "REMOVE" or not raw:
            return ""
        return raw
