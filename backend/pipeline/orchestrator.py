"""
GCIS Orchestrator — EGV-R loop coordinator with SSE streaming.
Extract → Generate → Verify → Refine
"""
from typing import AsyncGenerator

from pipeline.transcriber import Transcriber
from pipeline.extractor import ExtractionLayer
from pipeline.generator import GenerationLayer
from pipeline.verifier import VerificationLayer
from pipeline.refiner import RefinementLayer
from schemas.response import GCISResponse

FAITHFULNESS_THRESHOLD = 0.85
MAX_REFINEMENT_ITERATIONS = 2


class GCISOrchestrator:
    """Main orchestrator for the EGV-R pipeline."""

    def __init__(self):
        self.transcriber = Transcriber()
        self.extractor = ExtractionLayer()
        self.generator = GenerationLayer()
        self.verifier = VerificationLayer()
        self.refiner = RefinementLayer(self.generator, self.verifier)
        self.models_ready = False
        self._load_models()

    def _load_models(self):
        """Load all models at startup."""
        print("[Orchestrator] Loading models...")
        self.transcriber.load()
        self.extractor.load()
        self.verifier.load()
        self.models_ready = True
        print("[Orchestrator] All models loaded.")

    async def process_text_streaming(
        self, transcript: str, patient_info: dict
    ) -> AsyncGenerator[dict, None]:
        """Process a text transcript with SSE progress events."""
        yield {"stage": "extracting", "progress": 20,
               "message": "Extracting clinical entities..."}
        entities = self.extractor.extract(transcript)

        yield {"stage": "retrieving", "progress": 40,
               "message": "Searching clinical knowledge base..."}

        yield {"stage": "generating", "progress": 55,
               "message": "Generating SOAP note with LLM..."}
        soap_draft = await self.generator.generate(transcript, entities, patient_info)

        yield {"stage": "verifying", "progress": 80,
               "message": "Verifying every sentence against transcript..."}
        verification = self.verifier.verify(transcript, soap_draft)

        final_soap = soap_draft
        iterations = 0
        while (verification.faithfulness_score < FAITHFULNESS_THRESHOLD
               and iterations < MAX_REFINEMENT_ITERATIONS):
            iterations += 1
            yield {
                "stage": "refining",
                "progress": 85 + iterations * 5,
                "message": f"Auto-correcting hallucinations (pass {iterations} of {MAX_REFINEMENT_ITERATIONS})...",
                "iteration": iterations,
                "hallucinated_count": len(verification.hallucinated_sentences),
            }
            final_soap = await self.refiner.refine(transcript, final_soap, verification)
            verification = self.verifier.verify(transcript, final_soap)

        result = GCISResponse(
            transcript=transcript,
            patient_info=patient_info,
            entities=entities.to_dict(),
            soap=final_soap.to_dict(),
            verification=verification.to_dict(),
            refinement_iterations=iterations,
        )
        yield {"stage": "complete", "progress": 100,
               "message": "Done.", "result": result.to_dict()}

    async def process_audio_streaming(
        self, audio_bytes: bytes
    ) -> AsyncGenerator[dict, None]:
        """Process audio with SSE progress events."""
        yield {"stage": "transcribing", "progress": 5,
               "message": "Transcribing audio with Whisper..."}
        transcript = self.transcriber.transcribe_bytes(audio_bytes)
        yield {"stage": "transcribed", "progress": 15,
               "message": "Transcription complete.", "transcript": transcript}
        async for event in self.process_text_streaming(transcript, {}):
            yield event
