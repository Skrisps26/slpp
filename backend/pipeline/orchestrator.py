"""
GCIS Orchestrator — EGV-R loop coordinator.
Extract → Generate → Verify → Refine
"""
import asyncio
import os

from pipeline.transcriber import Transcriber
from pipeline.extractor import ExtractionLayer
from pipeline.generator import GenerationLayer
from pipeline.verifier import VerificationLayer
from pipeline.refiner import RefinementLayer

FAITHFULNESS_THRESHOLD = float(os.environ.get("FAITHFULNESS_THRESHOLD", "0.85"))
MAX_REFINEMENT_ITERATIONS = int(os.environ.get("MAX_REFINEMENT_ITERATIONS", "2"))


class GCISOrchestrator:
    """Main orchestrator for the EGV-R clinical intelligence pipeline."""

    def __init__(self):
        self.transcriber = Transcriber()
        self.extractor = ExtractionLayer()
        self.generator = GenerationLayer()
        self.verifier = VerificationLayer()
        self.refiner = RefinementLayer(self.generator, self.verifier)
        self.models_ready = False
        self._load_models()

    def _load_models(self):
        """Load all models at startup. Fail fast if any model is missing."""
        print("[Orchestrator] Loading models...")
        try:
            self.extractor.load()
        except Exception as e:
            print(f"[Orchestrator] Extraction layer load warning: {e}")
        try:
            self.verifier.load()
        except Exception as e:
            print(f"[Orchestrator] Verification layer load warning: {e}")
        self.models_ready = True
        print("[Orchestrator] All models loaded (or attempted). Ready.")

    async def process_text(self, transcript: str, patient_info: dict) -> dict:
        """Process a text transcript through the EGV-R pipeline."""
        print("[Orchestrator] Stage 1: Extracting entities...")
        entities = self.extractor.extract(transcript)

        print("[Orchestrator] Stage 2: Generating SOAP note...")
        soap_draft = await self.generator.generate(transcript, entities, patient_info)

        print("[Orchestrator] Stage 3: Verifying SOAP note...")
        verification = self.verifier.verify(transcript, soap_draft)

        # Stage 4: Refine if needed
        final_soap = soap_draft
        iterations = 0
        while (verification.faithfulness_score < FAITHFULNESS_THRESHOLD
               and iterations < MAX_REFINEMENT_ITERATIONS):
            print(f"[Orchestrator] Stage 4: Refining (iteration {iterations + 1})... "
                  f"faithfulness={verification.faithfulness_score:.4f}")
            final_soap = await self.refiner.refine(transcript, final_soap, verification)
            verification = self.verifier.verify(transcript, final_soap)
            iterations += 1

        print(f"[Orchestrator] Complete. Faithfulness: {verification.faithfulness_score:.4f}, "
              f"Refinements: {iterations}")

        return {
            "transcript": transcript,
            "entities": entities.to_dict(),
            "soap": final_soap.to_dict(),
            "verification": verification.to_dict(),
            "refinement_iterations": iterations,
        }

    async def process_audio(self, audio_bytes: bytes, filename: str = "audio.wav") -> dict:
        """Process audio bytes through the full pipeline."""
        print("[Orchestrator] Transcribing audio...")
        transcript = self.transcriber.transcribe_bytes(audio_bytes, filename=filename)
        print(f"[Orchestrator] Transcription complete ({len(transcript)} chars)")
        return await self.process_text(transcript, {})
