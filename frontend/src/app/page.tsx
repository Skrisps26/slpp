"use client";
import { useState, useCallback } from "react";
import type { GCISResponse, PatientInfo } from "@/lib/types";
import { processTranscriptStream, SSEEvent } from "@/lib/api";
import { PipelineProgress } from "@/components/PipelineProgress";
import { FaithfulnessGauge } from "@/components/FaithfulnessGauge";
import { TranscriptViewer } from "@/components/TranscriptViewer";
import { SOAPDisplay } from "@/components/SOAPDisplay";
import { SymptomTimeline } from "@/components/SymptomTimeline";
import { DialogueActStrip } from "@/components/DialogueActStrip";
import { DifferentialDx } from "@/components/DifferentialDx";
import { DeniedSymptoms } from "@/components/DeniedSymptoms";
import { ExportBar } from "@/components/ExportBar";

type View = "input" | "processing" | "results";

export default function Home() {
  const [view, setView] = useState<View>("input");
  const [transcript, setTranscript] = useState("");
  const [patientName, setPatientName] = useState("");
  const [patientAge, setPatientAge] = useState("");
  const [patientId, setPatientId] = useState("");
  const [sseEvent, setSseEvent] = useState<SSEEvent | null>(null);
  const [transcriptPreview, setTranscriptPreview] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<GCISResponse | null>(null);
  const isInputReady = transcript.trim().length > 0;

  const handleProcess = useCallback(async () => {
    if (!isInputReady) return;
    setView("processing");
    setSseEvent(null);
    setTranscriptPreview("");
    setError(null);
    setResult(null);

    try {
      const patientInfo: PatientInfo = {
        patient_name: patientName,
        patient_age: parseInt(patientAge) || 0,
        patient_id: patientId,
      };

      const final = await processTranscriptStream(transcript, patientInfo, (event) => {
        setSseEvent(event);
        if (event.transcript) setTranscriptPreview(event.transcript);
      });

      if (final) {
        setResult(final);
        setView("results");
      } else {
        setError("No result returned from backend.");
        setView("input");
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error occurred");
      setView("input");
    }
  }, [transcript, patientName, patientAge, patientId, isInputReady]);

  const handleReset = useCallback(() => {
    setView("input");
    setTranscript("");
    setSseEvent(null);
    setTranscriptPreview("");
    setError(null);
    setResult(null);
  }, []);

  return (
    <div className="space-y-6">
      {view === "input" && (
        <div className="card max-w-4xl mx-auto">
          <h1 className="text-xl font-bold text-slate-900 mb-2">GCIS</h1>
          <p className="text-sm text-slate-500 mb-6">Grounded Clinical Intelligence System</p>

          <h2 className="text-lg font-semibold text-slate-900 mb-4">Clinical Transcript Input</h2>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Patient Name</label>
              <input type="text" value={patientName} onChange={(e) => setPatientName(e.target.value)}
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:ring-2 focus:ring-medical-500 focus:border-medical-500"
                placeholder="Optional" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Patient Age</label>
              <input type="number" value={patientAge} onChange={(e) => setPatientAge(e.target.value)}
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:ring-2 focus:ring-medical-500 focus:border-medical-500"
                placeholder="Optional" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Patient ID</label>
              <input type="text" value={patientId} onChange={(e) => setPatientId(e.target.value)}
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:ring-2 focus:ring-medical-500 focus:border-medical-500"
                placeholder="Optional" />
            </div>
          </div>

          <div className="mb-4">
            <label className="block text-sm font-medium text-slate-700 mb-1">Transcript *</label>
            <textarea value={transcript} onChange={(e) => setTranscript(e.target.value)}
              className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:ring-2 focus:ring-medical-500 focus:border-medical-500 min-h-[200px]"
              placeholder="Paste the clinical transcript here..." required />
          </div>

          {error && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
              Error: {error}
            </div>
          )}

          <button onClick={handleProcess} disabled={!isInputReady}
            className="bg-medical-600 hover:bg-medical-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium px-6 py-2.5 rounded-lg transition-colors">
            Process Transcript →
          </button>
        </div>
      )}

      {view === "processing" && (
        <PipelineProgress event={sseEvent} transcriptPreview={transcriptPreview} />
      )}

      {view === "results" && result && (
        <div>
          <ExportBar result={result} onReset={handleReset} />
          <div className="space-y-6 max-w-6xl mx-auto mt-6">
            <FaithfulnessGauge
              score={result.verification.faithfulness_score}
              totalSentences={result.verification.sentence_results.length}
              entailedCount={result.verification.sentence_results.filter((s) => s.label === "ENTAILED").length}
              hallucinatedCount={result.verification.hallucinated_sentences.length}
              refinementIterations={result.refinement_iterations}
            />

            <DialogueActStrip dialogueActs={result.entities.dialogue_acts} />

            {result.entities.temporal_events.length > 0 && (
              <SymptomTimeline
                symptoms={result.entities.symptoms}
                temporalEvents={result.entities.temporal_events}
              />
            )}

            <DeniedSymptoms entities={result.entities.symptoms} />

            <SOAPDisplay
              soap={result.soap}
              verification={result.verification}
            />

            <TranscriptViewer
              transcript={result.transcript}
              sentences={result.entities.sentences}
              dialogueActs={result.entities.dialogue_acts}
              negationScopes={result.entities.negation_scopes}
            />

            {result.soap.differentials.length > 0 && (
              <DifferentialDx differentials={result.soap.differentials} />
            )}
          </div>
        </div>
      )}
    </div>
  );
}
