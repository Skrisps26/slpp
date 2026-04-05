"use client";
import { useState, useCallback } from "react";
import type { GCISResponse, PatientInfo } from "@/lib/types";
import { processTranscript } from "@/lib/api";
import { FaithfulnessScore } from "@/components/FaithfulnessScore";
import { TranscriptViewer } from "@/components/TranscriptViewer";
import { SOAPNote } from "@/components/SOAPNote";
import { SymptomTimeline } from "@/components/SymptomTimeline";
import { DialogueActs } from "@/components/DialogueActs";
import { DifferentialDx } from "@/components/DifferentialDx";

export default function Home() {
  const [transcript, setTranscript] = useState("");
  const [patientName, setPatientName] = useState("");
  const [patientAge, setPatientAge] = useState("");
  const [patientId, setPatientId] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<GCISResponse | null>(null);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!transcript.trim()) return;

      setLoading(true);
      setError(null);
      setResult(null);

      try {
        const patientInfo: PatientInfo = {
          patient_name: patientName,
          patient_age: parseInt(patientAge) || 0,
          patient_id: patientId,
        };
        const response = await processTranscript(transcript, patientInfo);
        setResult(response);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Unknown error occurred");
      } finally {
        setLoading(false);
      }
    },
    [transcript, patientName, patientAge, patientId]
  );

  return (
    <div className="space-y-8">
      {/* Input Form */}
      <section className="card">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">
          Clinical Transcript Input
        </h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Patient Name
              </label>
              <input
                type="text"
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:ring-2 focus:ring-medical-500 focus:border-medical-500"
                placeholder="Optional"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Patient Age
              </label>
              <input
                type="number"
                value={patientAge}
                onChange={(e) => setPatientAge(e.target.value)}
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:ring-2 focus:ring-medical-500 focus:border-medical-500"
                placeholder="Optional"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Patient ID
              </label>
              <input
                type="text"
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:ring-2 focus:ring-medical-500 focus:border-medical-500"
                placeholder="Optional"
              />
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              Transcript *
            </label>
            <textarea
              value={transcript}
              onChange={(e) => setTranscript(e.target.value)}
              className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:ring-2 focus:ring-medical-500 focus:border-medical-500 min-h-[200px]"
              placeholder="Paste or type the clinical transcript here..."
              required
            />
          </div>
          <button
            type="submit"
            disabled={loading || !transcript.trim()}
            className="bg-medical-600 hover:bg-medical-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium px-6 py-2.5 rounded-lg transition-colors"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                    fill="none"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                  />
                </svg>
                Processing...
              </span>
            ) : (
              "Process Transcript"
            )}
          </button>
        </form>
        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
            Error: {error}
          </div>
        )}
      </section>

      {/* Results */}
      {result && (
        <div className="space-y-6">
          <FaithfulnessScore
            score={result.verification.faithfulness_score}
            total={result.verification.sentence_results.length}
            refinements={result.refinement_iterations}
            hallucinated={result.verification.hallucinated_sentences}
          />

          <DialogueActs dialogueActs={result.entities.dialogue_acts} />

          {result.entities.temporal_events.length > 0 && (
            <SymptomTimeline
              symptoms={result.entities.symptoms}
              temporalEvents={result.entities.temporal_events}
            />
          )}

          <TranscriptViewer
            transcript={result.transcript}
            sentences={result.entities.sentences}
            dialogueActs={result.entities.dialogue_acts}
            negationScopes={result.entities.negation_scopes}
          />

          <SOAPNote
            soap={result.soap}
            verifications={result.verification.sentence_results}
          />

          {result.soap.differentials.length > 0 && (
            <DifferentialDx differentials={result.soap.differentials} />
          )}
        </div>
      )}
    </div>
  );
}
