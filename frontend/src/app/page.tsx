"use client";
import { useState, useCallback, useRef } from "react";
import type { GCISResponse, PatientInfo } from "@/lib/types";
import { processTranscriptStream, processAudioStream, SSEEvent } from "@/lib/api";
import { AudioRecorder } from "@/components/AudioRecorder";
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
type InputMode = "text" | "upload" | "record";

export default function Home() {
  const [view, setView] = useState<View>("input");
  const [inputMode, setInputMode] = useState<InputMode>("text");
  const [transcript, setTranscript] = useState("");
  const [patientName, setPatientName] = useState("");
  const [patientAge, setPatientAge] = useState("");
  const [patientId, setPatientId] = useState("");
  const [sseEvent, setSseEvent] = useState<SSEEvent | null>(null);
  const [transcriptPreview, setTranscriptPreview] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<GCISResponse | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const getPatientInfo = (): PatientInfo => ({
    patient_name: patientName,
    patient_age: parseInt(patientAge) || 0,
    patient_id: patientId,
  });

  const isTextReady = transcript.trim().length > 0;
  const isAudioReady = (inputMode === "upload" && audioFile !== null);

  const handleProcess = useCallback(async () => {
    if (!isTextReady && !isAudioReady) return;
    setView("processing");
    setSseEvent(null);
    setTranscriptPreview("");
    setError(null);
    setResult(null);

    try {
      if (inputMode === "upload" && audioFile) {
        // Process audio file via SSE
        const final = await processAudioStream(audioFile, getPatientInfo(), (event) => {
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
      } else {
        // Process text transcript via SSE
        const final = await processTranscriptStream(transcript, getPatientInfo(), (event) => {
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
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error occurred");
      setView("input");
    }
  }, [inputMode, transcript, audioFile, patientName, patientAge, patientId, isTextReady, isAudioReady]);

  /** Called when AudioRecorder stops and converts to WAV */
  const handleAudioReady = useCallback((wavBlob: Blob) => {
    const file = new File([wavBlob], "recording.wav", { type: "audio/wav" });
    setAudioFile(file);
  }, []);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    // Only accept .wav files
    if (!file.name.toLowerCase().endsWith(".wav") && file.type !== "audio/wav") {
      setError("Only .wav files are supported for transcription. Please convert your file to WAV first.");
      return;
    }
    setAudioFile(file);
    setError(null);
  }, []);

  const handleReset = useCallback(() => {
    setView("input");
    setTranscript("");
    setSseEvent(null);
    setTranscriptPreview("");
    setError(null);
    setResult(null);
    setAudioFile(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, []);

  return (
    <div className="space-y-6">
      {/* ── INPUT VIEW ── */}
      {view === "input" && (
        <div className="card max-w-4xl mx-auto">
          <h1 className="text-xl font-bold text-slate-900 mb-1">GCIS</h1>
          <p className="text-sm text-slate-500 mb-6">Grounded Clinical Intelligence System v2.0</p>

          {/* Patient Info */}
          <div className="flex gap-3 mb-6">
            <input type="text" value={patientName} onChange={(e) => setPatientName(e.target.value)}
              className="flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm focus:ring-2 focus:ring-medical-500" placeholder="Patient name" />
            <input type="number" value={patientAge} onChange={(e) => setPatientAge(e.target.value)}
              className="w-20 rounded-lg border border-slate-300 px-3 py-2 text-sm focus:ring-2 focus:ring-medical-500" placeholder="Age" />
            <input type="text" value={patientId} onChange={(e) => setPatientId(e.target.value)}
              className="w-28 rounded-lg border border-slate-300 px-3 py-2 text-sm focus:ring-2 focus:ring-medical-500" placeholder="ID" />
          </div>

          {/* Mode Tabs */}
          <div className="flex border-b border-slate-200 mb-5">
            {(["text", "upload", "record"] as InputMode[]).map((mode) => (
              <button key={mode} onClick={() => { setInputMode(mode); setError(null); }}
                className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors capitalize ${
                  inputMode === mode
                    ? "border-medical-600 text-medical-600"
                    : "border-transparent text-slate-400 hover:text-slate-600"
                }`}>
                {mode === "text" ? "📝 Transcript" : mode === "upload" ? "📁 Upload .wav" : "🎙️ Record"}
              </button>
            ))}
          </div>

          {/* Text mode */}
          {inputMode === "text" && (
            <textarea value={transcript} onChange={(e) => setTranscript(e.target.value)}
              className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:ring-2 focus:ring-medical-500 min-h-[200px] mb-4"
              placeholder="Paste the clinical transcript here..." />
          )}

          {/* Upload mode */}
          {inputMode === "upload" && (
            <div className="space-y-3 mb-4">
              <label className="flex flex-col items-center justify-center w-full h-36 border-2 border-dashed border-slate-300 rounded-lg cursor-pointer bg-slate-50 hover:bg-slate-100 transition-colors">
                <div className="text-center">
                  <svg className="w-10 h-10 mx-auto mb-2 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="1.5">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  <p className="text-sm text-slate-500">
                    {audioFile ? `✓ ${audioFile.name}` : "Click to upload or drag a .wav file"}
                  </p>
                  <p className="text-xs text-slate-400 mt-1">Only .wav files accepted · Whisper base model transcribes</p>
                </div>
                <input ref={fileInputRef} type="file" accept=".wav" className="hidden" onChange={handleFileChange} />
              </label>
              {audioFile && (
                <div className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg text-sm">
                  <svg className="w-5 h-5 text-medical-600 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                  </svg>
                  <span className="text-slate-700 font-medium">{audioFile.name}</span>
                  <span className="text-slate-400 ml-auto">
                    {(audioFile.size / 1024 / 1024).toFixed(2)} MB
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Record mode */}
          {inputMode === "record" && (
            <div className="mb-4">
              <AudioRecorder onAudioReady={handleAudioReady} onError={(msg) => setError(msg)} />
              {audioFile && (
                <div className="mt-3 p-3 bg-slate-50 rounded-lg text-sm text-center text-green-600 font-medium">
                  ✓ {audioFile.name} ready — click Process to transcribe
                </div>
              )}
            </div>
          )}

          {error && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
              ⚠ {error}
            </div>
          )}

          <button onClick={handleProcess}
            disabled={(inputMode === "text" && !isTextReady) || (inputMode === "upload" && !isAudioReady)}
            className="w-full sm:w-auto bg-medical-600 hover:bg-medical-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium px-6 py-2.5 rounded-lg transition-colors">
            {inputMode === "upload" || inputMode === "record" ? "Transcribe & Process →" : "Process Transcript →"}
          </button>
        </div>
      )}

      {/* ── PROCESSING VIEW ── */}
      {view === "processing" && (
        <PipelineProgress event={sseEvent} transcriptPreview={transcriptPreview} />
      )}

      {/* ── RESULTS VIEW ── */}
      {view === "results" && result && (
        <div>
          <ExportBar result={result} onReset={handleReset} />
          <div className="space-y-6 max-w-6xl mx-auto mt-6 pb-20">
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

            <SOAPDisplay soap={result.soap} verification={result.verification} />

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
