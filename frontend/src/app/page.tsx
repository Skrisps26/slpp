"use client";
import { useState, useCallback, useRef } from "react";
import type { GCISResponse, PatientInfo } from "@/lib/types";
import { processTranscript, processAudio } from "@/lib/api";
import { FaithfulnessScore } from "@/components/FaithfulnessScore";
import { TranscriptViewer } from "@/components/TranscriptViewer";
import { SOAPNote } from "@/components/SOAPNote";
import { SymptomTimeline } from "@/components/SymptomTimeline";
import { DialogueActs } from "@/components/DialogueActs";
import { DifferentialDx } from "@/components/DifferentialDx";

type InputMode = "text" | "upload" | "record";

export default function Home() {
  const [mode, setMode] = useState<InputMode>("text");
  const [transcript, setTranscript] = useState("");
  const [patientName, setPatientName] = useState("");
  const [patientAge, setPatientAge] = useState("");
  const [patientId, setPatientId] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<GCISResponse | null>(null);

  // Audio recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Audio upload state
  const [selectedAudio, setSelectedAudio] = useState<File | null>(null);

  const getPatientInfo = (): PatientInfo => ({
    patient_name: patientName,
    patient_age: parseInt(patientAge) || 0,
    patient_id: patientId,
  });

  // Handle text submission
  const handleTextSubmit = useCallback(async () => {
    if (!transcript.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await processTranscript(transcript, getPatientInfo());
      setResult(response);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error occurred");
    } finally {
      setLoading(false);
    }
  }, [transcript, patientName, patientAge, patientId]);

  // Handle audio upload submission
  const handleAudioUpload = useCallback(async () => {
    if (!selectedAudio) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await processAudio(selectedAudio, getPatientInfo());
      setResult(response);
      // Also populate the transcript textarea with the result
      setTranscript(response.transcript);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error occurred");
    } finally {
      setLoading(false);
    }
  }, [selectedAudio, patientName, patientAge, patientId]);

  // Audio recording functions
  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks: BlobPart[] = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      recorder.onstop = async () => {
        const blob = new Blob(chunks, { type: "audio/webm" });
        const wavFile = new File([blob], "recording.wav", { type: "audio/wav" });

        setLoading(true);
        setError(null);
        setResult(null);

        try {
          const response = await processAudio(wavFile, getPatientInfo());
          setResult(response);
          setTranscript(response.transcript);
        } catch (err: unknown) {
          setError(err instanceof Error ? err.message : "Unknown error occurred");
        } finally {
          setLoading(false);
        }

        stream.getTracks().forEach((track) => track.stop());
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      chunksRef.current = chunks;
      setIsRecording(true);
      setRecordingDuration(0);
      timerRef.current = setInterval(() => {
        setRecordingDuration((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      console.error("Microphone access denied:", err);
      setError("Could not access microphone. Please check browser permissions.");
    }
  }, [patientName, patientAge, patientId]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const formatDuration = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  // Mode tabs
  const tabs: { key: InputMode; label: string; icon: string }[] = [
    { key: "text", label: "Paste Transcript", icon: "📝" },
    { key: "upload", label: "Upload Audio", icon: "📁" },
    { key: "record", label: "Record Audio", icon: "🎙️" },
  ];

  return (
    <div className="space-y-8">
      {/* Input Section */}
      <section className="card">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">
          Clinical Input
        </h2>

        {/* Patient Info */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
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

        {/* Mode Tabs */}
        <div className="flex border-b border-slate-200 mb-4">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => {
                setMode(tab.key);
                setError(null);
              }}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                mode === tab.key
                  ? "border-medical-600 text-medical-600"
                  : "border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300"
              }`}
            >
              {tab.icon} {tab.label}
            </button>
          ))}
        </div>

        {/* Transcript Mode */}
        {mode === "text" && (
          <div className="space-y-4">
            <textarea
              value={transcript}
              onChange={(e) => setTranscript(e.target.value)}
              className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:ring-2 focus:ring-medical-500 focus:border-medical-500 min-h-[200px]"
              placeholder="Paste or type the clinical transcript here..."
            />
            <button
              onClick={handleTextSubmit}
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
          </div>
        )}

        {/* Upload Mode */}
        {mode === "upload" && (
          <div className="space-y-4">
            <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-slate-300 rounded-lg cursor-pointer bg-slate-50 hover:bg-slate-100 transition-colors">
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <svg className="w-8 h-8 mb-3 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p className="mb-1 text-sm text-slate-500">
                  {selectedAudio
                    ? `Selected: ${selectedAudio.name} (${(selectedAudio.size / 1024 / 1024).toFixed(2)} MB)`
                    : "Click to upload or drag and drop"}
                </p>
                <p className="text-xs text-slate-400">
                  .wav, .mp3, .mp4, .m4a, .ogg, .flac
                </p>
              </div>
              <input
                type="file"
                accept=".wav,.mp3,.mp4,.m4a,.ogg,.flac"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    setSelectedAudio(e.target.files[0]);
                  }
                }}
              />
            </label>
            <button
              onClick={handleAudioUpload}
              disabled={loading || !selectedAudio}
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
                  Transcribing & Processing...
                </span>
              ) : (
                "Process Audio"
              )}
            </button>
          </div>
        )}

        {/* Record Mode */}
        {mode === "record" && (
          <div className="space-y-4">
            <div className="flex flex-col items-center justify-center p-8 border-2 border-dashed border-slate-300 rounded-lg bg-slate-50">
              {isRecording ? (
                <>
                  <div className="w-16 h-16 bg-red-500 rounded-full flex items-center justify-center mb-4 animate-pulse">
                    <div className="w-4 h-4 bg-white rounded-full" />
                  </div>
                  <p className="text-lg font-medium text-slate-700 mb-2">
                    Recording... {formatDuration(recordingDuration)}
                  </p>
                  <button
                    onClick={stopRecording}
                    className="bg-red-500 hover:bg-red-600 text-white font-medium px-6 py-2 rounded-lg transition-colors"
                  >
                    Stop & Process
                  </button>
                </>
              ) : (
                <>
                  <svg className="w-16 h-16 text-slate-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="1.5">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
                  </svg>
                  <p className="text-lg font-medium text-slate-700 mb-2">
                    Record Doctor-Patient Conversation
                  </p>
                  <p className="text-sm text-slate-500 mb-4 text-center">
                    Click the button below to start recording. Audio will be transcribed and processed automatically.
                  </p>
                  <button
                    onClick={startRecording}
                    disabled={loading}
                    className="bg-medical-600 hover:bg-medical-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium px-6 py-2.5 rounded-lg transition-colors"
                  >
                    {loading ? "Processing..." : "Start Recording"}
                  </button>
                </>
              )}
            </div>
          </div>
        )}

        {/* Error Display */}
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
