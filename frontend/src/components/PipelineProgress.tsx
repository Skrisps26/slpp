import type { GCISResponse, PatientInfo } from "./types";

export interface PipelineStage {
  key: string;
  label: string;
  status: "waiting" | "running" | "done";
}

export interface SSEEvent {
  stage:
    | "transcribing"
    | "transcribed"
    | "extracting"
    | "retrieving"
    | "generating"
    | "verifying"
    | "refining"
    | "complete";
  progress: number;
  message: string;
  transcript?: string;
  iteration?: number;
  hallucinated_count?: number;
  result?: GCISResponse;
}

const STAGE_ORDER = [
  { key: "transcribing", label: "Transcribing audio" },
  { key: "extracting", label: "Extracting clinical entities" },
  { key: "retrieving", label: "Searching knowledge base" },
  { key: "generating", label: "Generating SOAP note" },
  { key: "verifying", label: "Verifying against transcript" },
  { key: "refining", label: "Auto-correcting hallucinations" },
  { key: "complete", label: "Complete" },
];

interface PipelineProgressProps {
  event: SSEEvent | null;
  transcriptPreview: string;
  onCancel?: () => void;
}

export function PipelineProgress({ event, transcriptPreview, onCancel }: PipelineProgressProps) {
  const currentKey = event?.stage ?? "extracting";
  const currentIdx = STAGE_ORDER.findIndex((s) => s.key === currentKey);
  const progress = event?.progress ?? 0;

  const stages = STAGE_ORDER.map((s, i) => {
    let status: PipelineStage["status"] = "waiting";
    if (i < currentIdx) status = "done";
    if (i === currentIdx) status = currentKey === "complete" ? "done" : "running";
    return { ...s, status };
  });

  return (
    <div className="card max-w-2xl mx-auto">
      <h2 className="text-lg font-semibold text-slate-900 mb-4">Processing…</h2>

      {/* Stage list */}
      <div className="space-y-2 mb-4">
        {stages.filter((s) => s.key !== "transcribing" || currentKey === "transcribing").map(
          (s) => (
            <div key={s.key} className="flex items-center gap-3 py-1.5">
              {s.status === "done" && (
                <svg className="w-5 h-5 text-green-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
              )}
              {s.status === "running" && (
                <svg className="w-5 h-5 text-medical-500 flex-shrink-0 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              )}
              {s.status === "waiting" && <div className="w-5 h-5 rounded-full border-2 border-slate-300 flex-shrink-0" />}
              <span className={`text-sm ${s.status === "running" ? "text-medical-600 font-medium" : "text-slate-600"}`}>
                {s.label}
              </span>
            </div>
          )
        )}
      </div>

      {/* Progress bar */}
      <div className="w-full bg-slate-200 rounded-full h-2 mb-2">
        <div
          className="bg-medical-500 rounded-full h-2 transition-all duration-300"
          style={{ width: `${Math.min(progress, 100)}%` }}
        />
      </div>
      <p className="text-xs text-slate-500 mb-2">
        {event?.message ?? "Initializing…"}
        {event?.iteration ? ` (pass ${event.iteration} of 2)` : ""}
        {event?.hallucinated_count ? ` · ${event.hallucinated_count} hallucination(s)` : ""}
      </p>

      {/* Transcript preview during transcription */}
      {currentKey === "transcribed" && (
        <div className="mt-4 p-3 bg-slate-50 rounded-lg text-sm text-slate-700 max-h-40 overflow-y-auto">
          <p className="text-xs font-medium text-slate-500 mb-1">Transcript Preview:</p>
          {transcriptPreview}
        </div>
      )}

      {onCancel && (
        <button onClick={onCancel} className="mt-4 text-sm text-slate-400 hover:text-slate-600">
          Cancel
        </button>
      )}
    </div>
  );
}
