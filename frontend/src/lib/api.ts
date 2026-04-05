import type { GCISResponse, PatientInfo } from "./types";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

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

async function consumeSSE(
  response: Response,
  onEvent: (event: SSEEvent) => void
): Promise<GCISResponse | null> {
  if (!response.ok || !response.body) {
    throw new Error(`Backend error: ${response.status}: ${await response.text()}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data: ")) continue;
      try {
        const event: SSEEvent = JSON.parse(trimmed.slice(6));
        onEvent(event);
        if (event.stage === "complete") return event.result ?? null;
      } catch {
        // skip malformed SSE lines
      }
    }
  }
  return null;
}

/** Process a text transcript via SSE streaming. */
export async function processTranscriptStream(
  transcript: string,
  patientInfo: PatientInfo,
  onEvent: (event: SSEEvent) => void
): Promise<GCISResponse | null> {
  const response = await fetch(`${BACKEND_URL}/api/process/transcript`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ transcript, ...patientInfo }),
  });
  return consumeSSE(response, onEvent);
}

/** Process an audio file via SSE streaming. File MUST be .wav. */
export async function processAudioStream(
  file: File,
  patientInfo: PatientInfo,
  onEvent: (event: SSEEvent) => void
): Promise<GCISResponse | null> {
  // Validate WAV before sending
  if (!file.name.toLowerCase().endsWith(".wav") && file.type !== "audio/wav") {
    throw new Error("Only .wav files are accepted for transcription.");
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append(
    "patient_info",
    JSON.stringify({
      patient_name: patientInfo.patient_name || "",
      patient_age: typeof patientInfo.patient_age === "number" ? patientInfo.patient_age : parseInt(String(patientInfo.patient_age)) || 0,
      patient_id: patientInfo.patient_id || "",
    })
  );

  const response = await fetch(`${BACKEND_URL}/api/process/audio`, {
    method: "POST",
    body: formData,
  });
  return consumeSSE(response, onEvent);
}

/** Download the SOAP note as a PDF. */
export async function exportPDF(payload: GCISResponse): Promise<void> {
  const res = await fetch(`${BACKEND_URL}/api/export/pdf`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`PDF export failed: ${res.status}`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "soap_note.pdf";
  a.click();
  URL.revokeObjectURL(url);
}

/** Health / readiness check. */
export async function checkHealth(): Promise<{ status: string; models_loaded: boolean }> {
  const res = await fetch(`${BACKEND_URL}/api/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}
