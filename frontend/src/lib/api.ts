import type { GCISResponse, PatientInfo } from "./types";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export async function processTranscript(
  transcript: string,
  patientInfo: PatientInfo
): Promise<GCISResponse> {
  const res = await fetch(`${BACKEND_URL}/api/process/transcript`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ transcript, ...patientInfo }),
  });
  if (!res.ok) throw new Error(`Backend error: ${res.status}`);
  return res.json();
}

export async function processAudio(audioBlob: Blob): Promise<GCISResponse> {
  const form = new FormData();
  form.append("file", audioBlob, "recording.wav");
  const res = await fetch(`${BACKEND_URL}/api/process/audio`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(`Backend error: ${res.status}`);
  return res.json();
}

export async function checkHealth(): Promise<{ status: string; models_loaded: boolean }> {
  const res = await fetch(`${BACKEND_URL}/api/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}
