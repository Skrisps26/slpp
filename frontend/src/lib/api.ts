import type { GCISResponse, PatientInfo } from "./types";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export async function processTranscript(
  transcript: string,
  patientInfo: PatientInfo
): Promise<GCISResponse> {
  const age = typeof patientInfo.patient_age === "number" ? patientInfo.patient_age : (parseInt(String(patientInfo.patient_age), 10) || 0);
  const res = await fetch(`${BACKEND_URL}/api/process/transcript`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      transcript,
      patient_name: patientInfo.patient_name || "",
      patient_age: age,
      patient_id: patientInfo.patient_id || "",
    }),
  });
  if (!res.ok) {
    const errorBody = await res.text().catch(() => "");
    throw new Error(`Backend error ${res.status}: ${errorBody}`);
  }
  return res.json();
}

export async function processAudio(
  file: File,
  patientInfo: PatientInfo
): Promise<GCISResponse> {
  const formData = new FormData();
  formData.append("file", file);
  // Send patient info as JSON in a separate field
  formData.append(
    "patient_info",
    JSON.stringify({
      patient_name: patientInfo.patient_name || "",
      patient_age: typeof patientInfo.patient_age === "number" ? patientInfo.patient_age : (parseInt(String(patientInfo.patient_age), 10) || 0),
      patient_id: patientInfo.patient_id || "",
    })
  );

  const res = await fetch(`${BACKEND_URL}/api/process/audio`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const errorBody = await res.text().catch(() => "");
    throw new Error(`Backend error ${res.status}: ${errorBody}`);
  }
  return res.json();
}

export async function checkHealth(): Promise<{ status: string; models_loaded: boolean }> {
  const res = await fetch(`${BACKEND_URL}/api/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}
