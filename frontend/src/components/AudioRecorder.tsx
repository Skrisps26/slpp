"use client";
import { useState } from "react";

interface AudioRecorderProps {
  onAudioReady: (blob: Blob) => void;
}

export function AudioRecorder({ onAudioReady }: AudioRecorderProps) {
  const [recording, setRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [duration, setDuration] = useState(0);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks: BlobPart[] = [];

      recorder.ondataavailable = (e) => chunks.push(e.data);
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: "audio/webm" });
        onAudioReady(blob);
        stream.getTracks().forEach((t) => t.stop());
      };

      recorder.start();
      setMediaRecorder(recorder);
      setRecording(true);
      setDuration(0);

      const interval = setInterval(() => setDuration((d) => d + 1), 1000);
      (recorder as any)._interval = interval;
    } catch (err) {
      console.error("Microphone access denied:", err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.stop();
      clearInterval((mediaRecorder as any)._interval);
      setRecording(false);
      setMediaRecorder(null);
    }
  };

  const formatDuration = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  return (
    <div className="flex items-center gap-4">
      <button
        onClick={recording ? stopRecording : startRecording}
        className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
          recording
            ? "bg-red-500 hover:bg-red-600 text-white"
            : "bg-slate-700 hover:bg-slate-800 text-white"
        }`}
      >
        {recording ? (
          <>
            <div className="w-3 h-3 bg-white rounded-full animate-pulse" />
            Stop ({formatDuration(duration)})
          </>
        ) : (
          <>
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
              <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
            </svg>
            Start Recording
          </>
        )}
      </button>
    </div>
  );
}
