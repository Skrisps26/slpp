"use client";
import { useState, useRef, useCallback } from "react";

type Status = "idle" | "recording" | "stopped";

interface Props {
  onAudioReady: (wavBlob: Blob) => void;
  onError?: (msg: string) => void;
}

/**
 * Converts a MediaRecorder blob to PCM WAV format client-side.
 * No ffmpeg needed in the browser.
 */
function audioBufferToWav(buffer: AudioBuffer): Blob {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const format = 1; // PCM
  const bitDepth = 16;
  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;
  const dataLength = buffer.length * blockAlign;
  const headerLength = 44;
  const totalLength = headerLength + dataLength;
  const arrayBuffer = new ArrayBuffer(totalLength);
  const view = new DataView(arrayBuffer);

  function writeString(offset: number, str: string) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  }

  writeString(0, "RIFF");
  view.setUint32(4, totalLength - 8, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true); // Subchunk1Size
  view.setUint16(20, format, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeString(36, "data");
  view.setUint32(40, dataLength, true);

  // Write interleaved PCM data
  const channels: Float32Array[] = [];
  for (let c = 0; c < numChannels; c++) {
    channels.push(buffer.getChannelData(c));
  }
  let offset = 44;
  for (let i = 0; i < buffer.length; i++) {
    for (let ch = 0; ch < numChannels; ch++) {
      let sample = Math.max(-1, Math.min(1, channels[ch][i]));
      // Convert float32 to int16
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
      offset += 2;
    }
  }

  return new Blob([arrayBuffer], { type: "audio/wav" });
}

export function AudioRecorder({ onAudioReady, onError }: Props) {
  const [status, setStatus] = useState<Status>("idle");
  const [elapsed, setElapsed] = useState(0);
  const [waveform, setWaveform] = useState<number[]>(Array(12).fill(0));
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const animRef = useRef<number | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Setup AudioContext + Analyser for waveform
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 64;
      source.connect(analyser);
      analyserRef.current = analyser;

      // Setup MediaRecorder to capture as webm/opus
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      mediaRecorderRef.current = recorder;
      recorder.start(100);

      setStatus("recording");
      setElapsed(0);
      timerRef.current = setInterval(() => setElapsed((s) => s + 1), 1000);

      // Animate waveform bars
      const freqData = new Uint8Array(analyser.frequencyBinCount);
      const updateWaveform = () => {
        if (!analyserRef.current) return;
        analyserRef.current.getByteFrequencyData(freqData);
        const step = Math.max(1, Math.floor(freqData.length / 12));
        const bars = Array.from({ length: 12 }, (_, i) => freqData[i * step] / 255);
        setWaveform(bars);
        animRef.current = requestAnimationFrame(updateWaveform);
      };
      animRef.current = requestAnimationFrame(updateWaveform);
    } catch (err) {
      console.error("Microphone error:", err);
      onError?.("Could not access microphone. Please grant permission.");
    }
  }, [onError]);

  const stopRecording = useCallback(async () => {
    // Stop recording
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }

    // Wait for onstop event
    mediaRecorderRef.current!.onstop = async () => {
      const webmBlob = new Blob(chunksRef.current, { type: "audio/webm" });

      // Convert to WAV offline
      try {
        const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
        const arrayBuffer = await webmBlob.arrayBuffer();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        const wavBlob = audioBufferToWav(audioBuffer);
        onAudioReady(wavBlob);
        setStatus("stopped");
      } catch (err) {
        console.error("WAV conversion error:", err);
        onError?.("Failed to convert audio to WAV. Please try again.");
      }
    };

    // Stop analyser stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (animRef.current) cancelAnimationFrame(animRef.current);
    if (timerRef.current) clearInterval(timerRef.current);
    setStatus("stopped");
  }, [onAudioReady, onError]);

  const reset = useCallback(() => {
    setStatus("idle");
    setElapsed(0);
    setWaveform(Array(12).fill(0));
    chunksRef.current = [];
  }, []);

  const formatTime = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, "0")}`;

  return (
    <div className="w-full max-w-xs mx-auto">
      {/* Waveform bars */}
      <div className="flex items-end gap-0.5 h-8 mb-2 mx-auto w-fit">
        {waveform.map((v, i) => (
          <div
            key={i}
            className={`w-1.5 rounded-full transition-all duration-75 ${
              status === "recording" ? (v > 0.3 ? "bg-red-500" : "bg-red-300") : "bg-slate-200"
            }`}
            style={{ height: `${Math.max(4, v * 100)}%` }}
          />
        ))}
      </div>

      {/* Controls */}
      {status === "idle" && (
        <button
          onClick={startRecording}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-red-500 hover:bg-red-600 text-white rounded-lg font-medium transition-colors"
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z" />
          </svg>
          Record Audio
        </button>
      )}

      {status === "recording" && (
        <div className="space-y-2">
          <div className="flex items-center justify-center gap-2 text-sm font-medium text-red-600">
            <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
            Recording… {formatTime(elapsed)}
          </div>
          <button
            onClick={stopRecording}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-slate-800 hover:bg-slate-700 text-white rounded-lg font-medium transition-colors"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <rect x="5" y="5" width="10" height="10" rx="2" />
            </svg>
            Stop Recording
          </button>
        </div>
      )}

      {status === "stopped" && (
        <div className="space-y-2">
          <p className="text-sm text-green-600 text-center font-medium">✓ Audio recorded as WAV</p>
          <div className="flex gap-2">
            <button
              onClick={reset}
              className="flex-1 px-4 py-2.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-sm font-medium transition-colors"
            >
              Re-record
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
