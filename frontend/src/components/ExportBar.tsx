"use client";
import type { GCISResponse } from "@/lib/types";
import { exportPDF } from "@/lib/api";
import { useState } from "react";

interface Props { result: GCISResponse; onReset: () => void }

export function ExportBar({ result, onReset }: Props) {
  const [copied, setCopied] = useState(false);
  const name = result.patient_info?.patient_name || "Unknown Patient";

  const handlePDF = async () => {
    try {
      await exportPDF(result);
    } catch (e) {
      console.error("PDF export failed:", e);
    }
  };

  const handleCopy = () => {
    const s = result.soap;
    const text = `SUBJECTIVE\n${s.subjective}\n\nOBJECTIVE\n${s.objective}\n\nASSESSMENT\n${s.assessment}\n\nPLAN\n${s.plan}`;
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div className="sticky bottom-0 bg-white border-t border-slate-200 px-4 py-3 flex flex-wrap items-center justify-between gap-3 z-50">
      <span className="text-sm font-medium text-slate-700">{name}</span>
      <div className="flex gap-2">
        <button onClick={handlePDF} className="px-4 py-2 text-sm bg-medical-600 text-white rounded-lg hover:bg-medical-700 transition-colors">
          ↓ Download PDF
        </button>
        <button onClick={handleCopy} className="px-4 py-2 text-sm bg-slate-100 text-slate-700 rounded-lg hover:bg-slate-200 transition-colors">
          {copied ? "Copied!" : "Copy SOAP"}
        </button>
        <button onClick={onReset} className="px-4 py-2 text-sm bg-slate-100 text-slate-700 rounded-lg hover:bg-slate-200 transition-colors">
          ← New Report
        </button>
      </div>
    </div>
  );
}
