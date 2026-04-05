"use client";
import type { SOAPNote, VerificationResult, SentenceVerification } from "@/lib/types";
import { useState, useCallback } from "react";

interface SOAPDisplayProps {
  soap: SOAPNote;
  verification: VerificationResult;
}

const dotColor = (label: string) => {
  if (label === "ENTAILED") return "bg-green-500";
  if (label === "NEUTRAL") return "bg-amber-400";
  if (label === "CONTRADICTED") return "bg-red-500";
  return "bg-slate-300";
};

export function SOAPDisplay({ soap, verification }: SOAPDisplayProps) {
  const [hoveredTranscriptIdx, setHoveredTranscriptIdx] = useState<number | null>(null);

  const findTranscriptIndex = useCallback(
    (sourceText: string) => {
      if (!sourceText) return null;
      return -1;
    },
    []
  );

  const sections: { name: string; text: string }[] = [
    { name: "Subjective", text: soap.subjective },
    { name: "Objective", text: soap.objective },
    { name: "Assessment", text: soap.assessment },
    { name: "Plan", text: soap.plan },
  ];

  return (
    <section className="card" data-verify-highlight={hoveredTranscriptIdx ?? ""}>
      <h2 className="text-lg font-semibold text-slate-900 mb-4">SOAP Note</h2>
      {sections.map(({ name, text }) => (
        <div key={name} className="mb-4">
          <h3 className="text-sm font-semibold text-slate-800 mb-1">{name}</h3>
          {text.split(/[.!?]+(?=\s)/).map((sent, i) => {
            const trimmed = sent.trim();
            if (!trimmed) return null;
            const sv = verification.sentence_results.find(
              (v) => v.soap_sentence === trimmed && v.soap_section === name.toLowerCase()
            );
            const isHallucinated = sv?.is_hallucinated;
            return (
              <div
                key={i}
                className={`flex items-start gap-2 py-1 group ${
                  isHallucinated ? "opacity-60" : ""
                }`}
                onMouseEnter={() => setHoveredTranscriptIdx(findTranscriptIndex(sv?.source_transcript_sentence ?? ""))}
                onMouseLeave={() => setHoveredTranscriptIdx(null)}
              >
                <span className={`mt-2 h-2 w-2 rounded-full flex-shrink-0 ${dotColor(sv?.label ?? "")}`} />
                <p className="flex-1 text-sm text-slate-700">
                  {trimmed}
                  {sv?.label === "NEUTRAL" && (
                    <span className="ml-1 text-amber-500 text-xs" title="Unverified">?</span>
                  )}
                  {isHallucinated && (
                    <span className="ml-1 text-orange-500 text-xs" title="Was hallucinated — auto-corrected">✓ corrected</span>
                  )}
                </p>
              </div>
            );
          })}
        </div>
      ))}

      {/* Legend */}
      <div className="flex items-center gap-4 mt-4 pt-4 border-t border-slate-200 text-xs text-slate-500">
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-green-500" /> Entailed</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-amber-400" /> Neutral</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-500" /> Contradicted</span>
      </div>
    </section>
  );
}
