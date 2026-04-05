"use client";
import type { SOAPNote as SOAPNoteType, SentenceVerification } from "@/lib/types";
import { useState } from "react";

interface Props {
  soap: SOAPNoteType;
  verifications: SentenceVerification[];
}

const LABEL_DOT_COLORS: Record<string, string> = {
  ENTAILED: "bg-green-400",
  NEUTRAL: "bg-amber-400",
  CONTRADICTED: "bg-red-400",
};

export function SOAPNote({ soap, verifications }: Props) {
  const [highlightSource, setHighlightSource] = useState<string | null>(null);

  const renderSection = (title: string, key: "subjective" | "objective" | "assessment" | "plan") => {
    const text = soap[key];
    const sentences = text.split(/(?<=[.!?])\s+/).filter((s) => s.trim());

    return (
      <div>
        <h3 className="text-md font-semibold text-slate-800 mb-2 capitalize">{title}</h3>
        <p className="text-slate-700 leading-relaxed">
          {sentences.map((sentence, i) => {
            const ver = verifications.find(
              (v) => v.soap_section === key && v.soap_sentence === sentence
            );
            const dotColor = ver ? LABEL_DOT_COLORS[ver.label] || "bg-slate-300" : "bg-slate-300";
            const isHallucinated = ver?.is_hallucinated;

            return (
              <span
                key={i}
                className={`inline cursor-pointer group relative ${
                  isHallucinated ? "bg-red-100" : ""
                }`}
                onMouseEnter={() => setHighlightSource(ver?.source_transcript_sentence || null)}
                onMouseLeave={() => setHighlightSource(null)}
              >
                <span className={`inline-block w-2 h-2 rounded-full ${dotColor} mr-0.5 align-middle`} />
                {sentence}
                {ver && (
                  <span className="hidden group-hover:inline ml-1 text-xs text-slate-500">
                    ({ver.label.toLowerCase()}: {(ver.confidence * 100).toFixed(0)}%)
                  </span>
                )}
                {isHallucinated && (
                  <span className="hidden group-hover:inline-block absolute z-10 bg-red-100 border border-red-200 text-red-700 text-xs rounded px-2 py-1 -bottom-8 left-0 whitespace-nowrap">
                    Not found in transcript
                  </span>
                )}
                {" "}
              </span>
            );
          })}
        </p>
      </div>
    );
  };

  return (
    <section className="card" data-verify-highlight={highlightSource}>
      <h2 className="text-lg font-semibold text-slate-900 mb-4">SOAP Note</h2>
      <div className="space-y-4">
        {renderSection("Subjective", "subjective")}
        {renderSection("Objective", "objective")}
        {renderSection("Assessment", "assessment")}
        {renderSection("Plan", "plan")}
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 mt-4 pt-4 border-t border-slate-200 text-xs text-slate-500">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-green-400" /> Entailed
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-amber-400" /> Neutral
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-red-400" /> Contradicted
        </span>
      </div>
    </section>
  );
}
