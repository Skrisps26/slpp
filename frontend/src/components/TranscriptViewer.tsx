"use client";
import type { DialogueActType } from "@/lib/types";
import { useState } from "react";

interface Props {
  transcript: string;
  sentences: string[];
  dialogueActs: DialogueActType[];
  negationScopes: Array<{ text: string; start: number; end: number }>;
}

const LABEL_COLORS: Record<string, string> = {
  SYMPTOM_REPORT: "bg-green-50 hover:bg-green-100",
  QUESTION: "bg-blue-50 hover:bg-blue-100",
  DIAGNOSIS_STATEMENT: "bg-purple-50 hover:bg-purple-100",
  TREATMENT_PLAN: "bg-teal-50 hover:bg-teal-100",
  REASSURANCE: "bg-gray-50 hover:bg-gray-100",
  HISTORY: "bg-amber-50 hover:bg-amber-100",
  OTHER: "",
};

const LABEL_BADGE: Record<string, string> = {
  SYMPTOM_REPORT: "bg-green-100 text-green-800",
  QUESTION: "bg-blue-100 text-blue-800",
  DIAGNOSIS_STATEMENT: "bg-purple-100 text-purple-800",
  TREATMENT_PLAN: "bg-teal-100 text-teal-800",
  REASSURANCE: "bg-gray-200 text-gray-800",
  HISTORY: "bg-amber-100 text-amber-800",
  OTHER: "bg-slate-100 text-slate-600",
};

export function TranscriptViewer({ transcript, sentences, dialogueActs, negationScopes }: Props) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const isNegated = (sentenceStart: number, end: number) => {
    return negationScopes?.some((s) => s.start <= sentenceStart && end <= s.end);
  };

  return (
    <section className="card">
      <h2 className="text-lg font-semibold text-slate-900 mb-4">Transcript</h2>
      <div className="space-y-0.5 text-sm leading-relaxed">
        {sentences.map((sentence, i) => {
          const da = dialogueActs?.[i];
          const neg = isNegated(transcript.indexOf(sentence), transcript.indexOf(sentence) + sentence.length);
          return (
            <span
              key={i}
              className={`inline mr-1 px-1 py-0.5 rounded cursor-pointer transition-colors ${
                i === hoveredIdx ? "ring-2 ring-amber-400 bg-amber-100" : LABEL_COLORS[da?.label ?? ""] || ""
              }`}
              onMouseEnter={() => setHoveredIdx(i)}
              onMouseLeave={() => setHoveredIdx(null)}
            >
              {neg ? (
                <span className="line-through text-red-500 opacity-70">{sentence}</span>
              ) : (
                <span className="text-slate-700">{sentence}</span>
              )}
              {hoveredIdx === i && da && (
                <span className={`ml-1 badge text-[10px] ${LABEL_BADGE[da.label] ?? LABEL_BADGE.OTHER}`}>
                  {da.label} ({Math.round(da.confidence * 100)}%)
                </span>
              )}
            </span>
          );
        })}
      </div>
    </section>
  );
}
