"use client";
import type { DialogueAct } from "@/lib/types";
import { useState } from "react";

interface Props {
  transcript: string;
  sentences: string[];
  dialogueActs: DialogueAct[];
  negationScopes: Array<{ text: string; start: number; end: number }>;
}

const LABEL_COLORS: Record<string, string> = {
  SYMPTOM_REPORT: "bg-green-100 text-green-800 border-green-200",
  QUESTION: "bg-blue-100 text-blue-800 border-blue-200",
  DIAGNOSIS_STATEMENT: "bg-purple-100 text-purple-800 border-purple-200",
  TREATMENT_PLAN: "bg-teal-100 text-teal-800 border-teal-200",
  REASSURANCE: "bg-gray-100 text-gray-800 border-gray-200",
  HISTORY: "bg-amber-100 text-amber-800 border-amber-200",
  OTHER: "bg-slate-100 text-slate-800 border-slate-200",
};

export function TranscriptViewer({ transcript, sentences, dialogueActs, negationScopes }: Props) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [highlightSource, setHighlightSource] = useState<string | null>(null);

  const getNegatedRanges = () => {
    const ranges: [number, number][] = [];
    for (const scope of negationScopes || []) {
      ranges.push([scope.start, scope.end]);
    }
    return ranges;
  };

  const negatedRanges = getNegatedRanges();

  const isNegated = (start: number, end: number) => {
    return negatedRanges.some(([s, e]) => s <= start && end <= e);
  };

  return (
    <section className="card">
      <h2 className="text-lg font-semibold text-slate-900 mb-4">Transcript</h2>
      
      {/* Dialogue act strip */}
      <div className="mb-4">
        <h3 className="text-sm font-medium text-slate-600 mb-2">Dialogue Acts</h3>
        <div className="flex flex-wrap gap-1.5">
          {dialogueActs.map((da, i) => (
            <button
              key={i}
              className={`badge border ${LABEL_COLORS[da.label] || LABEL_COLORS.OTHER} cursor-pointer hover:opacity-80 transition-opacity`}
              onClick={() => setHoveredIdx(i === hoveredIdx ? null : i)}
              title={`${da.label} (${(da.confidence * 100).toFixed(0)}%)`}
            >
              {da.label} ({(da.confidence * 100).toFixed(0)}%)
            </button>
          ))}
        </div>
      </div>

      {/* Sentence-by-sentence display */}
      <div className="space-y-1 text-base leading-relaxed">
        {sentences.map((sentence, i) => (
          <span
            key={i}
            className={`inline mr-1 px-0.5 py-0.5 rounded transition-colors cursor-pointer ${
              i === hoveredIdx
                ? "bg-amber-200"
                : sentence === highlightSource
                  ? "bg-amber-100"
                  : ""
            } ${
              dialogueActs[i]
                ? LABEL_COLORS[dialogueActs[i].label]
                : ""
            }`}
            onMouseEnter={() => setHoveredIdx(i)}
            onMouseLeave={() => setHoveredIdx(null)}
          >
            {isNegated(
              transcript.indexOf(sentence),
              transcript.indexOf(sentence) + sentence.length
            ) && (
              <span className="line-through text-red-600">{sentence}</span>
            )}
            {!isNegated(
              transcript.indexOf(sentence),
              transcript.indexOf(sentence) + sentence.length
            ) && sentence}
            {hoveredIdx === i && dialogueActs[i] && (
              <span className="ml-1 text-xs text-slate-500">
                [{dialogueActs[i].label}: {(dialogueActs[i].confidence * 100).toFixed(0)}%]
              </span>
            )}
          </span>
        ))}
      </div>
    </section>
  );
}
