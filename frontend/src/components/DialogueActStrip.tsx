"use client";
import type { DialogueActType } from "@/lib/types";

interface Props { dialogueActs: DialogueActType[] }

const COLORS: Record<string, string> = {
  SYMPTOM_REPORT: "bg-green-100 text-green-800",
  QUESTION: "bg-blue-100 text-blue-800",
  DIAGNOSIS_STATEMENT: "bg-purple-100 text-purple-800",
  TREATMENT_PLAN: "bg-teal-100 text-teal-800",
  REASSURANCE: "bg-slate-200 text-slate-800",
  HISTORY: "bg-amber-100 text-amber-800",
  OTHER: "bg-gray-100 text-gray-600",
};

export function DialogueActStrip({ dialogueActs }: Props) {
  if (!dialogueActs?.length) return null;
  return (
    <div className="card">
      <h2 className="text-sm font-semibold text-slate-900 mb-2">Dialogue Act Sequence</h2>
      <div className="flex flex-wrap gap-1.5">
        {dialogueActs.map((da, i) => (
          <span key={i} className={`text-[11px] font-medium px-2 py-0.5 rounded-full ${COLORS[da.label] ?? COLORS.OTHER}`}>
            {da.speaker === "patient" ? "Pt" : da.speaker === "doctor" ? "Dr" : da.label}:{da.label}
            ({Math.round(da.confidence * 100)}%)
          </span>
        ))}
      </div>
    </div>
  );
}
