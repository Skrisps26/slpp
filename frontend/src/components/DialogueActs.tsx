"use client";
import type { DialogueAct } from "@/lib/types";

interface Props {
  dialogueActs: DialogueAct[];
}

const LABEL_STYLES: Record<string, string> = {
  SYMPTOM_REPORT: "bg-green-100 text-green-800 border-green-200",
  QUESTION: "bg-blue-100 text-blue-800 border-blue-200",
  DIAGNOSIS_STATEMENT: "bg-purple-100 text-purple-800 border-purple-200",
  TREATMENT_PLAN: "bg-teal-100 text-teal-800 border-teal-200",
  REASSURANCE: "bg-gray-100 text-gray-800 border-gray-200",
  HISTORY: "bg-amber-100 text-amber-800 border-amber-200",
  OTHER: "bg-slate-100 text-slate-800 border-slate-200",
};

export function DialogueActs({ dialogueActs }: Props) {
  if (dialogueActs.length === 0) return null;

  return (
    <section className="card">
      <h2 className="text-lg font-semibold text-slate-900 mb-4">
        Dialogue Act Sequence
      </h2>
      <div className="flex flex-wrap gap-2">
        {dialogueActs.map((da, i) => (
          <div
            key={i}
            className={`badge border ${LABEL_STYLES[da.label] || LABEL_STYLES.OTHER}`}
          >
            <span className="font-medium">{da.label}</span>
            <span className="ml-1 opacity-75">
              ({(da.confidence * 100).toFixed(0)}%)
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}
