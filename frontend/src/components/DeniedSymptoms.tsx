"use client";
import type { ClinicalEntity } from "@/lib/types";

interface Props { entities: ClinicalEntity[] }

export function DeniedSymptoms({ entities }: Props) {
  const denied = entities.filter((e) => e.negated);
  if (!denied.length) return null;

  return (
    <div className="card">
      <h2 className="text-lg font-semibold text-slate-900 mb-4">Denied Symptoms</h2>
      <div className="flex flex-wrap gap-2">
        {denied.map((e, i) => (
          <span key={i} className="bg-red-50 text-red-400 line-through px-3 py-1 rounded-full text-sm border border-red-200">
            {e.text}
          </span>
        ))}
      </div>
    </div>
  );
}
