"use client";
import type { Differential } from "@/lib/types";

interface Props { differentials: Differential[] }

const badge = (l: string) => {
  const u = l.toLowerCase();
  if (u === "high") return "bg-red-100 text-red-700";
  if (u === "moderate") return "bg-amber-100 text-amber-700";
  return "bg-green-100 text-green-700";
};

export function DifferentialDx({ differentials }: Props) {
  if (!differentials?.length) return null;
  return (
    <div className="card">
      <h2 className="text-lg font-semibold text-slate-900 mb-4">Differential Diagnoses</h2>
      <div className="space-y-3">
        {differentials.map((dx, i) => (
          <div key={i} className="border border-slate-200 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-medium text-slate-900">{dx.diagnosis}</h3>
              <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${badge(dx.likelihood)}`}>
                {dx.likelihood}
              </span>
            </div>
            <p className="text-sm text-slate-600">{dx.evidence}</p>
            {dx.kb_source && <p className="text-xs text-slate-400 mt-1 italic">Source: {dx.kb_source}</p>}
          </div>
        ))}
      </div>
    </div>
  );
}
