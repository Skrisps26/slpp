"use client";
import type { DifferentialDiagnosis } from "@/lib/types";

interface Props {
  differentials: DifferentialDiagnosis[];
}

function getLikelihoodBadge(likelihood: string): string {
  switch (likelihood.toLowerCase()) {
    case "high":
      return "bg-red-100 text-red-800";
    case "moderate":
      return "bg-amber-100 text-amber-800";
    case "low":
      return "bg-green-100 text-green-800";
    default:
      return "bg-slate-100 text-slate-800";
  }
}

export function DifferentialDx({ differentials }: Props) {
  if (differentials.length === 0) return null;

  return (
    <section className="card">
      <h2 className="text-lg font-semibold text-slate-900 mb-4">
        Differential Diagnoses
      </h2>
      <div className="space-y-3">
        {differentials.map((dx, i) => (
          <div
            key={i}
            className="border border-slate-200 rounded-lg p-4 hover:border-medical-300 transition-colors"
          >
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-medium text-slate-900">{dx.diagnosis}</h3>
              <span
                className={`badge ${getLikelihoodBadge(dx.likelihood)}`}
              >
                {dx.likelihood}
              </span>
            </div>
            <p className="text-sm text-slate-600">{dx.evidence}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
