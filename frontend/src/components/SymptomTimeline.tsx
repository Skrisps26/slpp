"use client";
import type { ClinicalEntity, TemporalEventType } from "@/lib/types";

interface Props { symptoms: ClinicalEntity[]; temporalEvents: TemporalEventType[] }

export function SymptomTimeline({ temporalEvents }: Props) {
  if (!temporalEvents?.length) return null;
  return (
    <div className="card">
      <h2 className="text-lg font-semibold text-slate-900 mb-4">Symptom Timeline</h2>
      <div className="relative pl-10">
        <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-slate-200" />
        <div className="space-y-4">
          {temporalEvents.map((ev, i) => (
            <div key={i} className="relative">
              <div className="absolute -left-6 top-1 w-3 h-3 rounded-full bg-medical-400 border-2 border-white shadow" />
              <div>
                <span className="text-sm font-medium text-slate-700">{ev.text}</span>
                <span className="ml-2 text-xs text-slate-500">({ev.normalized})</span>
                {ev.type && (
                  <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded text-xs bg-slate-100 text-slate-600">
                    {ev.type}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
