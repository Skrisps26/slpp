"use client";
import type { Entity, TemporalEvent } from "@/lib/types";

interface Props {
  symptoms: Entity[];
  temporalEvents: TemporalEvent[];
}

export function SymptomTimeline({ symptoms, temporalEvents }: Props) {
  if (temporalEvents.length === 0 && symptoms.length === 0) return null;

  return (
    <section className="card">
      <h2 className="text-lg font-semibold text-slate-900 mb-4">
        Symptom Timeline
      </h2>
      <div className="relative">
        <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-slate-200" />
        <div className="space-y-4 ml-10">
          {temporalEvents.map((event, i) => (
            <div key={i} className="relative">
              <div className="absolute -left-6 top-1 w-3 h-3 rounded-full bg-medical-400 border-2 border-white shadow" />
              <div>
                <span className="text-sm font-medium text-slate-700">
                  {event.text}
                </span>
                <span className="ml-2 text-xs text-slate-500">
                  ({event.normalized})
                </span>
                {event.temporal_type && (
                  <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-slate-100 text-slate-600">
                    {event.temporal_type}
                  </span>
                )}
              </div>
            </div>
          ))}
          {symptoms.map((symptom, i) => (
            <div key={`s-${i}`} className="relative">
              <div className="absolute -left-6 top-1 w-3 h-3 rounded-full bg-red-400 border-2 border-white shadow" />
              <div>
                <span className="text-sm font-medium text-slate-700">
                  {symptom.text}
                </span>
                {symptom.negated && (
                  <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-600 line-through">
                    Negated
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
