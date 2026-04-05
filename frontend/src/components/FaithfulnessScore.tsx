"use client";

interface Props {
  score: number;
  total: number;
  refinements: number;
  hallucinated: Array<{ soap_sentence: string; soap_section: string }>;
}

function getScoreColor(score: number): string {
  if (score >= 0.85) return "text-green-500";
  if (score >= 0.70) return "text-amber-500";
  return "text-red-500";
}

function getScoreBg(score: number): string {
  if (score >= 0.85) return "stroke-green-500";
  if (score >= 0.70) return "stroke-amber-500";
  return "stroke-red-500";
}

export function FaithfulnessScore({ score, total, refinements, hallucinated }: Props) {
  const percentage = Math.round(score * 100);
  const circumference = 2 * Math.PI * 54;
  const offset = circumference - (score * circumference);

  return (
    <section className="card">
      <h2 className="text-lg font-semibold text-slate-900 mb-4">
        Verification Results
      </h2>
      <div className="flex flex-col sm:flex-row items-center gap-6">
        {/* Circular gauge */}
        <div className="relative w-32 h-32 flex-shrink-0">
          <svg className="w-32 h-32 -rotate-90" viewBox="0 0 120 120">
            <circle
              cx="60"
              cy="60"
              r="54"
              fill="none"
              stroke="#e2e8f0"
              strokeWidth="8"
            />
            <circle
              cx="60"
              cy="60"
              r="54"
              fill="none"
              className={getScoreBg(score)}
              strokeWidth="8"
              strokeDasharray={circumference}
              strokeDashoffset={offset}
              strokeLinecap="round"
              style={{ transition: "stroke-dashoffset 0.5s ease" }}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className={`text-2xl font-bold ${getScoreColor(score)}`}>
              {percentage}%
            </span>
            <span className="text-xs text-slate-500">Faithful</span>
          </div>
        </div>

        <div className="flex-1 space-y-2">
          <p className="text-slate-700">
            {Math.round(score * total)} of {total} sentences verified against transcript
          </p>
          {refinements > 0 && (
            <p className="text-sm text-amber-600 font-medium">
              Auto-corrected {refinements} time{refinements > 1 ? "s" : ""}
            </p>
          )}
          {hallucinated.length > 0 && (
            <div className="space-y-1 mt-2">
              <p className="text-sm font-medium text-red-600">
                Warning: {hallucinated.length} hallucinated sentence{hallucinated.length > 1 ? "s" : ""}
              </p>
              {hallucinated.map((h, i) => (
                <div
                  key={i}
                  className="text-xs text-red-500 bg-red-50 border border-red-100 rounded px-2 py-1"
                >
                  [<span className="font-medium">{h.soap_section}</span>] {h.soap_sentence}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
