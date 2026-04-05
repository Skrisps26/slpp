"use client";

interface FaithfulnessGaugeProps {
  score: number;
  totalSentences: number;
  entailedCount: number;
  hallucinatedCount: number;
  refinementIterations: number;
}

function polarToCartesian(cx: number, cy: number, r: number, angleDeg: number) {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function arcPath(cx: number, cy: number, r: number, startAngle: number, endAngle: number) {
  const s = polarToCartesian(cx, cy, r, startAngle);
  const e = polarToCartesian(cx, cy, r, endAngle);
  const large = endAngle - startAngle > 180 ? 1 : 0;
  return `M ${s.x} ${s.y} A ${r} ${r} 0 ${large} 1 ${e.x} ${e.y}`;
}

export function FaithfulnessGauge({
  score, totalSentences, entailedCount, hallucinatedCount, refinementIterations,
}: FaithfulnessGaugeProps) {
  const pct = Math.round(score * 100);
  const sweep = 270;
  const startAngle = 135;
  const endAngle = startAngle + (sweep * score);
  const color = pct >= 85 ? "#16a34a" : pct >= 70 ? "#d97706" : "#dc2626";
  const bgPath = arcPath(60, 60, 50, 135, 405);
  const fillPath = arcPath(60, 60, 50, 135, endAngle);

  return (
    <div className="card">
      <div className="flex flex-col sm:flex-row items-center gap-6">
        <svg viewBox="0 0 120 120" className="w-36 h-36 flex-shrink-0">
          <path d={bgPath} fill="none" stroke="#e2e8f0" strokeWidth="10" strokeLinecap="round" />
          <path d={fillPath} fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"
            style={{ transition: "stroke-dasharray 0.5s ease" }} />
          <text x="60" y="55" textAnchor="middle" fontSize="20" fontWeight="bold" fill={color}>
            {pct}%
          </text>
          <text x="60" y="72" textAnchor="middle" fontSize="9" fill="#94a3b8">
            Faithfulness
          </text>
        </svg>

        <div className="flex-1 space-y-1.5 text-sm">
          <p className="text-slate-700 font-medium">{entailedCount}/{totalSentences} sentences verified</p>
          {hallucinatedCount > 0 && (
            <p className="text-amber-600">⚠ {hallucinatedCount} auto-corrected</p>
          )}
          {refinementIterations > 0 && (
            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-medical-100 text-medical-700">
              Refined {refinementIterations}×
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
