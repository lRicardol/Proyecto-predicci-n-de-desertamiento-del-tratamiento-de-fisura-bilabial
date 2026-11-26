import React from 'react';
import { RiskFactorInsight } from '../../types/clinical';

interface RiskFactorsBarChartProps {
  factors: RiskFactorInsight[];
  selectedId: string;
  onSelect: (id: string) => void;
}

export const RiskFactorsBarChart: React.FC<RiskFactorsBarChartProps> = ({
  factors,
  selectedId,
  onSelect,
}) => {
  const maxVal = Math.max(...factors.map((f) => Math.max(0.002, f.aucDrop)));

  return (
    <div className="space-y-3">
      {factors.map((factor) => {
        const isSelected = factor.id === selectedId;
        const barWidthPct = Math.max(2, (Math.max(0, factor.aucDrop) / maxVal) * 100);

        return (
          <button
            key={factor.id}
            type="button"
            onClick={() => onSelect(factor.id)}
            className={`w-full text-left p-3.5 min-h-[56px] rounded-lg border transition-colors ${
              isSelected
                ? 'bg-teal-50/70 border-teal-600'
                : 'bg-white border-slate-200 hover:border-slate-300 hover:bg-slate-50'
            }`}
          >
            <div className="flex items-center justify-between gap-4 mb-2">
              <div className="min-w-0">
                <span className="text-sm font-semibold text-slate-900 block truncate">
                  {factor.name}
                </span>
                <span className="text-xs text-slate-500">
                  {factor.category} · OR por DE {factor.oddsRatioSd.toFixed(3)}× · Estabilidad{' '}
                  {factor.signStability}%
                </span>
              </div>
              <div className="text-right shrink-0">
                <span className="text-sm font-mono font-semibold text-slate-900 tabular-nums">
                  {factor.aucDrop >= 0 ? `+${factor.aucDrop.toFixed(4)}` : factor.aucDrop.toFixed(4)}
                </span>
                <span className="text-[11px] text-slate-500 block">caída AUC</span>
              </div>
            </div>

            <div className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-200 ${
                  isSelected ? 'bg-teal-700' : 'bg-slate-400'
                }`}
                style={{ width: `${barWidthPct}%` }}
              />
            </div>
          </button>
        );
      })}
    </div>
  );
};
