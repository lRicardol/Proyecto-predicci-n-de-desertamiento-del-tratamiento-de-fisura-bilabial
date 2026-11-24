import React, { useState } from 'react';
import {
  MATRIX_DISTANCE_SES,
  MATRIX_DISTANCE_SUPPORT,
} from '../../data/syntheticCohort';
import { MatrixCell } from '../../types/clinical';

export const RiskMatrixHeatmap: React.FC = () => {
  const [mode, setMode] = useState<'ses' | 'support'>('support');
  const [selectedCell, setSelectedCell] = useState<MatrixCell>(MATRIX_DISTANCE_SUPPORT[6]);

  const distances = [
    'Q1 (≤ 19.1 km)',
    'Q2 (19.1–40.2 km)',
    'Q3 (40.2–75.2 km)',
    'Q4 (> 75.2 km)',
  ];

  const cols =
    mode === 'ses'
      ? ['SES 0 (Bajo)', 'SES 1 (Medio-Bajo)', 'SES 2 (Medio)', 'SES 4 (Alto)']
      : ['Sin apoyo (0)', 'Con apoyo (1)'];

  const data = mode === 'ses' ? MATRIX_DISTANCE_SES : MATRIX_DISTANCE_SUPPORT;

  const handleModeChange = (nextMode: 'ses' | 'support') => {
    setMode(nextMode);
    setSelectedCell(nextMode === 'ses' ? MATRIX_DISTANCE_SES[12] : MATRIX_DISTANCE_SUPPORT[6]);
  };

  const getCellStyle = (cell: MatrixCell, isSelected: boolean) => {
    const ring = isSelected ? 'ring-2 ring-slate-900 ring-offset-1' : '';
    if (cell.prioridad === 'Alta') {
      return `bg-red-50 border-red-200 text-red-900 hover:bg-red-100/80 ${ring}`;
    }
    if (cell.prioridad === 'Media') {
      return `bg-amber-50 border-amber-200 text-amber-900 hover:bg-amber-100/80 ${ring}`;
    }
    return `bg-emerald-50 border-emerald-200 text-emerald-900 hover:bg-emerald-100/80 ${ring}`;
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-xs text-slate-500">
          Cruce exacto sobre `dataset_modelado_v2.csv` (n = 3,000)
        </span>
        <div className="flex items-center gap-1 p-1 bg-slate-100 border border-slate-200 rounded-lg">
          <button
            type="button"
            onClick={() => handleModeChange('support')}
            className={`px-3 py-1.5 min-h-[36px] text-xs font-medium rounded-md transition-colors ${
              mode === 'support' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600'
            }`}
          >
            Distancia × Apoyo Psicológico
          </button>
          <button
            type="button"
            onClick={() => handleModeChange('ses')}
            className={`px-3 py-1.5 min-h-[36px] text-xs font-medium rounded-md transition-colors ${
              mode === 'ses' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600'
            }`}
          >
            Distancia × Nivel SES
          </button>
        </div>
      </div>

      <div className="overflow-x-auto">
        <div className="min-w-[500px]">
          <div
            className={`grid gap-2 mb-2 ${
              mode === 'ses'
                ? 'grid-cols-[135px_repeat(4,minmax(0,1fr))]'
                : 'grid-cols-[135px_repeat(2,minmax(0,1fr))]'
            }`}
          >
            <div className="text-xs font-medium text-slate-500 flex items-end pb-1">
              Cuartil distancia
            </div>
            {cols.map((col) => (
              <div key={col} className="text-xs font-semibold text-slate-700 text-center pb-1">
                {col}
              </div>
            ))}
          </div>

          <div className="space-y-2">
            {distances.map((dist) => (
              <div
                key={dist}
                className={`grid gap-2 items-stretch ${
                  mode === 'ses'
                    ? 'grid-cols-[135px_repeat(4,minmax(0,1fr))]'
                    : 'grid-cols-[135px_repeat(2,minmax(0,1fr))]'
                }`}
              >
                <div className="text-xs font-semibold text-slate-700 flex items-center font-mono">
                  {dist}
                </div>
                {cols.map((col) => {
                  const cell = data.find(
                    (c) => c.distanceLabel === dist && c.colLabel === col
                  )!;
                  const isSelected =
                    selectedCell.distanceLabel === cell.distanceLabel &&
                    selectedCell.colLabel === cell.colLabel;

                  return (
                    <button
                      key={`${dist}-${col}`}
                      type="button"
                      onClick={() => setSelectedCell(cell)}
                      className={`p-2.5 min-h-[58px] rounded-lg border text-center transition-all flex flex-col items-center justify-center ${getCellStyle(
                        cell,
                        isSelected
                      )}`}
                    >
                      <span className="text-base font-mono font-bold tabular-nums leading-tight">
                        {cell.dropoutRate.toFixed(1)}%
                      </span>
                      <span className="text-[11px] font-medium opacity-85 mt-0.5">
                        {cell.prioridad} · n={cell.syntheticCount}
                      </span>
                    </button>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <div className="text-xs text-slate-500">
            Cruce seleccionado:{' '}
            <span className="font-semibold text-slate-800">{selectedCell.distanceLabel}</span> ·{' '}
            <span className="font-semibold text-slate-800">{selectedCell.colLabel}</span>
          </div>
          <p className="text-sm text-slate-800 font-medium mt-1">{selectedCell.actionSummary}</p>
        </div>
        <div className="shrink-0 text-left sm:text-right font-mono tabular-nums">
          <span className="text-xs text-slate-500 block">Deserción observada v2</span>
          <span className="text-lg font-bold text-slate-900">
            {selectedCell.dropoutRate.toFixed(1)}% ({selectedCell.prioridad})
          </span>
        </div>
      </div>
    </div>
  );
};
