import React, { useState } from 'react';
import { REPO_MODEL_METADATA, T0_CALIBRATION_BINS } from '../../data/syntheticCohort';

export const RetentionCurveChart: React.FC = () => {
  const [activeBinIdx, setActiveBinIdx] = useState<number>(8);

  const width = 760;
  const height = 270;
  const padLeft = 52;
  const padRight = 28;
  const padTop = 24;
  const padBottom = 44;
  const innerW = width - padLeft - padRight;
  const innerH = height - padTop - padBottom;

  const points = T0_CALIBRATION_BINS.map((item, idx) => {
    const x = padLeft + (idx / (T0_CALIBRATION_BINS.length - 1)) * innerW;
    const yPred = padTop + ((100 - item.predicted) / 100) * innerH;
    const yObs = padTop + ((100 - item.observed) / 100) * innerH;
    return { ...item, idx, x, yPred, yObs };
  });

  const obsPath = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.yObs.toFixed(1)}`)
    .join(' ');

  const obsArea = `${obsPath} L ${points[points.length - 1].x.toFixed(1)} ${(
    padTop + innerH
  ).toFixed(1)} L ${points[0].x.toFixed(1)} ${(padTop + innerH).toFixed(1)} Z`;

  const predPath = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.yPred.toFixed(1)}`)
    .join(' ');

  const activePoint = points[activeBinIdx];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h3 className="text-base font-semibold text-slate-900">
            Calibración del modelo seleccionado (Regresión Logística T0 · `calibracion_cajas.csv`)
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Deciles de riesgo predicho vs. tasa real observada en `dataset_modelado_v2.csv` (ECE ={' '}
            {REPO_MODEL_METADATA.eceT0}) · Toca un decil para inspeccionar
          </p>
        </div>
        <div className="flex items-center gap-5 text-xs text-slate-600">
          <span className="inline-flex items-center gap-2">
            <span className="w-3 h-0.5 bg-teal-700 inline-block" aria-hidden="true" />
            Deserción observada (%)
          </span>
          <span className="inline-flex items-center gap-2">
            <span
              className="w-3 h-0.5 border-t-2 border-dashed border-amber-600 inline-block"
              aria-hidden="true"
            />
            Probabilidad predicha T0 (%)
          </span>
        </div>
      </div>

      <div className="w-full overflow-x-auto">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto min-w-[560px] select-none"
          role="img"
          aria-label="Curva de calibración por deciles del modelo T0"
        >
          <defs>
            <linearGradient id="calAreaGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#0f766e" stopOpacity="0.16" />
              <stop offset="100%" stopColor="#0f766e" stopOpacity="0.01" />
            </linearGradient>
          </defs>

          {[0, 25, 50, 75, 100].map((tick) => {
            const y = padTop + ((100 - tick) / 100) * innerH;
            return (
              <g key={tick}>
                <line
                  x1={padLeft}
                  y1={y}
                  x2={width - padRight}
                  y2={y}
                  stroke="#e2e8f0"
                  strokeWidth="1"
                />
                <text
                  x={padLeft - 10}
                  y={y + 4}
                  textAnchor="end"
                  className="fill-slate-400 text-[11px] font-mono tabular-nums"
                >
                  {tick}%
                </text>
              </g>
            );
          })}

          <line
            x1={activePoint.x}
            y1={padTop}
            x2={activePoint.x}
            y2={padTop + innerH}
            stroke="#94a3b8"
            strokeWidth="1"
            strokeDasharray="3 3"
          />

          <path d={obsArea} fill="url(#calAreaGrad)" />
          <path d={obsPath} fill="none" stroke="#0f766e" strokeWidth="2.5" strokeLinejoin="round" />
          <path
            d={predPath}
            fill="none"
            stroke="#d97706"
            strokeWidth="2"
            strokeDasharray="5 4"
            strokeLinejoin="round"
          />

          {points.map((p) => {
            const isActive = p.idx === activeBinIdx;
            return (
              <g
                key={p.bin}
                className="cursor-pointer"
                onClick={() => setActiveBinIdx(p.idx)}
                onMouseEnter={() => setActiveBinIdx(p.idx)}
              >
                <rect x={p.x - 28} y={padTop} width={56} height={innerH + 34} fill="transparent" />
                <circle
                  cx={p.x}
                  cy={p.yObs}
                  r={isActive ? 6 : 4}
                  fill={isActive ? '#0f766e' : '#ffffff'}
                  stroke="#0f766e"
                  strokeWidth="2.5"
                />
                <circle
                  cx={p.x}
                  cy={p.yPred}
                  r={isActive ? 5.5 : 3.5}
                  fill={isActive ? '#d97706' : '#ffffff'}
                  stroke="#d97706"
                  strokeWidth="2"
                />
                <text
                  x={p.x}
                  y={padTop + innerH + 22}
                  textAnchor="middle"
                  className={`text-[11px] font-mono ${
                    isActive ? 'fill-slate-900 font-semibold' : 'fill-slate-500'
                  }`}
                >
                  D{p.bin}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-3 border-t border-slate-100">
        <div>
          <span className="text-xs text-slate-500 block">Decil inspeccionado</span>
          <span className="text-sm font-semibold text-slate-900">
            Decil {activePoint.bin} de 10 (n = 720 evaluaciones CV)
          </span>
        </div>
        <div>
          <span className="text-xs text-slate-500 block">Probabilidad predicha media</span>
          <span className="text-lg font-mono font-semibold text-amber-700 tabular-nums">
            {activePoint.predicted.toFixed(1)}%
          </span>
        </div>
        <div>
          <span className="text-xs text-slate-500 block">Deserción real observada</span>
          <span className="text-lg font-mono font-semibold text-teal-800 tabular-nums">
            {activePoint.observed.toFixed(1)}%
          </span>
        </div>
        <div>
          <span className="text-xs text-slate-500 block">Desviación de calibración</span>
          <span className="text-sm font-mono font-semibold text-slate-900 tabular-nums">
            {activePoint.observed - activePoint.predicted >= 0 ? '+' : ''}
            {(activePoint.observed - activePoint.predicted).toFixed(1)} pts
          </span>
          <span className="text-xs text-slate-500 block mt-0.5">
            Las probabilidades de `modelo_t0` reflejan riesgo real sin sobreajuste.
          </span>
        </div>
      </div>
    </div>
  );
};
