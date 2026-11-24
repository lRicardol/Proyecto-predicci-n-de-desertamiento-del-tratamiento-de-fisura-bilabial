import React, { useState } from 'react';
import { RadarSegmentProfile } from '../../types/clinical';

interface RiskRadarChartProps {
  segment: RadarSegmentProfile;
}

export const RiskRadarChart: React.FC<RiskRadarChartProps> = ({ segment }) => {
  const [focusedAxisIndex, setFocusedAxisIndex] = useState<number>(0);

  const size = 340;
  const cx = size / 2;
  const cy = size / 2;
  const radius = 112;
  const totalAxes = segment.axes.length;

  const getPoint = (axisIdx: number, valuePct: number) => {
    const angle = (Math.PI * 2 * axisIdx) / totalAxes - Math.PI / 2;
    const r = (valuePct / 100) * radius;
    return {
      x: cx + Math.cos(angle) * r,
      y: cy + Math.sin(angle) * r,
    };
  };

  const rings = [25, 50, 75, 100];

  const segmentPolygon = segment.axes
    .map((a, idx) => {
      const pt = getPoint(idx, a.segmentValue);
      return `${pt.x.toFixed(1)},${pt.y.toFixed(1)}`;
    })
    .join(' ');

  const baselinePolygon = segment.axes
    .map((a, idx) => {
      const pt = getPoint(idx, a.cohortBaseline);
      return `${pt.x.toFixed(1)},${pt.y.toFixed(1)}`;
    })
    .join(' ');

  const activeAxis = segment.axes[focusedAxisIndex];

  return (
    <div className="space-y-4">
      <div className="flex flex-col items-center">
        <svg
          viewBox={`0 0 ${size} ${size}`}
          className="w-full max-w-[330px] h-auto select-none"
          role="img"
          aria-label={`Gráfico radar de subgrupo: ${segment.label}`}
        >
          {rings.map((ring) => {
            const pts = Array.from({ length: totalAxes })
              .map((_, i) => {
                const p = getPoint(i, ring);
                return `${p.x.toFixed(1)},${p.y.toFixed(1)}`;
              })
              .join(' ');
            return (
              <polygon
                key={ring}
                points={pts}
                fill="none"
                stroke="#e2e8f0"
                strokeWidth="1"
              />
            );
          })}

          {segment.axes.map((axis, idx) => {
            const outer = getPoint(idx, 100);
            const labelPos = getPoint(idx, 123);
            const isFocused = idx === focusedAxisIndex;
            return (
              <g
                key={axis.axis}
                className="cursor-pointer"
                onClick={() => setFocusedAxisIndex(idx)}
                onMouseEnter={() => setFocusedAxisIndex(idx)}
              >
                <line
                  x1={cx}
                  y1={cy}
                  x2={outer.x}
                  y2={outer.y}
                  stroke={isFocused ? '#0f766e' : '#cbd5e1'}
                  strokeWidth={isFocused ? '1.75' : '1'}
                />
                <text
                  x={labelPos.x}
                  y={labelPos.y + 4}
                  textAnchor="middle"
                  className={`text-[11px] font-medium ${
                    isFocused ? 'fill-teal-800 font-semibold' : 'fill-slate-600'
                  }`}
                >
                  {axis.shortAxis}
                </text>
              </g>
            );
          })}

          <polygon
            points={baselinePolygon}
            fill="#64748b"
            fillOpacity="0.12"
            stroke="#64748b"
            strokeWidth="1.5"
            strokeDasharray="4 3"
          />

          <polygon
            points={segmentPolygon}
            fill="#0f766e"
            fillOpacity="0.18"
            stroke="#0f766e"
            strokeWidth="2.2"
          />

          {segment.axes.map((axis, idx) => {
            const pt = getPoint(idx, axis.segmentValue);
            const isFocused = idx === focusedAxisIndex;
            return (
              <circle
                key={axis.axis}
                cx={pt.x}
                cy={pt.y}
                r={isFocused ? 6 : 4}
                fill={isFocused ? '#0f766e' : '#ffffff'}
                stroke="#0f766e"
                strokeWidth="2"
                className="cursor-pointer"
                onClick={() => setFocusedAxisIndex(idx)}
                onMouseEnter={() => setFocusedAxisIndex(idx)}
              />
            );
          })}
        </svg>
      </div>

      <div className="flex flex-wrap items-center justify-center gap-5 text-xs text-slate-600">
        <span className="inline-flex items-center gap-2">
          <span className="w-3 h-3 rounded-sm bg-teal-700/20 border border-teal-700 inline-block" />
          Subgrupo ({segment.n} pac. · ROC-AUC {segment.rocAuc.toFixed(3)})
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="w-3 h-3 rounded-sm bg-slate-500/15 border border-dashed border-slate-500 inline-block" />
          Cohorte v2 (n=3,000)
        </span>
      </div>

      <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg flex items-center justify-between gap-4">
        <div>
          <span className="text-xs text-slate-500 block">Dimensión inspeccionada</span>
          <span className="text-sm font-semibold text-slate-900">{activeAxis.axis}</span>
        </div>
        <div className="flex items-center gap-5 text-right font-mono tabular-nums">
          <div>
            <span className="text-[11px] text-slate-500 block">Subgrupo</span>
            <span className="text-sm font-semibold text-teal-800">
              {activeAxis.segmentValue.toFixed(1)}
              {activeAxis.unit}
            </span>
          </div>
          <div>
            <span className="text-[11px] text-slate-500 block">Cohorte v2</span>
            <span className="text-sm font-medium text-slate-600">
              {activeAxis.cohortBaseline.toFixed(1)}
              {activeAxis.unit}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
