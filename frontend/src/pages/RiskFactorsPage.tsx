import React, { useState } from 'react';
import {
  RADAR_SEGMENTS_V2,
  RISK_FACTORS_T0_DATA,
} from '../data/syntheticCohort';
import { RiskFactorsBarChart } from '../components/charts/RiskFactorsBarChart';
import { RiskRadarChart } from '../components/charts/RiskRadarChart';
import { RiskMatrixHeatmap } from '../components/charts/RiskMatrixHeatmap';

export const RiskFactorsPage: React.FC = () => {
  const [selectedFactorId, setSelectedFactorId] = useState<string>(RISK_FACTORS_T0_DATA[0].id);
  const [categoryFilter, setCategoryFilter] = useState<string>('Todos');
  const [selectedSegmentId, setSelectedSegmentId] = useState<string>(RADAR_SEGMENTS_V2[0].id);

  const categories = ['Todos', 'Psicosocial', 'Geográfico', 'Socioeconómico', 'Clínico'];

  const filteredFactors =
    categoryFilter === 'Todos'
      ? RISK_FACTORS_T0_DATA
      : RISK_FACTORS_T0_DATA.filter((f) => f.category === categoryFilter);

  const activeFactor =
    RISK_FACTORS_T0_DATA.find((f) => f.id === selectedFactorId) || RISK_FACTORS_T0_DATA[0];

  const activeSegment =
    RADAR_SEGMENTS_V2.find((s) => s.id === selectedSegmentId) || RADAR_SEGMENTS_V2[0];

  return (
    <div className="space-y-6">
      <header className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <p className="text-xs font-medium text-teal-800">
            Interpretabilidad oficial `interp_T0` · Cohorte sintética v2 (n = 3,000)
          </p>
          <h1 className="text-2xl font-bold text-slate-900 mt-1 text-balance">
            Factores de riesgo en el modelo seleccionado (`regresion_logistica` T0)
          </h1>
          <p className="text-sm text-slate-600 mt-1 max-w-2xl">
            Basado en importancia por permutación (`interpretabilidad_permutacion.csv`), Odds Ratios
            estandarizados (`interpretabilidad_coeficientes.csv`) y análisis de subgrupos
            (`subgrupos.csv`).
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-1 p-1 bg-slate-100 border border-slate-200 rounded-lg">
          {categories.map((cat) => (
            <button
              key={cat}
              type="button"
              onClick={() => {
                setCategoryFilter(cat);
                const nextList =
                  cat === 'Todos'
                    ? RISK_FACTORS_T0_DATA
                    : RISK_FACTORS_T0_DATA.filter((f) => f.category === cat);
                if (nextList.length > 0 && !nextList.some((f) => f.id === selectedFactorId)) {
                  setSelectedFactorId(nextList[0].id);
                }
              }}
              className={`px-3 py-2 min-h-[40px] text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                categoryFilter === cat
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>
      </header>

      {/* Section 1: Permutation Importance & Odds Ratio Deep Dive */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-5 sm:p-6">
          <div className="mb-4">
            <h2 className="text-base font-semibold text-slate-900">
              Importancia por permutación (`caida_auc_media` en T0)
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Pérdida promedio de ROC-AUC al permutar cada una de las 7 variables de entrada T0
            </p>
          </div>
          <RiskFactorsBarChart
            factors={filteredFactors}
            selectedId={activeFactor.id}
            onSelect={setSelectedFactorId}
          />
        </div>

        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-5 sm:p-6 flex flex-col justify-between gap-5">
          <div className="space-y-4">
            <div>
              <span className="text-xs text-slate-500">
                Variable `{activeFactor.variable}` · {activeFactor.category}
              </span>
              <h2 className="text-lg font-bold text-slate-900 mt-1">{activeFactor.name}</h2>
            </div>

            <div className="grid grid-cols-3 gap-2.5 pt-2 border-t border-slate-100">
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <span className="text-[11px] text-slate-500 block">Caída AUC</span>
                <span className="text-base font-mono font-bold text-slate-900 tabular-nums">
                  {activeFactor.aucDrop.toFixed(4)}
                </span>
              </div>
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <span className="text-[11px] text-slate-500 block">OR por DE</span>
                <span className="text-base font-mono font-bold text-teal-800 tabular-nums">
                  {activeFactor.oddsRatioSd.toFixed(3)}×
                </span>
              </div>
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <span className="text-[11px] text-slate-500 block">Estabilidad</span>
                <span className="text-base font-mono font-bold text-slate-900 tabular-nums">
                  {activeFactor.signStability}%
                </span>
              </div>
            </div>

            <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg space-y-2 text-xs">
              <div className="flex justify-between gap-3">
                <span className="text-slate-500">Coeficiente estandarizado (T0):</span>
                <span className="font-mono font-semibold text-slate-900 tabular-nums">
                  {activeFactor.coefStd > 0 ? `+${activeFactor.coefStd}` : activeFactor.coefStd}{' '}
                  (IC 95%: {activeFactor.ic95[0]}–{activeFactor.ic95[1]})
                </span>
              </div>
              <div className="flex justify-between gap-3">
                <span className="text-red-800 font-medium">Desertores (n=2,142):</span>
                <span className="font-mono text-red-900 text-right">
                  {activeFactor.desertorMeanLabel}
                </span>
              </div>
              <div className="flex justify-between gap-3">
                <span className="text-emerald-800 font-medium">Continuadores (n=858):</span>
                <span className="font-mono text-emerald-900 text-right">
                  {activeFactor.continuadorMeanLabel}
                </span>
              </div>
            </div>

            <div className="space-y-1.5">
              <h3 className="text-xs font-semibold text-slate-900">Evidencia estadística en v2</h3>
              <p className="text-sm text-slate-600 leading-relaxed">
                {activeFactor.clinicalDescription}
              </p>
            </div>
          </div>

          <div className="p-4 bg-teal-50/70 border border-teal-200 rounded-lg">
            <h3 className="text-xs font-semibold text-teal-900">
              Acción clínica recomendada
            </h3>
            <p className="text-sm text-teal-950 mt-1 leading-relaxed">
              {activeFactor.mitigationProtocol}
            </p>
          </div>
        </div>
      </section>

      {/* Section 2: Subgroups Radar + Distance Matrix from dataset_modelado_v2.csv */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-5 sm:p-6 space-y-4">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Subgrupos reales (`reports/modelos/subgrupos.csv`)
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Comparación de prevalencia, carga de factores y desempeño T0 frente a la cohorte v2
            </p>
          </div>

          <div className="flex flex-wrap gap-1.5">
            {RADAR_SEGMENTS_V2.map((seg) => (
              <button
                key={seg.id}
                type="button"
                onClick={() => setSelectedSegmentId(seg.id)}
                className={`px-3 py-2 min-h-[40px] text-xs font-medium rounded-lg border transition-colors ${
                  selectedSegmentId === seg.id
                    ? 'bg-teal-700 text-white border-teal-700'
                    : 'bg-white text-slate-700 border-slate-200 hover:bg-slate-50'
                }`}
              >
                {seg.label}
              </button>
            ))}
          </div>

          <p className="text-xs text-slate-600">{activeSegment.description}</p>

          <RiskRadarChart segment={activeSegment} />
        </div>

        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-5 sm:p-6 space-y-4">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Matriz empírica: Cuartiles de distancia × Apoyo psicológico / SES
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Tasas reales observadas en `dataset_modelado_v2.csv` (sin columnas con fuga de datos)
            </p>
          </div>

          <RiskMatrixHeatmap />
        </div>
      </section>
    </div>
  );
};
