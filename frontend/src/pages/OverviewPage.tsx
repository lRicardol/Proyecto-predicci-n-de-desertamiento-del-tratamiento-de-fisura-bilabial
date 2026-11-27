import React from 'react';
import { ArrowRight, BarChart3, Calculator, ClipboardCheck, ShieldCheck } from 'lucide-react';
import {
  MODEL_COMPARISON_TABLE,
  REPO_MODEL_METADATA,
  REPO_SAMPLE_PATIENTS,
  predictT0Logistic,
} from '../data/syntheticCohort';
import { PageId } from '../types/clinical';
import { RetentionCurveChart } from '../components/charts/RetentionCurveChart';
import { RiskLabel } from '../components/RiskStatusText';
import heroImage from '../assets/images/clinical_care_hero_1795476472432.jpg';

interface OverviewPageProps {
  onNavigate: (page: PageId) => void;
  specialistSignedIn: boolean;
  assignedCasesCount: number;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({
  onNavigate,
  specialistSignedIn,
  assignedCasesCount,
}) => {
  return (
    <div className="space-y-8">
      {/* Hero Header Split */}
      <section className="bg-white border border-slate-200 rounded-xl overflow-hidden">
        <div className="grid grid-cols-1 lg:grid-cols-12 items-stretch">
          <div className="lg:col-span-7 p-6 sm:p-8 flex flex-col justify-between gap-6">
            <div>
              <p className="text-xs font-medium text-teal-800 tracking-wide">
                Modelo seleccionado del repositorio: {REPO_MODEL_METADATA.modelName} ·{' '}
                {REPO_MODEL_METADATA.scenario}
              </p>
              <h1 className="text-2xl sm:text-3xl font-bold text-slate-900 mt-2 text-balance leading-tight">
                Predicción de deserción en fisura bilabial con cohorte sintética v2
              </h1>
              <p className="text-sm sm:text-base text-slate-600 mt-3 leading-relaxed max-w-2xl">
                Interfaz clínica basada directamente en `dataset_modelado_v2.csv` (3,000 pacientes
                sintéticos) y en el artefacto `modelo_t0` entrenado sin fuga de datos sobre las 7
                variables disponibles al ingreso.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={() => onNavigate('factores')}
                className="inline-flex items-center gap-2 px-4 py-2.5 min-h-[44px] bg-teal-700 hover:bg-teal-800 text-white text-sm font-medium rounded-lg transition-colors whitespace-nowrap shrink-0"
              >
                <BarChart3 className="w-4 h-4" />
                Explorar factores T0
              </button>
              <button
                type="button"
                onClick={() => onNavigate('simulador')}
                className="inline-flex items-center gap-2 px-4 py-2.5 min-h-[44px] bg-white hover:bg-slate-50 text-slate-800 border border-slate-300 text-sm font-medium rounded-lg transition-colors whitespace-nowrap shrink-0"
              >
                <Calculator className="w-4 h-4" />
                Simulador modelo T0
              </button>
            </div>
          </div>

          <div className="lg:col-span-5 relative min-h-[220px] bg-slate-100 border-t lg:border-t-0 lg:border-l border-slate-200">
            <img
              src={heroImage}
              alt="Sala de consulta pediátrica craneofacial con iluminación natural"
              referrerPolicy="no-referrer"
              className="w-full h-full object-cover lg:absolute lg:inset-0"
            />
            <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-slate-950/75 via-slate-900/35 to-transparent p-4 sm:p-5">
              <p className="text-xs text-white/95 font-medium">
                Cohorte v2 (n = 3,000) · Sin fuga temporal (`EXCLUDED_LEAKAGE`)
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 4-column KPI strip */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <span className="text-xs text-slate-500 block">Cohorte sintética v2</span>
          <span className="text-2xl font-mono font-bold text-slate-900 tabular-nums mt-1 block">
            {REPO_MODEL_METADATA.totalPatients.toLocaleString('es-CO')}
          </span>
          <span className="text-xs text-slate-500 mt-1 block tabular-nums">
            {REPO_MODEL_METADATA.desertorCount.toLocaleString('es-CO')} desertores ·{' '}
            {REPO_MODEL_METADATA.continuadorCount} continuadores
          </span>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <span className="text-xs text-slate-500 block">Tasa de deserción observada</span>
          <span className="text-2xl font-mono font-bold text-red-700 tabular-nums mt-1 block">
            {REPO_MODEL_METADATA.desertorRatePct}%
          </span>
          <span className="text-xs text-slate-500 mt-1 block">
            Prevalencia en `dataset_modelado_v2.csv`
          </span>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <span className="text-xs text-slate-500 block">Rendimiento `modelo_t0` (CV)</span>
          <span className="text-2xl font-mono font-bold text-teal-800 tabular-nums mt-1 block">
            {REPO_MODEL_METADATA.rocAucOOF.toFixed(3)} AUC
          </span>
          <span className="text-xs text-slate-500 mt-1 block tabular-nums">
            PR-AUC {REPO_MODEL_METADATA.prAucOOF.toFixed(3)} · ECE {REPO_MODEL_METADATA.eceT0}
          </span>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5">
          <span className="text-xs text-slate-500 block">Seguimiento personalizado</span>
          <span className="text-2xl font-mono font-bold text-slate-900 tabular-nums mt-1 block">
            {specialistSignedIn ? assignedCasesCount : 'Requiere acceso'}
          </span>
          <span className="text-xs text-slate-500 mt-1 block">
            {specialistSignedIn
              ? 'Casos asignados en tu panel seguro'
              : 'Autenticación médica verificada'}
          </span>
        </div>
      </section>

      {/* Main Calibration Chart from calibracion_cajas.csv */}
      <section className="bg-white border border-slate-200 rounded-xl p-5 sm:p-6">
        <RetentionCurveChart />
      </section>

      {/* Model Comparison Benchmark from reports/comparacion_modelos.md */}
      <section className="bg-white border border-slate-200 rounded-xl p-5 sm:p-6 space-y-4">
        <div>
          <h2 className="text-base font-semibold text-slate-900">
            Selección del modelo de referencia (`reports/comparacion_modelos.md`)
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Evaluación en holdout (n=600) con IC 95% bootstrap. Se seleccionó Regresión Logística T0
            por parsimonia, calibración (ECE 0.019) e interpretabilidad directa.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse min-w-[640px]">
            <thead>
              <tr className="border-b border-slate-200 text-xs text-slate-500">
                <th className="py-2.5 pr-4 font-medium">Modelo evaluado</th>
                <th className="py-2.5 pr-4 font-medium">ROC-AUC T0</th>
                <th className="py-2.5 pr-4 font-medium">PR-AUC T0</th>
                <th className="py-2.5 pr-4 font-medium">Brier T0 (CV)</th>
                <th className="py-2.5 pr-4 font-medium">ECE T0</th>
                <th className="py-2.5 font-medium">ROC-AUC T1 (cota)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-xs sm:text-sm">
              {MODEL_COMPARISON_TABLE.map((row) => (
                <tr
                  key={row.model}
                  className={row.isSelectedReference ? 'bg-teal-50/50 font-medium' : 'hover:bg-slate-50/80'}
                >
                  <td className="py-3 pr-4 text-slate-900">
                    {row.model}
                    {row.isSelectedReference && (
                      <span className="text-xs text-teal-800 font-semibold ml-2">
                        · Seleccionado
                      </span>
                    )}
                  </td>
                  <td className="py-3 pr-4 font-mono text-xs tabular-nums text-slate-700">
                    {row.rocAucT0}
                  </td>
                  <td className="py-3 pr-4 font-mono text-xs tabular-nums text-slate-700">
                    {row.prAucT0}
                  </td>
                  <td className="py-3 pr-4 font-mono text-xs tabular-nums text-slate-700">
                    {row.brierT0}
                  </td>
                  <td className="py-3 pr-4 font-mono text-xs tabular-nums text-slate-700">
                    {row.eceT0}
                  </td>
                  <td className="py-3 font-mono text-xs tabular-nums text-slate-600">
                    {row.rocAucT1}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Real sample records from dataset_modelado_v2.csv */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-8 bg-white border border-slate-200 rounded-xl p-5 sm:p-6">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
            <div>
              <h2 className="text-base font-semibold text-slate-900">
                Muestra real de `dataset_modelado_v2.csv` evaluada con `modelo_t0`
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Registros sintéticos `PAC-2000`–`PAC-2005` con puntaje T0 y etiqueta observada
              </p>
            </div>
            <button
              type="button"
              onClick={() => onNavigate('seguimiento')}
              className="inline-flex items-center gap-1.5 text-xs font-semibold text-teal-800 hover:text-teal-900 min-h-[40px] px-2 whitespace-nowrap"
            >
              Ir a seguimiento clínico
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse min-w-[580px]">
              <thead>
                <tr className="border-b border-slate-200 text-xs text-slate-500">
                  <th className="py-2.5 pr-3 font-medium">ID sintético</th>
                  <th className="py-2.5 pr-3 font-medium">Fisura / Apoyo</th>
                  <th className="py-2.5 pr-3 font-medium text-right">Distancia</th>
                  <th className="py-2.5 pr-3 font-medium text-right">Ingresos</th>
                  <th className="py-2.5 pr-3 font-medium text-right">Predicción T0</th>
                  <th className="py-2.5 font-medium text-right">Real v2</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-sm">
                {REPO_SAMPLE_PATIENTS.map((row) => {
                  const pred = predictT0Logistic(row);
                  return (
                    <tr key={row.syntheticCode} className="hover:bg-slate-50/80">
                      <td className="py-3 pr-3 font-mono text-xs font-semibold text-slate-900 tabular-nums">
                        {row.syntheticCode}
                      </td>
                      <td className="py-3 pr-3">
                        <span className="font-medium text-slate-900 block">
                          Fisura {row.tipoFisura}
                        </span>
                        <span className="text-xs text-slate-500">
                          Apoyo psic.: {row.apoyoPsicologico ? 'Sí' : 'No'} · SES{' '}
                          {row.nivelSocioeconomicoOrd}
                        </span>
                      </td>
                      <td className="py-3 pr-3 text-right font-mono text-xs text-slate-700 tabular-nums">
                        {row.distanciaHospitalKm} km
                      </td>
                      <td className="py-3 pr-3 text-right font-mono text-xs text-slate-700 tabular-nums">
                        ${row.ingresosMensualesUsd}
                      </td>
                      <td className="py-3 pr-3 text-right">
                        <RiskLabel level={pred.prioridad} score={pred.riskScore} />
                      </td>
                      <td className="py-3 text-right text-xs font-semibold">
                        <span
                          className={
                            row.observedDesertor ? 'text-red-700' : 'text-emerald-700'
                          }
                        >
                          {row.observedDesertor ? 'Desertor (1)' : 'Continuador (0)'}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        <div className="lg:col-span-4 bg-white border border-slate-200 rounded-xl p-5 sm:p-6 flex flex-col justify-between gap-5">
          <div className="space-y-4">
            <div className="flex items-center gap-2.5 text-teal-800">
              <ShieldCheck className="w-5 h-5 shrink-0" />
              <h2 className="text-base font-semibold text-slate-900">
                Cortes de prioridad `modelo_t0`
              </h2>
            </div>
            <p className="text-sm text-slate-600 leading-relaxed">
              Calculados con cuantiles fuera de muestra (percentiles 80 y 30) en 5 pliegues sobre
              los 3,000 pacientes sintéticos de la cohorte v2:
            </p>
            <div className="pt-2 border-t border-slate-100 space-y-2 text-xs text-slate-700 font-mono tabular-nums">
              <p>· Prioridad Alta (≥ 83.2%): tasa real 89.7%</p>
              <p>· Prioridad Media (61.9%–83.1%): tasa real 77.3%</p>
              <p>· Prioridad Habitual (&lt; 61.9%): tasa real 49.4%</p>
              <p>· Umbral alerta Recall ≥ 90%: 55.5%</p>
            </div>
          </div>

          <div className="flex flex-col gap-2.5">
            <button
              type="button"
              onClick={() => onNavigate('seguimiento')}
              className="w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 min-h-[44px] bg-slate-900 hover:bg-slate-800 text-white text-sm font-medium rounded-lg transition-colors"
            >
              <ClipboardCheck className="w-4 h-4" />
              Abrir seguimiento clínico
            </button>
            <button
              type="button"
              onClick={() => onNavigate('privacidad')}
              className="w-full inline-flex items-center justify-center px-4 py-2.5 min-h-[44px] bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 text-xs font-medium rounded-lg transition-colors"
            >
              Revisar ficha de datos sintéticos y normativa
            </button>
          </div>
        </div>
      </section>
    </div>
  );
};
