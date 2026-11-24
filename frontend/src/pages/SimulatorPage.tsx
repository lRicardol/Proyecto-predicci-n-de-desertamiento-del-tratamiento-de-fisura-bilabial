import React, { useMemo, useState } from 'react';
import { AlertCircle, CheckCircle2, ClipboardPlus, RotateCcw } from 'lucide-react';
import {
  FISSURE_TYPES,
  REPO_MODEL_METADATA,
  REPO_SAMPLE_PATIENTS,
  SES_OPTIONS,
  predictT0Logistic,
} from '../data/syntheticCohort';
import {
  FissureType,
  PageId,
  SesLevel,
  SimulationInputT0,
} from '../types/clinical';
import { RiskLabel } from '../components/RiskStatusText';
import { useAuth } from '../context/AuthContext';

interface SimulatorPageProps {
  onNavigate: (page: PageId) => void;
}

const DEFAULT_SIMULATION: SimulationInputT0 = {
  tipoFisura: 'Asimetrica',
  edadInicioMeses: 4,
  distanciaHospitalKm: 70.8,
  nivelSocioeconomicoOrd: 0,
  ingresosMensualesUsd: 297,
  apoyoPsicologico: false,
  dificultadAlimentacionPreop: true,
};

export const SimulatorPage: React.FC<SimulatorPageProps> = ({ onNavigate }) => {
  const { user, createFollowUp } = useAuth();
  const [input, setInput] = useState<SimulationInputT0>(DEFAULT_SIMULATION);
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  const result = useMemo(() => predictT0Logistic(input), [input]);

  const handleSaveToFollowUp = async () => {
    setSaveMessage(null);
    setSaveError(null);

    if (!user) {
      onNavigate('seguimiento');
      return;
    }

    setSaving(true);
    try {
      const randomCode = `PAC-SYN-${Math.floor(2000 + Math.random() * 7999)}`;
      const status =
        result.prioridad === 'Alta'
          ? 'Alerta Crítica'
          : result.prioridad === 'Media'
          ? 'En Intervención'
          : 'Activo';

      await createFollowUp({
        syntheticCode: randomCode,
        tipoFisura: input.tipoFisura,
        edadInicioMeses: input.edadInicioMeses,
        distanciaHospitalKm: input.distanciaHospitalKm,
        nivelSocioeconomicoOrd: input.nivelSocioeconomicoOrd,
        ingresosMensualesUsd: input.ingresosMensualesUsd,
        apoyoPsicologico: input.apoyoPsicologico,
        dificultadAlimentacionPreop: input.dificultadAlimentacionPreop,
        riskScore: result.riskScore,
        prioridad: result.prioridad,
        status,
        nextAppointmentDate: '2026-12-18',
        interventionPlan: result.recommendedPlan,
        clinicalNotes: `Evaluado con modelo_t0 (${REPO_MODEL_METADATA.version}). Principales factores log-odds: ${result.contributions
          .slice(0, 2)
          .map((c) => `${c.variable} (${c.logOddsDelta > 0 ? '+' : ''}${c.logOddsDelta})`)
          .join(', ')}.`,
      });
      setSaveMessage(
        `Expediente sintético ${randomCode} guardado en tu módulo de seguimiento clínico.`
      );
    } catch {
      setSaveError('No se pudo guardar el caso simulado. Verifica tu sesión e intenta de nuevo.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6">
      <header className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div>
          <p className="text-xs font-medium text-teal-800">
            Inferencia exacta `modelo_t0` ({REPO_MODEL_METADATA.version}) · 7 variables T0 al ingreso
          </p>
          <h1 className="text-2xl font-bold text-slate-900 mt-1 text-balance">
            Simulador de Regresión Logística Regularizada (Escenario T0)
          </h1>
          <p className="text-sm text-slate-600 mt-1 max-w-2xl">
            Calcula la probabilidad calibrada de deserción y descompone los log-odds frente al
            paciente promedio de `dataset_modelado_v2.csv`, igual que `logistic_predictor.py`.
          </p>
        </div>

        <button
          type="button"
          onClick={() => {
            setInput(DEFAULT_SIMULATION);
            setSaveMessage(null);
            setSaveError(null);
          }}
          className="inline-flex items-center gap-2 px-4 py-2.5 min-h-[44px] bg-white hover:bg-slate-50 text-slate-700 border border-slate-300 text-xs font-medium rounded-lg transition-colors whitespace-nowrap shrink-0"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          Restablecer (PAC-2002)
        </button>
      </header>

      {/* Quick Load Real Repo Sample Patients */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium text-slate-600 mr-1">
          Cargar paciente de `dataset_modelado_v2.csv`:
        </span>
        {REPO_SAMPLE_PATIENTS.map((sample) => (
          <button
            key={sample.syntheticCode}
            type="button"
            onClick={() => {
              setInput({
                tipoFisura: sample.tipoFisura,
                edadInicioMeses: sample.edadInicioMeses,
                distanciaHospitalKm: sample.distanciaHospitalKm,
                nivelSocioeconomicoOrd: sample.nivelSocioeconomicoOrd,
                ingresosMensualesUsd: sample.ingresosMensualesUsd,
                apoyoPsicologico: sample.apoyoPsicologico,
                dificultadAlimentacionPreop: sample.dificultadAlimentacionPreop,
              });
              setSaveMessage(null);
            }}
            className="px-3 py-1.5 min-h-[38px] bg-slate-100 hover:bg-teal-50 text-slate-800 hover:text-teal-900 border border-slate-200 rounded-lg text-xs font-mono font-medium transition-colors"
          >
            {sample.syntheticCode} ({sample.observedDesertor ? 'Desertor' : 'Continuador'})
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* T0 Input Controls */}
        <section className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-5 sm:p-6 space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label
                htmlFor="sim-fissure"
                className="block text-xs font-semibold text-slate-700 mb-2"
              >
                Tipo de fisura (`tipo_fisura`)
              </label>
              <select
                id="sim-fissure"
                value={input.tipoFisura}
                onChange={(e) =>
                  setInput({ ...input, tipoFisura: e.target.value as FissureType })
                }
                className="w-full min-h-[44px] px-3 py-2 bg-white border border-slate-300 rounded-lg text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-teal-600"
              >
                {FISSURE_TYPES.map((type) => (
                  <option key={type.value} value={type.value}>
                    {type.label}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label
                htmlFor="sim-ses"
                className="block text-xs font-semibold text-slate-700 mb-2"
              >
                Nivel socioeconómico (`nivel_socioeconomico_ord`)
              </label>
              <select
                id="sim-ses"
                value={input.nivelSocioeconomicoOrd}
                onChange={(e) =>
                  setInput({
                    ...input,
                    nivelSocioeconomicoOrd: Number(e.target.value) as SesLevel,
                  })
                }
                className="w-full min-h-[44px] px-3 py-2 bg-white border border-slate-300 rounded-lg text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-teal-600"
              >
                {SES_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="space-y-5 pt-2 border-t border-slate-100">
            <div>
              <div className="flex items-center justify-between mb-2">
                <label htmlFor="sim-distance" className="text-sm font-medium text-slate-800">
                  Distancia al hospital (`distancia_hospital_km`)
                </label>
                <span className="text-sm font-mono font-semibold text-slate-900 tabular-nums">
                  {input.distanciaHospitalKm.toFixed(1)} km
                </span>
              </div>
              <input
                id="sim-distance"
                type="range"
                min={5}
                max={300}
                step={0.5}
                value={input.distanciaHospitalKm}
                onChange={(e) =>
                  setInput({ ...input, distanciaHospitalKm: Number(e.target.value) })
                }
                className="clinical-slider w-full"
              />
              <div className="flex justify-between text-[11px] text-slate-400 mt-1 font-mono">
                <span>5.0 km (mín v2)</span>
                <span>40.2 km (mediana)</span>
                <span>300.0 km (máx v2)</span>
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <label htmlFor="sim-income" className="text-sm font-medium text-slate-800">
                  Ingresos mensuales del hogar (`ingresos_mensuales_usd`)
                </label>
                <span className="text-sm font-mono font-semibold text-slate-900 tabular-nums">
                  ${input.ingresosMensualesUsd.toLocaleString('es-CO')} USD
                </span>
              </div>
              <input
                id="sim-income"
                type="range"
                min={250}
                max={5977}
                step={25}
                value={input.ingresosMensualesUsd}
                onChange={(e) =>
                  setInput({ ...input, ingresosMensualesUsd: Number(e.target.value) })
                }
                className="clinical-slider w-full"
              />
              <div className="flex justify-between text-[11px] text-slate-400 mt-1 font-mono">
                <span>$250 USD</span>
                <span>$661 USD (mediana v2)</span>
                <span>$5,977 USD</span>
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <label htmlFor="sim-age" className="text-sm font-medium text-slate-800">
                  Edad al inicio del tratamiento (`edad_inicio_meses`)
                </label>
                <span className="text-sm font-mono font-semibold text-slate-900 tabular-nums">
                  {input.edadInicioMeses} meses
                </span>
              </div>
              <input
                id="sim-age"
                type="range"
                min={1}
                max={11}
                step={1}
                value={input.edadInicioMeses}
                onChange={(e) =>
                  setInput({ ...input, edadInicioMeses: Number(e.target.value) })
                }
                className="clinical-slider w-full"
              />
              <div className="flex justify-between text-[11px] text-slate-400 mt-1 font-mono">
                <span>1 mes</span>
                <span>6 meses (mediana v2)</span>
                <span>11 meses</span>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
              <button
                type="button"
                onClick={() =>
                  setInput({ ...input, apoyoPsicologico: !input.apoyoPsicologico })
                }
                className={`w-full min-h-[48px] px-4 py-2.5 rounded-lg border text-sm font-medium text-left flex items-center justify-between transition-colors ${
                  input.apoyoPsicologico
                    ? 'bg-emerald-50 border-emerald-300 text-emerald-900'
                    : 'bg-red-50 border-red-200 text-red-900'
                }`}
              >
                <span>Apoyo psicológico (`apoyo_psicologico`)</span>
                <span className="font-mono text-xs font-bold">
                  {input.apoyoPsicologico ? '1 (Sí)' : '0 (No)'}
                </span>
              </button>

              <button
                type="button"
                onClick={() =>
                  setInput({
                    ...input,
                    dificultadAlimentacionPreop: !input.dificultadAlimentacionPreop,
                  })
                }
                className={`w-full min-h-[48px] px-4 py-2.5 rounded-lg border text-sm font-medium text-left flex items-center justify-between transition-colors ${
                  input.dificultadAlimentacionPreop
                    ? 'bg-amber-50 border-amber-300 text-amber-900'
                    : 'bg-slate-50 border-slate-200 text-slate-700'
                }`}
              >
                <span>Dificultad alimentación preop</span>
                <span className="font-mono text-xs font-bold">
                  {input.dificultadAlimentacionPreop ? '1 (Sí)' : '0 (No)'}
                </span>
              </button>
            </div>
          </div>
        </section>

        {/* Prediction Output Column */}
        <section className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-5 sm:p-6 space-y-5">
          <div className="flex items-start justify-between gap-4 pb-4 border-b border-slate-100">
            <div>
              <span className="text-xs text-slate-500 block">
                Probabilidad T0 (`predict_proba`)
              </span>
              <div className="flex items-baseline gap-3 mt-1">
                <span className="text-4xl font-mono font-bold text-slate-900 tabular-nums">
                  {result.riskScore}%
                </span>
                <RiskLabel level={result.prioridad} />
              </div>
            </div>
            <div className="text-right">
              <span className="text-xs text-slate-500 block">Base cohorte promedio</span>
              <span className="text-lg font-mono font-semibold text-slate-700 tabular-nums">
                {result.referenceProb}%
              </span>
            </div>
          </div>

          {/* Threshold bar with exact repo cutoffs */}
          <div>
            <div className="w-full h-3 bg-slate-100 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-200 ${
                  result.prioridad === 'Alta'
                    ? 'bg-red-600'
                    : result.prioridad === 'Media'
                    ? 'bg-amber-500'
                    : 'bg-emerald-600'
                }`}
                style={{ width: `${result.riskScore}%` }}
              />
            </div>
            <div className="flex justify-between text-[11px] text-slate-400 mt-1.5 font-mono">
              <span>0% Habitual</span>
              <span>61.9% Media</span>
              <span>83.2% Alta</span>
            </div>
          </div>

          {/* Recall 90% operational threshold alert */}
          <div
            className={`p-3 rounded-lg border text-xs flex items-center gap-2 ${
              result.alertaRecall90
                ? 'bg-amber-50 border-amber-200 text-amber-900'
                : 'bg-slate-50 border-slate-200 text-slate-700'
            }`}
          >
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>
              Umbral operativo Recall ≥ 90% (`0.5547`):{' '}
              <strong>{result.alertaRecall90 ? 'Alerta activada' : 'Por debajo del corte'}</strong>
            </span>
          </div>

          {/* Exact Log-Odds Contributions */}
          <div className="space-y-2">
            <h2 className="text-xs font-semibold text-slate-900">
              Explicación en log-odds vs. paciente promedio (`explain_prediction`)
            </h2>
            <div className="space-y-2">
              {result.contributions.map((c) => (
                <div
                  key={c.variable}
                  className="p-2.5 bg-slate-50 border border-slate-200/80 rounded-lg flex items-center justify-between gap-3"
                >
                  <div className="min-w-0">
                    <span className="text-xs font-medium text-slate-900 block truncate">
                      {c.label}
                    </span>
                    <span className="text-[11px] text-slate-500 block truncate">{c.detail}</span>
                  </div>
                  <span
                    className={`text-xs font-mono font-semibold tabular-nums shrink-0 ${
                      c.direction === 'protective' ? 'text-emerald-700' : 'text-red-700'
                    }`}
                  >
                    {c.logOddsDelta > 0 ? `+${c.logOddsDelta.toFixed(3)}` : c.logOddsDelta.toFixed(3)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg space-y-1.5">
            <h3 className="text-xs font-semibold text-slate-900">
              Conducta sugerida en consulta
            </h3>
            <ul className="space-y-1 text-xs text-slate-700">
              {result.suggestedActions.map((act) => (
                <li key={act}>· {act}</li>
              ))}
            </ul>
          </div>

          {saveMessage && (
            <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-lg flex items-start gap-2 text-xs text-emerald-900">
              <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5 text-emerald-700" />
              <span>{saveMessage}</span>
            </div>
          )}

          {saveError && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-xs text-red-800">
              {saveError}
            </div>
          )}

          <button
            type="button"
            disabled={saving}
            onClick={handleSaveToFollowUp}
            className="w-full inline-flex items-center justify-center gap-2 px-4 py-3 min-h-[44px] bg-teal-700 hover:bg-teal-800 disabled:opacity-60 text-white text-sm font-medium rounded-lg transition-colors"
          >
            <ClipboardPlus className="w-4 h-4" />
            {user
              ? saving
                ? 'Guardando expediente sintético...'
                : 'Registrar como caso sintético en Seguimiento'
              : 'Iniciar sesión médica para guardar en Seguimiento'}
          </button>
        </section>
      </div>
    </div>
  );
};
