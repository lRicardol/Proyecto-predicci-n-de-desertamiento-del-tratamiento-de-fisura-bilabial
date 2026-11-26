import React from 'react';
import { Database, FileCheck2, Lock, ShieldCheck } from 'lucide-react';
import { REPO_MODEL_METADATA } from '../data/syntheticCohort';

export const PrivacyGovernancePage: React.FC = () => {
  return (
    <div className="space-y-6">
      <header>
        <p className="text-xs font-medium text-teal-800">
          Transparencia metodológica y cumplimiento normativo
        </p>
        <h1 className="text-2xl font-bold text-slate-900 mt-1 text-balance">
          Datos 100% sintéticos (`dataset_modelado_v2.csv`) y privacidad clínica
        </h1>
        <p className="text-sm text-slate-600 mt-2 max-w-3xl leading-relaxed">
          Esta plataforma declara explícitamente que todos los registros analizados (
          {REPO_MODEL_METADATA.totalPatients.toLocaleString('es-CO')} pacientes sintéticos de la
          cohorte v2 del repositorio) y los casos guardados (`PAC-SYN-XXXX`) son sintéticos y no
          representan ni describen pacientes reales.
        </p>
      </header>

      <section className="bg-teal-50/70 border border-teal-200 rounded-xl p-6">
        <div className="flex items-start gap-4">
          <Database className="w-6 h-6 text-teal-800 shrink-0 mt-0.5" />
          <div className="space-y-2">
            <h2 className="text-base font-bold text-slate-900">
              Contrato de datos del repositorio (`modelo_t0.json` · Escenario T0 sin fuga)
            </h2>
            <p className="text-sm text-slate-700 leading-relaxed">
              De acuerdo con `reports/diccionario_datos.md` y `modeling_features.py`, el modelo
              seleccionado (`{REPO_MODEL_METADATA.modelName}`) utiliza únicamente las 7 variables
              disponibles al ingreso del paciente (`tipo_fisura`, `edad_inicio_meses`,
              `distancia_hospital_km`, `nivel_socioeconomico_ord`, `ingresos_mensuales_usd`,
              `apoyo_psicologico`, `dificultad_alimentacion_preop`), excluyendo columnas con fuga
              temporal (`citas_perdidas_noshow`, `citas_terapia_asistidas`,
              `ratio_asistencia_terapia`, `peso_z_score_2anos`).
            </p>
          </div>
        </div>
      </section>

      <section className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <article className="bg-white border border-slate-200 rounded-xl p-5 space-y-3">
          <div className="flex items-center gap-2 text-slate-900">
            <ShieldCheck className="w-5 h-5 text-teal-700 shrink-0" />
            <h3 className="text-sm font-bold">Ley 1581 de 2012 (Habeas Data)</h3>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Alineado con los principios de finalidad, seguridad y confidencialidad para datos
            sensibles en salud. El esquema impide almacenar nombres, documentos o teléfonos de
            pacientes reales.
          </p>
        </article>

        <article className="bg-white border border-slate-200 rounded-xl p-5 space-y-3">
          <div className="flex items-center gap-2 text-slate-900">
            <FileCheck2 className="w-5 h-5 text-teal-700 shrink-0" />
            <h3 className="text-sm font-bold">RGPD (UE 2016/679 · Art. 5 y 25)</h3>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Privacidad desde el diseño: minimización estricta a las 7 variables T0, seudonimización
            obligatoria (`PAC-SYN-XXXX`) y acceso aislado por especialista autenticado.
          </p>
        </article>

        <article className="bg-white border border-slate-200 rounded-xl p-5 space-y-3">
          <div className="flex items-center gap-2 text-slate-900">
            <Lock className="w-5 h-5 text-teal-700 shrink-0" />
            <h3 className="text-sm font-bold">Estándar HIPAA Safe Harbor</h3>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Eliminación total de identificadores directos y ubicación exacta: la distancia al
            hospital se procesa de forma numérica agregada sin direcciones domiciliarias.
          </p>
        </article>
      </section>

      <section className="bg-white border border-slate-200 rounded-xl p-5 sm:p-6 space-y-4">
        <h2 className="text-base font-semibold text-slate-900">
          Controles técnicos y límites documentados del modelo
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse min-w-[540px]">
            <thead>
              <tr className="border-b border-slate-200 text-xs text-slate-500">
                <th className="py-2.5 pr-4 font-medium">Control aplicado</th>
                <th className="py-2.5 pr-4 font-medium">Regla / Artefacto</th>
                <th className="py-2.5 font-medium">Propósito</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-xs sm:text-sm text-slate-700">
              <tr>
                <td className="py-3 pr-4 font-semibold text-slate-900">
                  Validación de registro sintético
                </td>
                <td className="py-3 pr-4 font-mono text-xs">
                  isSynthetic == true · ^PAC-SYN-[0-9]&#123;4&#125;$
                </td>
                <td className="py-3">
                  Bloquea la persistencia de identificadores reales en Firestore.
                </td>
              </tr>
              <tr>
                <td className="py-3 pr-4 font-semibold text-slate-900">
                  Aislamiento por especialista
                </td>
                <td className="py-3 pr-4 font-mono text-xs">
                  specialistId == request.auth.uid
                </td>
                <td className="py-3">
                  Cada profesional autenticado accede únicamente a sus expedientes asignados.
                </td>
              </tr>
              <tr>
                <td className="py-3 pr-4 font-semibold text-slate-900">
                  Criterio clínico no automatizado
                </td>
                <td className="py-3 pr-4 font-mono text-xs">
                  reports/comparacion_modelos.md §11
                </td>
                <td className="py-3">
                  La alerta de `modelo_t0` apoya la priorización, nunca sustituye la valoración clínica.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};
