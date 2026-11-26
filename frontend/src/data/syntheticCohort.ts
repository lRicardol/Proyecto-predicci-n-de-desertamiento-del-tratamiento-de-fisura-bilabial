import {
  CalibrationBin,
  FissureType,
  LogOddsContribution,
  MatrixCell,
  ModelBenchmarkRow,
  PriorityTier,
  RadarSegmentProfile,
  RiskFactorInsight,
  SesLevel,
  SimulationInputT0,
  SimulationResultT0,
} from '../types/clinical';

export const FISSURE_TYPES: { value: FissureType; label: string }[] = [
  { value: 'Completa', label: 'Fisura Bilabial Completa' },
  { value: 'Incompleta', label: 'Fisura Bilabial Incompleta' },
  { value: 'Asimetrica', label: 'Fisura Bilabial Asimétrica' },
];

export const SES_OPTIONS: { value: SesLevel; label: string; short: string }[] = [
  { value: 0, label: 'SES 0 · Bajo (n=1,393)', short: 'SES 0 (Bajo)' },
  { value: 1, label: 'SES 1 · Medio-Bajo (n=1,016)', short: 'SES 1 (Medio-Bajo)' },
  { value: 2, label: 'SES 2 · Medio (n=440)', short: 'SES 2 (Medio)' },
  { value: 4, label: 'SES 4 · Alto (n=151)', short: 'SES 4 (Alto)' },
];

/**
 * Metadatos exactos de `src/infrastructure/ai_models/serialized/modelo_t0.json`
 * y `data/processed/dataset_modelado_v2.csv` (3,000 pacientes sintéticos).
 */
export const REPO_MODEL_METADATA = {
  version: 't0-logistica-1',
  modelName: 'Regresión Logística Regularizada (C = 0.01)',
  scenario: 'T0 (Variables al ingreso, sin fuga)',
  cohort: 'dataset_modelado_v2.csv (v2)',
  totalPatients: 3000,
  desertorCount: 2142,
  continuadorCount: 858,
  desertorRatePct: 71.4,
  rocAucOOF: 0.722,
  rocAucHoldout: 0.737,
  prAucOOF: 0.858,
  brierT0: 0.179,
  eceT0: 0.019,
  recallThreshold: 0.5547195226902468,
  priorityCutoffs: {
    alta: 0.8319376695287867,
    habitual: 0.619384094353562,
  },
  priorityObservedRates: {
    alta: 89.7,
    media: 77.3,
    habitual: 49.4,
  },
  medianDistanceKm: 40.2,
  psychSupportRatePct: 39.4,
};

/**
 * Parámetros exactos ajustados con L2 C=0.01 sobre `dataset_modelado_v2.csv` (n=3,000)
 * replicando el Pipeline de `final_model.py` + `logistic_predictor.py`.
 */
const T0_SCALER = {
  edad_inicio_meses: { mean: 6.087667, std: 3.16712 },
  distancia_hospital_km: { mean: 55.098967, std: 49.425413 },
  nivel_socioeconomico_ord: { mean: 0.833333, std: 1.01795 },
  ingresos_mensuales_usd: { mean: 990.938667, std: 951.321813 },
  apoyo_psicologico: { mean: 0.394, std: 0.488635 },
  dificultad_alimentacion_preop: { mean: 0.641, std: 0.479707 },
};

const T0_COEFFICIENTS = {
  intercept: 1.038858,
  edad_inicio_meses: -0.033444,
  distancia_hospital_km: 0.327513,
  nivel_socioeconomico_ord: -0.058824,
  ingresos_mensuales_usd: -0.253573,
  apoyo_psicologico: -0.579887,
  dificultad_alimentacion_preop: 0.012189,
  tipo_fisura_Asimetrica: 0.023347,
  tipo_fisura_Completa: -0.077026,
  tipo_fisura_Incompleta: 0.053679,
};

const T0_REF_MEAN_CAT = {
  Asimetrica: 0.211,
  Completa: 0.499667,
  Incompleta: 0.289333,
};

function sigmoid(z: number): number {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const ez = Math.exp(z);
  return ez / (1 + ez);
}

/**
 * Predicción y descomposición exacta en log-odds frente al paciente promedio
 * siguiendo `LogisticDropoutPredictor.predict_dropout_probability` y `explain_prediction`.
 */
export function predictT0Logistic(input: SimulationInputT0): SimulationResultT0 {
  const zEdad = (input.edadInicioMeses - T0_SCALER.edad_inicio_meses.mean) / T0_SCALER.edad_inicio_meses.std;
  const zDist = (input.distanciaHospitalKm - T0_SCALER.distancia_hospital_km.mean) / T0_SCALER.distancia_hospital_km.std;
  const zSes = (input.nivelSocioeconomicoOrd - T0_SCALER.nivel_socioeconomico_ord.mean) / T0_SCALER.nivel_socioeconomico_ord.std;
  const zIng = (input.ingresosMensualesUsd - T0_SCALER.ingresos_mensuales_usd.mean) / T0_SCALER.ingresos_mensuales_usd.std;
  const zApoyo = ((input.apoyoPsicologico ? 1 : 0) - T0_SCALER.apoyo_psicologico.mean) / T0_SCALER.apoyo_psicologico.std;
  const zAlim = ((input.dificultadAlimentacionPreop ? 1 : 0) - T0_SCALER.dificultad_alimentacion_preop.mean) / T0_SCALER.dificultad_alimentacion_preop.std;

  const isAsim = input.tipoFisura === 'Asimetrica' ? 1 : 0;
  const isComp = input.tipoFisura === 'Completa' ? 1 : 0;
  const isInc = input.tipoFisura === 'Incompleta' ? 1 : 0;

  const refLogOdds =
    T0_COEFFICIENTS.intercept +
    T0_REF_MEAN_CAT.Asimetrica * T0_COEFFICIENTS.tipo_fisura_Asimetrica +
    T0_REF_MEAN_CAT.Completa * T0_COEFFICIENTS.tipo_fisura_Completa +
    T0_REF_MEAN_CAT.Incompleta * T0_COEFFICIENTS.tipo_fisura_Incompleta;

  const cApoyo = zApoyo * T0_COEFFICIENTS.apoyo_psicologico;
  const cDist = zDist * T0_COEFFICIENTS.distancia_hospital_km;
  const cIng = zIng * T0_COEFFICIENTS.ingresos_mensuales_usd;
  const cSes = zSes * T0_COEFFICIENTS.nivel_socioeconomico_ord;
  const cEdad = zEdad * T0_COEFFICIENTS.edad_inicio_meses;
  const cAlim = zAlim * T0_COEFFICIENTS.dificultad_alimentacion_preop;
  const cFisura =
    (isAsim - T0_REF_MEAN_CAT.Asimetrica) * T0_COEFFICIENTS.tipo_fisura_Asimetrica +
    (isComp - T0_REF_MEAN_CAT.Completa) * T0_COEFFICIENTS.tipo_fisura_Completa +
    (isInc - T0_REF_MEAN_CAT.Incompleta) * T0_COEFFICIENTS.tipo_fisura_Incompleta;

  const totalLogOdds = refLogOdds + cApoyo + cDist + cIng + cSes + cEdad + cAlim + cFisura;
  const prob = Math.max(0, Math.min(1, sigmoid(totalLogOdds)));
  const riskScore = Number((prob * 100).toFixed(1));

  const prioridad: PriorityTier =
    prob >= REPO_MODEL_METADATA.priorityCutoffs.alta
      ? 'Alta'
      : prob < REPO_MODEL_METADATA.priorityCutoffs.habitual
      ? 'Habitual'
      : 'Media';

  const alertaRecall90 = prob >= REPO_MODEL_METADATA.recallThreshold;

  const rawContributions: LogOddsContribution[] = [
    {
      variable: 'apoyo_psicologico',
      label: `Apoyo psicológico al ingreso (${input.apoyoPsicologico ? 'Sí' : 'No'})`,
      logOddsDelta: Number(cApoyo.toFixed(4)),
      direction: cApoyo <= 0 ? 'protective' : 'increase',
      detail: input.apoyoPsicologico
        ? 'Principal factor protector en T0 (deserción en cohorte: 54.7% vs 82.3% sin apoyo).'
        : 'Ausencia de apoyo psicológico eleva el riesgo (caída AUC 0.150 en permutación).',
    },
    {
      variable: 'distancia_hospital_km',
      label: `Distancia al hospital (${input.distanciaHospitalKm.toFixed(1)} km)`,
      logOddsDelta: Number(cDist.toFixed(4)),
      direction: cDist <= 0 ? 'protective' : 'increase',
      detail:
        input.distanciaHospitalKm > 55.1
          ? 'Supera la media de la cohorte (55.1 km); OR 1.36 por desviación estándar.'
          : 'Por debajo de la media de cohorte (55.1 km).',
    },
    {
      variable: 'ingresos_mensuales_usd',
      label: `Ingresos del hogar ($${input.ingresosMensualesUsd} USD/mes)`,
      logOddsDelta: Number(cIng.toFixed(4)),
      direction: cIng <= 0 ? 'protective' : 'increase',
      detail:
        input.ingresosMensualesUsd < 991
          ? 'Por debajo de la media muestral ($991 USD); mayor vulnerabilidad financiera.'
          : 'Ingresos superiores a la media ($991 USD) amortiguan costos de traslado.',
    },
    {
      variable: 'nivel_socioeconomico_ord',
      label: `Nivel socioeconómico (SES ${input.nivelSocioeconomicoOrd})`,
      logOddsDelta: Number(cSes.toFixed(4)),
      direction: cSes <= 0 ? 'protective' : 'increase',
      detail: 'Escala ordinal v2 (0=Bajo, 1=Medio-Bajo, 2=Medio, 4=Alto).',
    },
    {
      variable: 'tipo_fisura',
      label: `Tipo de fisura (${input.tipoFisura})`,
      logOddsDelta: Number(cFisura.toFixed(4)),
      direction: cFisura <= 0 ? 'protective' : 'increase',
      detail: 'Codificación one-hot respecto al promedio de la cohorte v2.',
    },
    {
      variable: 'edad_inicio_meses',
      label: `Edad de inicio (${input.edadInicioMeses} meses)`,
      logOddsDelta: Number(cEdad.toFixed(4)),
      direction: cEdad <= 0 ? 'protective' : 'increase',
      detail: 'Rango observado en v2: 1 a 11 meses (mediana 6 meses).',
    },
    {
      variable: 'dificultad_alimentacion_preop',
      label: `Dificultad de alimentación preop (${input.dificultadAlimentacionPreop ? 'Sí' : 'No'})`,
      logOddsDelta: Number(cAlim.toFixed(4)),
      direction: cAlim <= 0 ? 'protective' : 'increase',
      detail: 'Efecto marginal cercano a nulo en T0 (caída AUC ≈ 0.000).',
    },
  ];

  const contributions = rawContributions.sort(
    (a, b) => Math.abs(b.logOddsDelta) - Math.abs(a.logOddsDelta)
  );

  const suggestedActions: string[] = [];
  if (!input.apoyoPsicologico) {
    suggestedActions.push('Asignar acompañamiento psicosocial al ingreso (reduce tasa observada de 82.3% a 54.7%).');
  }
  if (input.distanciaHospitalKm >= 75.2) {
    suggestedActions.push('Cuartil 4 de distancia (>75.2 km): coordinar subsidio de transporte o teleasistencia.');
  }
  if (input.nivelSocioeconomicoOrd <= 1 && input.ingresosMensualesUsd < 800) {
    suggestedActions.push('Evaluación por trabajo social para mitigar barreras de gasto de bolsillo.');
  }
  if (suggestedActions.length === 0) {
    suggestedActions.push('Mantener calendario de controles T0 con seguimiento habitual.');
  }

  const recommendedPlan =
    prioridad === 'Alta'
      ? `Prioridad Alta (≥83.2%): ${suggestedActions[0]}`
      : prioridad === 'Media'
      ? `Prioridad Media (61.9%–83.1%): ${suggestedActions[0]}`
      : `Prioridad Habitual (<61.9%): ${suggestedActions[0]}`;

  return {
    probability: prob,
    riskScore,
    prioridad,
    alertaRecall90,
    retentionProbability: Number((100 - riskScore).toFixed(1)),
    referenceProb: Number((sigmoid(refLogOdds) * 100).toFixed(1)),
    contributions,
    recommendedPlan,
    suggestedActions,
  };
}

/**
 * Comparación de modelos de `reports/comparacion_modelos.md` (Holdout & CV 5×3).
 */
export const MODEL_COMPARISON_TABLE: ModelBenchmarkRow[] = [
  {
    model: 'Regresión logística (C=0.01)',
    rocAucT0: '0.737 [0.693–0.783]',
    prAucT0: '0.868 [0.833–0.902]',
    brierT0: '0.179 ± 0.004',
    eceT0: '0.019',
    rocAucT1: '0.877 [0.846–0.908]',
    isSelectedReference: true,
  },
  {
    model: 'XGBoost (lr=0.05, depth=2)',
    rocAucT0: '0.734 [0.690–0.778]',
    prAucT0: '0.868 [0.833–0.901]',
    brierT0: '0.179 ± 0.005',
    eceT0: '0.017',
    rocAucT1: '0.868 [0.835–0.900]',
  },
  {
    model: 'Random Forest (depth=4)',
    rocAucT0: '0.732 [0.689–0.777]',
    prAucT0: '0.867 [0.831–0.901]',
    brierT0: '0.180 ± 0.004',
    eceT0: '0.033',
    rocAucT1: '0.864 [0.831–0.897]',
  },
  {
    model: 'Árbol de decisión (depth=3)',
    rocAucT0: '0.721 [0.678–0.763]',
    prAucT0: '0.844 [0.809–0.878]',
    brierT0: '0.183 ± 0.007',
    eceT0: '0.024',
    rocAucT1: '0.854 [0.819–0.891]',
  },
  {
    model: 'Dummy prior (línea base)',
    rocAucT0: '0.500 [0.500–0.500]',
    prAucT0: '0.713 [0.678–0.750]',
    brierT0: '0.204 ± 0.000',
    eceT0: '—',
    rocAucT1: '0.500 [0.500–0.500]',
  },
];

/**
 * Curva de calibración exacta de T0 Regresión Logística (`reports/modelos/calibracion_cajas.csv`, n=720 por caja en CV).
 */
export const T0_CALIBRATION_BINS: CalibrationBin[] = [
  { bin: 1, predicted: 44.4, observed: 46.8 },
  { bin: 2, predicted: 54.4, observed: 49.2 },
  { bin: 3, predicted: 59.1, observed: 53.5 },
  { bin: 4, predicted: 66.4, observed: 65.0 },
  { bin: 5, predicted: 74.8, observed: 74.2 },
  { bin: 6, predicted: 78.3, observed: 81.5 },
  { bin: 7, predicted: 80.3, observed: 82.5 },
  { bin: 8, predicted: 82.2, observed: 82.4 },
  { bin: 9, predicted: 84.8, observed: 87.5 },
  { bin: 10, predicted: 89.6, observed: 91.7 },
];

/**
 * Datos exactos de `reports/modelos/interp_T0/interpretabilidad_permutacion.csv` y
 * `interpretabilidad_coeficientes.csv` + estadísticas de `dataset_modelado_v2.csv`.
 */
export const RISK_FACTORS_T0_DATA: RiskFactorInsight[] = [
  {
    id: 'apoyo_psicologico',
    variable: 'apoyo_psicologico',
    name: 'Apoyo psicológico al ingreso',
    category: 'Psicosocial',
    aucDrop: 0.1496,
    oddsRatioSd: 0.576,
    coefStd: -0.5513,
    ic95: [0.536, 0.615],
    signStability: 100,
    desertorMeanLabel: '30.2% con apoyo (82.3% deserción sin apoyo)',
    continuadorMeanLabel: '62.5% con apoyo (54.7% deserción con apoyo)',
    clinicalDescription:
      'Es el predictor más determinante en el escenario T0 (estabilidad de signo 100% en 200 remuestreos bootstrap). Los pacientes sin apoyo psicológico presentan una tasa observada de deserción del 82.3% frente al 54.7% cuando reciben acompañamiento.',
    mitigationProtocol:
      'Priorizar evaluación e inclusión temprana en programa de apoyo psicosocial familiar desde la primera consulta de ingreso.',
  },
  {
    id: 'distancia_hospital_km',
    variable: 'distancia_hospital_km',
    name: 'Distancia al hospital (km)',
    category: 'Geográfico',
    aucDrop: 0.0425,
    oddsRatioSd: 1.363,
    coefStd: 0.3096,
    ic95: [1.265, 1.48],
    signStability: 100,
    desertorMeanLabel: '59.7 km en promedio (Q4 >75.2 km: 82.2%)',
    continuadorMeanLabel: '43.7 km en promedio (Q1 ≤19.1 km: 65.2%)',
    clinicalDescription:
      'Cada desviación estándar adicional (+49.4 km) incrementa 1.36 veces los momios de deserción (estabilidad 100%). El cuartil más lejano (>75.2 km) alcanza 82.2% de deserción.',
    mitigationProtocol:
      'Activar enlace con redes territoriales, teleorientación intermedia y agrupación de valoraciones clínicas en una sola jornada.',
  },
  {
    id: 'ingresos_mensuales_usd',
    variable: 'ingresos_mensuales_usd',
    name: 'Ingresos mensuales del hogar (USD)',
    category: 'Socioeconómico',
    aucDrop: 0.0264,
    oddsRatioSd: 0.793,
    coefStd: -0.232,
    ic95: [0.735, 0.852],
    signStability: 100,
    desertorMeanLabel: '$898.7 USD promedio',
    continuadorMeanLabel: '$1,221.2 USD promedio',
    clinicalDescription:
      'Mayor ingreso familiar reduce de forma consistente la probabilidad de abandono (OR 0.793 por DE, estabilidad 100%), reflejando capacidad para costear desplazamientos y ausencias laborales.',
    mitigationProtocol:
      'Vincular a hogares de menores ingresos con subsidios de traslado y redes de apoyo de trabajo social.',
  },
  {
    id: 'nivel_socioeconomico_ord',
    variable: 'nivel_socioeconomico_ord',
    name: 'Nivel socioeconómico (SES ordinal)',
    category: 'Socioeconómico',
    aucDrop: 0.0024,
    oddsRatioSd: 0.946,
    coefStd: -0.0555,
    ic95: [0.88, 1.026],
    signStability: 91.5,
    desertorMeanLabel: 'SES 0: 74.6% · SES 1: 73.1%',
    continuadorMeanLabel: 'SES 2: 65.5% · SES 4: 47.7%',
    clinicalDescription:
      'Presenta colinealidad parcial con los ingresos mensuales (`ingresos_mensuales_usd`). En análisis univariado, el grupo SES 0 duplica casi la prevalencia de deserción frente a SES 4 (74.6% vs 47.7%).',
    mitigationProtocol:
      'Monitorear sesgo de umbral por subgrupo socioeconómico antes de asignar alertas administrativas.',
  },
  {
    id: 'tipo_fisura',
    variable: 'tipo_fisura',
    name: 'Tipo de fisura bilabial',
    category: 'Clínico',
    aucDrop: 0.0023,
    oddsRatioSd: 1.047,
    coefStd: 0.0461,
    ic95: [0.973, 1.134],
    signStability: 88.5,
    desertorMeanLabel: 'Incompleta 73.6% · Asimétrica 72.5%',
    continuadorMeanLabel: 'Completa 69.6% (n=1,499)',
    clinicalDescription:
      'Contribución discreta en el modelo T0 ajustado: Completa muestra OR 0.948 frente a Incompleta OR 1.047 respecto a la referencia.',
    mitigationProtocol:
      'Reforzar educación en continuidad independientemente de la severidad anatómica inicial.',
  },
  {
    id: 'edad_inicio_meses',
    variable: 'edad_inicio_meses',
    name: 'Edad al inicio del tratamiento (meses)',
    category: 'Clínico',
    aucDrop: 0.0017,
    oddsRatioSd: 0.976,
    coefStd: -0.0239,
    ic95: [0.913, 1.048],
    signStability: 71.0,
    desertorMeanLabel: '6.05 meses promedio',
    continuadorMeanLabel: '6.17 meses promedio',
    clinicalDescription:
      'Señal débil en T0 (intervalo de confianza cruza 1.0). En el reporte de diccionario se señala revisar la definición operativa de fecha de inicio.',
    mitigationProtocol:
      'Registrar con precisión la fecha de ingreso frente a hitos quirúrgicos primarios.',
  },
  {
    id: 'dificultad_alimentacion_preop',
    variable: 'dificultad_alimentacion_preop',
    name: 'Dificultad de alimentación preoperatoria',
    category: 'Clínico',
    aucDrop: -0.0001,
    oddsRatioSd: 1.016,
    coefStd: 0.0162,
    ic95: [0.936, 1.097],
    signStability: 63.0,
    desertorMeanLabel: '64.4% presente',
    continuadorMeanLabel: '63.4% presente',
    clinicalDescription:
      'Prácticamente sin capacidad discriminatoria incremental en T0 (caída AUC ≈ 0.000, estabilidad 63%).',
    mitigationProtocol:
      'Mantener soporte nutricional clínico estándar sin usarlo como criterio único de alerta de abandono.',
  },
];

/**
 * Subgrupos reales de `reports/modelos/subgrupos.csv` y `dataset_modelado_v2.csv`.
 */
export const RADAR_SEGMENTS_V2: RadarSegmentProfile[] = [
  {
    id: 'q4_distancia',
    label: 'Cuartil 4 Distancia (> 75.2 km)',
    description: 'Subgrupo más distante en v2 (n=600 en holdout/CV): prevalencia de deserción 82.2%, ROC-AUC T0 0.732.',
    n: 749,
    prevalencePct: 82.2,
    rocAuc: 0.732,
    axes: [
      { axis: 'Tasa de deserción observada', shortAxis: 'Deserción', segmentValue: 82.2, cohortBaseline: 71.4, unit: '%' },
      { axis: 'Falta de apoyo psicológico', shortAxis: 'Sin apoyo', segmentValue: 63.7, cohortBaseline: 60.6, unit: '%' },
      { axis: 'Carga de distancia relativa', shortAxis: 'Distancia', segmentValue: 92.0, cohortBaseline: 46.0, unit: '%' },
      { axis: 'Vulnerabilidad SES 0–1', shortAxis: 'SES bajo', segmentValue: 79.6, cohortBaseline: 80.3, unit: '%' },
      { axis: 'Dificultad alimentación preop', shortAxis: 'Alimentación', segmentValue: 64.5, cohortBaseline: 64.1, unit: '%' },
      { axis: 'Sensibilidad modelo T0', shortAxis: 'Recall T0', segmentValue: 98.2, cohortBaseline: 89.0, unit: '%' },
    ],
  },
  {
    id: 'ses_0',
    label: 'Nivel Socioeconómico SES 0 (Bajo)',
    description: 'Grupo mayoritario de v2 (n=1,393): prevalencia de deserción 74.6%, ROC-AUC T0 0.713, Recall 93.6%.',
    n: 1393,
    prevalencePct: 74.6,
    rocAuc: 0.713,
    axes: [
      { axis: 'Tasa de deserción observada', shortAxis: 'Deserción', segmentValue: 74.6, cohortBaseline: 71.4, unit: '%' },
      { axis: 'Falta de apoyo psicológico', shortAxis: 'Sin apoyo', segmentValue: 61.8, cohortBaseline: 60.6, unit: '%' },
      { axis: 'Carga de distancia relativa', shortAxis: 'Distancia', segmentValue: 48.0, cohortBaseline: 46.0, unit: '%' },
      { axis: 'Vulnerabilidad SES 0–1', shortAxis: 'SES bajo', segmentValue: 100.0, cohortBaseline: 80.3, unit: '%' },
      { axis: 'Dificultad alimentación preop', shortAxis: 'Alimentación', segmentValue: 65.1, cohortBaseline: 64.1, unit: '%' },
      { axis: 'Sensibilidad modelo T0', shortAxis: 'Recall T0', segmentValue: 93.6, cohortBaseline: 89.0, unit: '%' },
    ],
  },
  {
    id: 'ses_4',
    label: 'Nivel Socioeconómico SES 4 (Alto)',
    description: 'Grupo de menor deserción (n=151): prevalencia 47.7% (50.0% en muestra de subgrupos), Especificidad 70.6%.',
    n: 151,
    prevalencePct: 47.7,
    rocAuc: 0.698,
    axes: [
      { axis: 'Tasa de deserción observada', shortAxis: 'Deserción', segmentValue: 47.7, cohortBaseline: 71.4, unit: '%' },
      { axis: 'Falta de apoyo psicológico', shortAxis: 'Sin apoyo', segmentValue: 49.0, cohortBaseline: 60.6, unit: '%' },
      { axis: 'Carga de distancia relativa', shortAxis: 'Distancia', segmentValue: 41.0, cohortBaseline: 46.0, unit: '%' },
      { axis: 'Vulnerabilidad SES 0–1', shortAxis: 'SES bajo', segmentValue: 0.0, cohortBaseline: 80.3, unit: '%' },
      { axis: 'Dificultad alimentación preop', shortAxis: 'Alimentación', segmentValue: 60.9, cohortBaseline: 64.1, unit: '%' },
      { axis: 'Sensibilidad modelo T0', shortAxis: 'Recall T0', segmentValue: 57.2, cohortBaseline: 89.0, unit: '%' },
    ],
  },
];

/**
 * Matriz exacta extraída de `dataset_modelado_v2.csv`: Cuartiles de distancia × SES.
 */
export const MATRIX_DISTANCE_SES: MatrixCell[] = [
  // Q1 <= 19.1 km
  { distanceLabel: 'Q1 (≤ 19.1 km)', colLabel: 'SES 0 (Bajo)', dropoutRate: 68.9, syntheticCount: 344, prioridad: 'Media', actionSummary: 'Seguimiento T0 regular con verificación de apoyo psicológico.' },
  { distanceLabel: 'Q1 (≤ 19.1 km)', colLabel: 'SES 1 (Medio-Bajo)', dropoutRate: 68.7, syntheticCount: 268, prioridad: 'Media', actionSummary: 'Recordatorio asistido previo al control quirúrgico.' },
  { distanceLabel: 'Q1 (≤ 19.1 km)', colLabel: 'SES 2 (Medio)', dropoutRate: 58.4, syntheticCount: 101, prioridad: 'Habitual', actionSummary: 'Prioridad habitual según corte < 61.9%.' },
  { distanceLabel: 'Q1 (≤ 19.1 km)', colLabel: 'SES 4 (Alto)', dropoutRate: 46.2, syntheticCount: 39, prioridad: 'Habitual', actionSummary: 'Menor tasa observada en cercanía urbana.' },
  // Q2 19.1–40.2 km
  { distanceLabel: 'Q2 (19.1–40.2 km)', colLabel: 'SES 0 (Bajo)', dropoutRate: 72.2, syntheticCount: 324, prioridad: 'Media', actionSummary: 'Confirmar acceso a apoyo psicológico en sede.' },
  { distanceLabel: 'Q2 (19.1–40.2 km)', colLabel: 'SES 1 (Medio-Bajo)', dropoutRate: 69.4, syntheticCount: 278, prioridad: 'Media', actionSummary: 'Agrupar consultas multidisciplinarias en un solo día.' },
  { distanceLabel: 'Q2 (19.1–40.2 km)', colLabel: 'SES 2 (Medio)', dropoutRate: 60.7, syntheticCount: 117, prioridad: 'Habitual', actionSummary: 'Seguimiento estándar de continuidad.' },
  { distanceLabel: 'Q2 (19.1–40.2 km)', colLabel: 'SES 4 (Alto)', dropoutRate: 34.4, syntheticCount: 32, prioridad: 'Habitual', actionSummary: 'Mínima deserción observada (34.4%).' },
  // Q3 40.2–75.2 km
  { distanceLabel: 'Q3 (40.2–75.2 km)', colLabel: 'SES 0 (Bajo)', dropoutRate: 72.5, syntheticCount: 364, prioridad: 'Media', actionSummary: 'Evaluación preventiva de costos de traslado.' },
  { distanceLabel: 'Q3 (40.2–75.2 km)', colLabel: 'SES 1 (Medio-Bajo)', dropoutRate: 71.9, syntheticCount: 235, prioridad: 'Media', actionSummary: 'Ofrecer apoyo psicosocial y teleorientación.' },
  { distanceLabel: 'Q3 (40.2–75.2 km)', colLabel: 'SES 2 (Medio)', dropoutRate: 63.9, syntheticCount: 108, prioridad: 'Media', actionSummary: 'Seguimiento intermedio con confirmación de agenda.' },
  { distanceLabel: 'Q3 (40.2–75.2 km)', colLabel: 'SES 4 (Alto)', dropoutRate: 41.5, syntheticCount: 41, prioridad: 'Habitual', actionSummary: 'Control habitual programado.' },
  // Q4 > 75.2 km
  { distanceLabel: 'Q4 (> 75.2 km)', colLabel: 'SES 0 (Bajo)', dropoutRate: 84.2, syntheticCount: 361, prioridad: 'Alta', actionSummary: 'Prioridad alta: subsidio de transporte y gestor psicosocial.' },
  { distanceLabel: 'Q4 (> 75.2 km)', colLabel: 'SES 1 (Medio-Bajo)', dropoutRate: 83.8, syntheticCount: 235, prioridad: 'Alta', actionSummary: 'Prioridad alta: articulación territorial y telemedicina.' },
  { distanceLabel: 'Q4 (> 75.2 km)', colLabel: 'SES 2 (Medio)', dropoutRate: 78.1, syntheticCount: 114, prioridad: 'Media', actionSummary: 'Planificación anticipada de traslados intermunicipales.' },
  { distanceLabel: 'Q4 (> 75.2 km)', colLabel: 'SES 4 (Alto)', dropoutRate: 66.7, syntheticCount: 39, prioridad: 'Media', actionSummary: 'Coordinar bloques quirúrgicos para reducir viajes.' },
];

/**
 * Matriz exacta de `dataset_modelado_v2.csv`: Cuartiles de distancia × Apoyo psicológico.
 */
export const MATRIX_DISTANCE_SUPPORT: MatrixCell[] = [
  { distanceLabel: 'Q1 (≤ 19.1 km)', colLabel: 'Sin apoyo (0)', dropoutRate: 79.8, syntheticCount: 441, prioridad: 'Media', actionSummary: 'Inclusión prioritaria en programa psicosocial al ingreso.' },
  { distanceLabel: 'Q1 (≤ 19.1 km)', colLabel: 'Con apoyo (1)', dropoutRate: 46.9, syntheticCount: 311, prioridad: 'Habitual', actionSummary: 'El apoyo psicológico reduce la deserción en 32.9 puntos porcentuales.' },
  { distanceLabel: 'Q2 (19.1–40.2 km)', colLabel: 'Sin apoyo (0)', dropoutRate: 78.4, syntheticCount: 463, prioridad: 'Media', actionSummary: 'Asignar acompañamiento psicosocial antes del primer control.' },
  { distanceLabel: 'Q2 (19.1–40.2 km)', colLabel: 'Con apoyo (1)', dropoutRate: 50.7, syntheticCount: 288, prioridad: 'Habitual', actionSummary: 'Adherencia favorecida por soporte emocional continuo.' },
  { distanceLabel: 'Q3 (40.2–75.2 km)', colLabel: 'Sin apoyo (0)', dropoutRate: 79.6, syntheticCount: 437, prioridad: 'Media', actionSummary: 'Doble intervención: apoyo psicológico + gestión logística.' },
  { distanceLabel: 'Q3 (40.2–75.2 km)', colLabel: 'Con apoyo (1)', dropoutRate: 55.0, syntheticCount: 311, prioridad: 'Habitual', actionSummary: 'Mantener seguimiento híbrido presencial/teleorientación.' },
  { distanceLabel: 'Q4 (> 75.2 km)', colLabel: 'Sin apoyo (0)', dropoutRate: 90.8, syntheticCount: 477, prioridad: 'Alta', actionSummary: 'Grupo de máximo riesgo (90.8%): intervención inmediata obligatoria.' },
  { distanceLabel: 'Q4 (> 75.2 km)', colLabel: 'Con apoyo (1)', dropoutRate: 67.3, syntheticCount: 272, prioridad: 'Media', actionSummary: 'El apoyo psicológico reduce la deserción en 23.5 puntos incluso >75 km.' },
];

/**
 * Registros reales de muestra de `data/processed/dataset_modelado_v2.csv` (PAC-2000..PAC-2005).
 */
export const REPO_SAMPLE_PATIENTS: (SimulationInputT0 & {
  syntheticCode: string;
  observedDesertor: boolean;
})[] = [
  {
    syntheticCode: 'PAC-2001',
    tipoFisura: 'Asimetrica',
    edadInicioMeses: 6,
    distanciaHospitalKm: 155.5,
    nivelSocioeconomicoOrd: 1,
    ingresosMensualesUsd: 612,
    apoyoPsicologico: true,
    dificultadAlimentacionPreop: true,
    observedDesertor: true,
  },
  {
    syntheticCode: 'PAC-2002',
    tipoFisura: 'Asimetrica',
    edadInicioMeses: 4,
    distanciaHospitalKm: 70.8,
    nivelSocioeconomicoOrd: 0,
    ingresosMensualesUsd: 297,
    apoyoPsicologico: false,
    dificultadAlimentacionPreop: true,
    observedDesertor: true,
  },
  {
    syntheticCode: 'PAC-2004',
    tipoFisura: 'Incompleta',
    edadInicioMeses: 5,
    distanciaHospitalKm: 13.5,
    nivelSocioeconomicoOrd: 1,
    ingresosMensualesUsd: 890,
    apoyoPsicologico: false,
    dificultadAlimentacionPreop: true,
    observedDesertor: true,
  },
  {
    syntheticCode: 'PAC-2000',
    tipoFisura: 'Completa',
    edadInicioMeses: 10,
    distanciaHospitalKm: 28.5,
    nivelSocioeconomicoOrd: 1,
    ingresosMensualesUsd: 1112,
    apoyoPsicologico: true,
    dificultadAlimentacionPreop: true,
    observedDesertor: false,
  },
  {
    syntheticCode: 'PAC-2005',
    tipoFisura: 'Completa',
    edadInicioMeses: 11,
    distanciaHospitalKm: 13.5,
    nivelSocioeconomicoOrd: 2,
    ingresosMensualesUsd: 1531,
    apoyoPsicologico: true,
    dificultadAlimentacionPreop: false,
    observedDesertor: false,
  },
];

export function buildInitialSyntheticFollowUps(specialistId: string) {
  const seedConfigs: {
    id: string;
    syntheticCode: string;
    input: SimulationInputT0;
    status: 'Alerta Crítica' | 'En Intervención' | 'Activo';
    nextAppointmentDate: string;
    notes: string;
  }[] = [
    {
      id: 'syn_pac_2002',
      syntheticCode: 'PAC-SYN-2002',
      input: {
        tipoFisura: 'Asimetrica',
        edadInicioMeses: 4,
        distanciaHospitalKm: 70.8,
        nivelSocioeconomicoOrd: 0,
        ingresosMensualesUsd: 297,
        apoyoPsicologico: false,
        dificultadAlimentacionPreop: true,
      },
      status: 'Alerta Crítica',
      nextAppointmentDate: '2026-12-04',
      notes:
        'Caso sintético basado en PAC-2002 de dataset_modelado_v2.csv. Sin apoyo psicológico al ingreso, SES 0 e ingresos bajos ($297 USD).',
    },
    {
      id: 'syn_pac_2001',
      syntheticCode: 'PAC-SYN-2001',
      input: {
        tipoFisura: 'Asimetrica',
        edadInicioMeses: 6,
        distanciaHospitalKm: 155.5,
        nivelSocioeconomicoOrd: 1,
        ingresosMensualesUsd: 612,
        apoyoPsicologico: true,
        dificultadAlimentacionPreop: true,
      },
      status: 'En Intervención',
      nextAppointmentDate: '2026-12-10',
      notes:
        'Caso sintético basado en PAC-2001 de dataset_modelado_v2.csv. Alta distancia geográfica (155.5 km) mitigada parcialmente por apoyo psicológico.',
    },
    {
      id: 'syn_pac_2005',
      syntheticCode: 'PAC-SYN-2005',
      input: {
        tipoFisura: 'Completa',
        edadInicioMeses: 11,
        distanciaHospitalKm: 13.5,
        nivelSocioeconomicoOrd: 2,
        ingresosMensualesUsd: 1531,
        apoyoPsicologico: true,
        dificultadAlimentacionPreop: false,
      },
      status: 'Activo',
      nextAppointmentDate: '2026-12-15',
      notes:
        'Caso sintético basado en PAC-2005 de dataset_modelado_v2.csv. Cercanía urbana (13.5 km), apoyo psicológico activo y prioridad habitual.',
    },
  ];

  return seedConfigs.map((cfg) => {
    const pred = predictT0Logistic(cfg.input);
    return {
      id: cfg.id,
      specialistId,
      syntheticCode: cfg.syntheticCode,
      tipoFisura: cfg.input.tipoFisura,
      edadInicioMeses: cfg.input.edadInicioMeses,
      distanciaHospitalKm: cfg.input.distanciaHospitalKm,
      nivelSocioeconomicoOrd: cfg.input.nivelSocioeconomicoOrd,
      ingresosMensualesUsd: cfg.input.ingresosMensualesUsd,
      apoyoPsicologico: cfg.input.apoyoPsicologico,
      dificultadAlimentacionPreop: cfg.input.dificultadAlimentacionPreop,
      riskScore: pred.riskScore,
      prioridad: pred.prioridad,
      status: cfg.status,
      nextAppointmentDate: cfg.nextAppointmentDate,
      interventionPlan: pred.recommendedPlan,
      clinicalNotes: cfg.notes,
      isSynthetic: true as const,
    };
  });
}
