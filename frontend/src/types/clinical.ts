export type PageId = 'resumen' | 'factores' | 'simulador' | 'seguimiento' | 'privacidad';

export type FissureType = 'Completa' | 'Incompleta' | 'Asimetrica';

export type SesLevel = 0 | 1 | 2 | 4;

export type PriorityTier = 'Alta' | 'Media' | 'Habitual';

export type FollowUpStatus = 'Activo' | 'En Intervención' | 'Retenido' | 'Alerta Crítica';

export type ActionType =
  | 'Control Presencial'
  | 'Telemedicina'
  | 'Apoyo Psicosocial'
  | 'Gestión Transporte'
  | 'Reprogramación';

export type EntryOutcome = 'Asistió' | 'Reprogramado' | 'Inasistencia' | 'Pendiente';

export interface SpecialistProfile {
  uid: string;
  displayName: string;
  email: string;
  specialty: string;
  institution: string;
  role: 'specialist';
  createdAt?: unknown;
  updatedAt?: unknown;
}

export interface PatientFollowUp {
  id: string;
  specialistId: string;
  syntheticCode: string;
  tipoFisura: FissureType;
  edadInicioMeses: number;
  distanciaHospitalKm: number;
  nivelSocioeconomicoOrd: SesLevel;
  ingresosMensualesUsd: number;
  apoyoPsicologico: boolean;
  dificultadAlimentacionPreop: boolean;
  riskScore: number;
  prioridad: PriorityTier;
  status: FollowUpStatus;
  nextAppointmentDate: string;
  interventionPlan: string;
  clinicalNotes: string;
  isSynthetic: true;
  createdAt?: unknown;
  updatedAt?: unknown;
}

export interface FollowUpEntry {
  id: string;
  followUpId: string;
  specialistId: string;
  actionType: ActionType;
  outcome: EntryOutcome;
  note: string;
  createdAt?: unknown;
}

export interface SimulationInputT0 {
  tipoFisura: FissureType;
  edadInicioMeses: number;
  distanciaHospitalKm: number;
  nivelSocioeconomicoOrd: SesLevel;
  ingresosMensualesUsd: number;
  apoyoPsicologico: boolean;
  dificultadAlimentacionPreop: boolean;
}

export interface LogOddsContribution {
  variable: string;
  label: string;
  logOddsDelta: number;
  direction: 'increase' | 'protective';
  detail: string;
}

export interface SimulationResultT0 {
  probability: number; // 0-1
  riskScore: number; // 0-100%
  prioridad: PriorityTier;
  alertaRecall90: boolean; // >= 0.5547 umbral_recall_minimo
  retentionProbability: number;
  referenceProb: number;
  contributions: LogOddsContribution[];
  recommendedPlan: string;
  suggestedActions: string[];
}

export interface RiskFactorInsight {
  id: string;
  variable: string;
  name: string;
  category: 'Psicosocial' | 'Geográfico' | 'Socioeconómico' | 'Clínico';
  aucDrop: number; // caida_auc_media from interpretabilidad_permutacion.csv
  oddsRatioSd: number; // or_por_de from interpretabilidad_coeficientes.csv
  coefStd: number;
  ic95: [number, number];
  signStability: number;
  desertorMeanLabel: string;
  continuadorMeanLabel: string;
  clinicalDescription: string;
  mitigationProtocol: string;
}

export interface ModelBenchmarkRow {
  model: string;
  rocAucT0: string;
  prAucT0: string;
  brierT0: string;
  eceT0: string;
  rocAucT1: string;
  isSelectedReference?: boolean;
}

export interface CalibrationBin {
  bin: number;
  predicted: number;
  observed: number;
}

export interface MatrixCell {
  distanceLabel: string;
  colLabel: string;
  dropoutRate: number;
  syntheticCount: number;
  prioridad: PriorityTier;
  actionSummary: string;
}

export interface RadarSegmentProfile {
  id: string;
  label: string;
  description: string;
  n: number;
  prevalencePct: number;
  rocAuc: number;
  axes: {
    axis: string;
    shortAxis: string;
    segmentValue: number;
    cohortBaseline: number;
    unit: string;
  }[];
}
