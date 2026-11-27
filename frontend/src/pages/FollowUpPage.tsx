import React, { useEffect, useMemo, useState } from 'react';
import {
  Calendar,
  CheckCircle2,
  Lock,
  LogOut,
  Plus,
  Search,
  Send,
  ShieldCheck,
  Trash2,
  UserCheck,
  X,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import {
  FISSURE_TYPES,
  SES_OPTIONS,
  predictT0Logistic,
} from '../data/syntheticCohort';
import {
  ActionType,
  EntryOutcome,
  FissureType,
  FollowUpEntry,
  FollowUpStatus,
  PatientFollowUp,
  PriorityTier,
  SesLevel,
} from '../types/clinical';
import { RiskLabel, StatusLabel } from '../components/RiskStatusText';
import heroImage from '../assets/images/clinical_care_hero_1795476472432.jpg';
import specialistAvatar from '../assets/images/specialist_avatar_1795476477934.jpg';

const ACTION_TYPES: ActionType[] = [
  'Control Presencial',
  'Telemedicina',
  'Apoyo Psicosocial',
  'Gestión Transporte',
  'Reprogramación',
];

const ENTRY_OUTCOMES: EntryOutcome[] = [
  'Asistió',
  'Reprogramado',
  'Inasistencia',
  'Pendiente',
];

const STATUS_OPTIONS: FollowUpStatus[] = [
  'Activo',
  'En Intervención',
  'Alerta Crítica',
  'Retenido',
];

export const FollowUpPage: React.FC = () => {
  const {
    user,
    profile,
    followUps,
    authReady,
    dataLoading,
    authError,
    signInWithGoogle,
    signOutSpecialist,
    updateSpecialistProfile,
    createFollowUp,
    updateFollowUpClinical,
    deleteFollowUp,
    addFollowUpEntry,
    subscribeFollowUpEntries,
    clearAuthError,
  } = useAuth();

  const [searchQuery, setSearchQuery] = useState('');
  const [priorityFilter, setPriorityFilter] = useState<'Todos' | PriorityTier>('Todos');
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);
  const [entries, setEntries] = useState<FollowUpEntry[]>([]);

  // Specialist profile edit state
  const [editingProfile, setEditingProfile] = useState(false);
  const [profName, setProfName] = useState('');
  const [profSpecialty, setProfSpecialty] = useState('');
  const [profInstitution, setProfInstitution] = useState('');
  const [profileMsg, setProfileMsg] = useState<string | null>(null);

  // New case modal state (exact 7 T0 variables)
  const [showNewModal, setShowNewModal] = useState(false);
  const [newCode, setNewCode] = useState(`PAC-SYN-${Math.floor(2000 + Math.random() * 2999)}`);
  const [newFisura, setNewFisura] = useState<FissureType>('Completa');
  const [newEdad, setNewEdad] = useState(6);
  const [newDist, setNewDist] = useState(62);
  const [newSes, setNewSes] = useState<SesLevel>(0);
  const [newIngresos, setNewIngresos] = useState(480);
  const [newApoyo, setNewApoyo] = useState(false);
  const [newAlim, setNewAlim] = useState(true);
  const [newDate, setNewDate] = useState('2026-12-18');
  const [newNotes, setNewNotes] = useState(
    'Caso sintético T0 registrado para seguimiento longitudinal.'
  );
  const [creating, setCreating] = useState(false);

  // Edit clinical plan state
  const [editStatus, setEditStatus] = useState<FollowUpStatus>('Activo');
  const [editDate, setEditDate] = useState('2026-12-15');
  const [editPlan, setEditPlan] = useState('');
  const [editNotes, setEditNotes] = useState('');
  const [savingCase, setSavingCase] = useState(false);
  const [caseFeedback, setCaseFeedback] = useState<string | null>(null);

  // New timeline entry state
  const [entryAction, setEntryAction] = useState<ActionType>('Apoyo Psicosocial');
  const [entryOutcome, setEntryOutcome] = useState<EntryOutcome>('Asistió');
  const [entryNote, setEntryNote] = useState('');
  const [savingEntry, setSavingEntry] = useState(false);

  const filteredCases = useMemo(() => {
    return followUps.filter((item) => {
      const matchesPriority =
        priorityFilter === 'Todos' || item.prioridad === priorityFilter;
      const q = searchQuery.trim().toLowerCase();
      const matchesSearch =
        !q ||
        item.syntheticCode.toLowerCase().includes(q) ||
        item.tipoFisura.toLowerCase().includes(q) ||
        item.prioridad.toLowerCase().includes(q);
      return matchesPriority && matchesSearch;
    });
  }, [followUps, priorityFilter, searchQuery]);

  const selectedCase: PatientFollowUp | null = useMemo(() => {
    if (filteredCases.length === 0) return null;
    return filteredCases.find((c) => c.id === selectedCaseId) || filteredCases[0];
  }, [filteredCases, selectedCaseId]);

  useEffect(() => {
    if (selectedCase) {
      setEditStatus(selectedCase.status);
      setEditDate(selectedCase.nextAppointmentDate);
      setEditPlan(selectedCase.interventionPlan);
      setEditNotes(selectedCase.clinicalNotes);
      setCaseFeedback(null);
    }
  }, [selectedCase?.id, selectedCase?.status]);

  useEffect(() => {
    if (!user || !selectedCase) {
      setEntries([]);
      return;
    }
    const unsub = subscribeFollowUpEntries(selectedCase.id, (list) => {
      setEntries(list);
    });
    return () => unsub();
  }, [user, selectedCase?.id, subscribeFollowUpEntries]);

  useEffect(() => {
    if (profile) {
      setProfName(profile.displayName);
      setProfSpecialty(profile.specialty);
      setProfInstitution(profile.institution);
    }
  }, [profile]);

  if (!authReady) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-8 text-center">
        <p className="text-sm text-slate-600">Verificando sesión médica segura...</p>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
        <div className="grid grid-cols-1 lg:grid-cols-12 items-stretch">
          <div className="lg:col-span-7 p-6 sm:p-10 flex flex-col justify-center space-y-5">
            <div>
              <p className="text-xs font-medium text-teal-800">
                Acceso exclusivo para especialistas médicos
              </p>
              <h1 className="text-2xl sm:text-3xl font-bold text-slate-900 mt-1.5 text-balance">
                Módulo de seguimiento clínico personalizado
              </h1>
              <p className="text-sm sm:text-base text-slate-600 mt-3 leading-relaxed max-w-xl">
                Autentícate de forma segura para gestionar tu cartera de casos sintéticos evaluados
                con `modelo_t0`, registrar notas de apoyo psicosocial y consultar la bitácora
                longitudinal con aislamiento estricto por especialista.
              </p>
            </div>

            <div className="space-y-2 text-xs text-slate-600 border-t border-slate-100 pt-4">
              <p>· Autenticación verificada mediante Google Identity y Firebase Auth</p>
              <p>· Reglas de seguridad Firestore con aislamiento estricto por UID profesional</p>
              <p>· Solo se almacenan códigos sintéticos seudonimizados (`PAC-SYN-XXXX`)</p>
            </div>

            {authError && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-lg flex items-center justify-between gap-3 text-xs text-red-800">
                <span>{authError}</span>
                <button
                  type="button"
                  onClick={clearAuthError}
                  className="text-red-700 font-semibold whitespace-nowrap"
                >
                  Cerrar
                </button>
              </div>
            )}

            <div className="pt-2">
              <button
                type="button"
                onClick={signInWithGoogle}
                className="inline-flex items-center justify-center gap-2.5 px-5 py-3 min-h-[48px] bg-teal-700 hover:bg-teal-800 text-white text-sm font-semibold rounded-lg transition-colors whitespace-nowrap"
              >
                <Lock className="w-4 h-4" />
                Iniciar sesión segura con Google
              </button>
            </div>
          </div>

          <div className="lg:col-span-5 relative min-h-[240px] bg-slate-100 border-t lg:border-t-0 lg:border-l border-slate-200">
            <img
              src={heroImage}
              alt="Unidad clínica pediátrica de atención multidisciplinaria"
              referrerPolicy="no-referrer"
              className="w-full h-full object-cover lg:absolute lg:inset-0"
            />
            <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-slate-950/80 via-slate-900/35 to-transparent p-5">
              <p className="text-xs text-white/95 font-medium">
                Al ingresar se precargan 3 casos sintéticos de `dataset_modelado_v2.csv` para tu
                sesión
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const handleSaveProfile = async (e: React.FormEvent) => {
    e.preventDefault();
    setProfileMsg(null);
    await updateSpecialistProfile({
      displayName: profName,
      specialty: profSpecialty,
      institution: profInstitution,
    });
    setEditingProfile(false);
    setProfileMsg('Perfil profesional actualizado.');
  };

  const handleCreateCase = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreating(true);
    try {
      const pred = predictT0Logistic({
        tipoFisura: newFisura,
        edadInicioMeses: newEdad,
        distanciaHospitalKm: newDist,
        nivelSocioeconomicoOrd: newSes,
        ingresosMensualesUsd: newIngresos,
        apoyoPsicologico: newApoyo,
        dificultadAlimentacionPreop: newAlim,
      });

      const createdId = await createFollowUp({
        syntheticCode: newCode,
        tipoFisura: newFisura,
        edadInicioMeses: newEdad,
        distanciaHospitalKm: newDist,
        nivelSocioeconomicoOrd: newSes,
        ingresosMensualesUsd: newIngresos,
        apoyoPsicologico: newApoyo,
        dificultadAlimentacionPreop: newAlim,
        riskScore: pred.riskScore,
        prioridad: pred.prioridad,
        status:
          pred.prioridad === 'Alta'
            ? 'Alerta Crítica'
            : pred.prioridad === 'Media'
            ? 'En Intervención'
            : 'Activo',
        nextAppointmentDate: newDate,
        interventionPlan: pred.recommendedPlan,
        clinicalNotes: newNotes,
      });

      setSelectedCaseId(createdId);
      setNewCode(`PAC-SYN-${Math.floor(2000 + Math.random() * 2999)}`);
      setShowNewModal(false);
    } finally {
      setCreating(false);
    }
  };

  const handleSaveCaseChanges = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedCase) return;
    setSavingCase(true);
    setCaseFeedback(null);
    try {
      await updateFollowUpClinical({
        id: selectedCase.id,
        status: editStatus,
        interventionPlan: editPlan,
        clinicalNotes: editNotes,
        nextAppointmentDate: editDate,
      });
      setCaseFeedback('Seguimiento actualizado y guardado en Firestore.');
    } finally {
      setSavingCase(false);
    }
  };

  const handleAddEntry = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedCase || entryNote.trim().length < 3) return;
    setSavingEntry(true);
    try {
      await addFollowUpEntry(selectedCase.id, {
        actionType: entryAction,
        outcome: entryOutcome,
        note: entryNote,
      });
      setEntryNote('');
    } finally {
      setSavingEntry(false);
    }
  };

  const handleDeleteSelected = async () => {
    if (!selectedCase) return;
    await deleteFollowUp(selectedCase.id);
    setSelectedCaseId(null);
  };

  const isTerminalRetained = selectedCase?.status === 'Retenido';

  return (
    <div className="space-y-6">
      {/* Specialist Identity Bar */}
      <section className="bg-white border border-slate-200 rounded-xl p-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-4 min-w-0">
            <img
              src={user.photoURL || specialistAvatar}
              alt={profile?.displayName || 'Especialista médico'}
              referrerPolicy="no-referrer"
              className="w-12 h-12 rounded-full object-cover border border-slate-200 shrink-0"
            />
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <h1 className="text-base sm:text-lg font-bold text-slate-900 truncate">
                  {profile?.displayName || user.displayName}
                </h1>
                <span className="text-xs text-teal-800 font-medium">
                  · Sesión verificada
                </span>
              </div>
              <p className="text-xs text-slate-500 truncate mt-0.5">
                {profile?.specialty} · {profile?.institution}
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2.5 shrink-0">
            <button
              type="button"
              onClick={() => setEditingProfile((v) => !v)}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 min-h-[44px] bg-white hover:bg-slate-50 text-slate-700 border border-slate-300 text-xs font-medium rounded-lg transition-colors whitespace-nowrap"
            >
              <UserCheck className="w-3.5 h-3.5" />
              {editingProfile ? 'Cancelar edición' : 'Editar perfil'}
            </button>
            <button
              type="button"
              onClick={() => setShowNewModal(true)}
              className="inline-flex items-center gap-1.5 px-4 py-2 min-h-[44px] bg-teal-700 hover:bg-teal-800 text-white text-xs font-semibold rounded-lg transition-colors whitespace-nowrap"
            >
              <Plus className="w-4 h-4" />
              Nuevo caso T0
            </button>
            <button
              type="button"
              onClick={signOutSpecialist}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 min-h-[44px] bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-medium rounded-lg transition-colors whitespace-nowrap"
            >
              <LogOut className="w-3.5 h-3.5" />
              Salir
            </button>
          </div>
        </div>

        {profileMsg && <p className="text-xs text-emerald-700 mt-3">{profileMsg}</p>}

        {editingProfile && (
          <form
            onSubmit={handleSaveProfile}
            className="mt-4 pt-4 border-t border-slate-100 grid grid-cols-1 md:grid-cols-4 gap-3 items-end"
          >
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">
                Nombre profesional
              </label>
              <input
                type="text"
                required
                minLength={2}
                maxLength={100}
                value={profName}
                onChange={(e) => setProfName(e.target.value)}
                className="w-full min-h-[44px] px-3 border border-slate-300 rounded-lg text-sm"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">
                Especialidad clínica
              </label>
              <input
                type="text"
                required
                minLength={2}
                maxLength={80}
                value={profSpecialty}
                onChange={(e) => setProfSpecialty(e.target.value)}
                className="w-full min-h-[44px] px-3 border border-slate-300 rounded-lg text-sm"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">
                Institución / Unidad
              </label>
              <input
                type="text"
                required
                minLength={2}
                maxLength={120}
                value={profInstitution}
                onChange={(e) => setProfInstitution(e.target.value)}
                className="w-full min-h-[44px] px-3 border border-slate-300 rounded-lg text-sm"
              />
            </div>
            <button
              type="submit"
              className="min-h-[44px] px-4 bg-slate-900 hover:bg-slate-800 text-white text-xs font-semibold rounded-lg transition-colors"
            >
              Guardar perfil
            </button>
          </form>
        )}
      </section>

      {/* Main Split Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        <section className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-4 sm:p-5 space-y-4">
          <div className="flex items-center justify-between gap-2">
            <h2 className="text-sm font-semibold text-slate-900">
              Casos sintéticos asignados ({filteredCases.length})
            </h2>
            <span className="text-[11px] text-slate-500">Puntaje `modelo_t0`</span>
          </div>

          <div className="space-y-2.5">
            <div className="relative">
              <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Buscar PAC-SYN o tipo de fisura..."
                className="w-full min-h-[44px] pl-9 pr-3 border border-slate-300 rounded-lg text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-teal-600"
              />
            </div>

            <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg">
              {(['Todos', 'Alta', 'Media', 'Habitual'] as const).map((tier) => (
                <button
                  key={tier}
                  type="button"
                  onClick={() => setPriorityFilter(tier)}
                  className={`flex-1 py-1.5 min-h-[38px] text-xs font-medium rounded-md transition-colors ${
                    priorityFilter === tier
                      ? 'bg-white text-slate-900 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {tier}
                </button>
              ))}
            </div>
          </div>

          {dataLoading ? (
            <div className="space-y-2.5 py-2">
              {[1, 2, 3].map((s) => (
                <div key={s} className="h-20 bg-slate-100 rounded-lg animate-pulse" />
              ))}
            </div>
          ) : filteredCases.length === 0 ? (
            <div className="p-6 text-center border border-dashed border-slate-200 rounded-lg space-y-3">
              <p className="text-sm text-slate-600">
                No hay casos sintéticos que coincidan con el filtro actual.
              </p>
              <button
                type="button"
                onClick={() => setShowNewModal(true)}
                className="inline-flex items-center gap-1.5 px-4 py-2 min-h-[44px] bg-teal-700 text-white text-xs font-medium rounded-lg"
              >
                <Plus className="w-4 h-4" />
                Registrar primer caso sintético
              </button>
            </div>
          ) : (
            <div className="space-y-2 max-h-[620px] overflow-y-auto pr-0.5">
              {filteredCases.map((item) => {
                const isSelected = selectedCase?.id === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setSelectedCaseId(item.id)}
                    className={`w-full text-left p-3.5 rounded-lg border transition-colors ${
                      isSelected
                        ? 'bg-teal-50/70 border-teal-600'
                        : 'bg-white border-slate-200 hover:bg-slate-50'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-mono text-xs font-bold text-slate-900 tabular-nums">
                        {item.syntheticCode}
                      </span>
                      <RiskLabel level={item.prioridad} score={item.riskScore} />
                    </div>
                    <p className="text-sm font-semibold text-slate-900 mt-1 truncate">
                      Fisura {item.tipoFisura} · {item.distanciaHospitalKm} km
                    </p>
                    <div className="flex items-center justify-between gap-2 mt-1.5 text-xs text-slate-500">
                      <span className="truncate">
                        SES {item.nivelSocioeconomicoOrd} · Apoyo:{' '}
                        {item.apoyoPsicologico ? 'Sí' : 'No'}
                      </span>
                      <StatusLabel status={item.status} />
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </section>

        {/* Case Detail & Intervention Timeline */}
        <section className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-5 sm:p-6">
          {!selectedCase ? (
            <div className="py-12 text-center text-sm text-slate-500">
              Selecciona un caso sintético de la lista para gestionar su plan de seguimiento clínico.
            </div>
          ) : (
            <div className="space-y-6">
              <div className="flex flex-wrap items-start justify-between gap-4 pb-4 border-b border-slate-100">
                <div>
                  <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
                    <span className="font-mono font-semibold text-slate-900">
                      {selectedCase.syntheticCode}
                    </span>
                    <span>·</span>
                    <span>Registro sintético v2</span>
                    <span>·</span>
                    <StatusLabel status={selectedCase.status} />
                  </div>
                  <h2 className="text-xl font-bold text-slate-900 mt-1">
                    Fisura Bilabial {selectedCase.tipoFisura}
                  </h2>
                  <p className="text-xs text-slate-600 mt-0.5 font-mono">
                    Edad inicio: {selectedCase.edadInicioMeses}m · Dist:{' '}
                    {selectedCase.distanciaHospitalKm} km · SES {selectedCase.nivelSocioeconomicoOrd} ·
                    Ingresos: ${selectedCase.ingresosMensualesUsd} USD · Apoyo psic.:{' '}
                    {selectedCase.apoyoPsicologico ? 'Sí' : 'No'}
                  </p>
                </div>

                <div className="flex items-center gap-3">
                  <div className="text-right">
                    <span className="text-xs text-slate-500 block">Probabilidad `modelo_t0`</span>
                    <span className="text-2xl font-mono font-bold text-slate-900 tabular-nums">
                      {selectedCase.riskScore}%
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={handleDeleteSelected}
                    title="Eliminar registro sintético"
                    className="p-2.5 min-h-[44px] min-w-[44px] inline-flex items-center justify-center text-slate-500 hover:text-red-700 hover:bg-red-50 border border-slate-200 rounded-lg transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {isTerminalRetained && (
                <div className="p-3.5 bg-emerald-50 border border-emerald-200 rounded-lg flex items-start gap-2.5 text-xs text-emerald-900">
                  <ShieldCheck className="w-4 h-4 shrink-0 mt-0.5 text-emerald-700" />
                  <span>
                    Este expediente sintético alcanzó estado terminal <strong>Retenido</strong>. Por
                    trazabilidad clínica, sus parámetros quedan sellados contra modificaciones
                    silenciosas.
                  </span>
                </div>
              )}

              <form onSubmit={handleSaveCaseChanges} className="space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                      Estado de seguimiento
                    </label>
                    <select
                      disabled={isTerminalRetained}
                      value={editStatus}
                      onChange={(e) => setEditStatus(e.target.value as FollowUpStatus)}
                      className="w-full min-h-[44px] px-3 border border-slate-300 rounded-lg text-sm bg-white disabled:bg-slate-100"
                    >
                      {STATUS_OPTIONS.map((st) => (
                        <option key={st} value={st}>
                          {st}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                      Próximo control programado
                    </label>
                    <div className="relative">
                      <Calendar className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
                      <input
                        type="date"
                        disabled={isTerminalRetained}
                        value={editDate}
                        onChange={(e) => setEditDate(e.target.value)}
                        className="w-full min-h-[44px] pl-9 pr-3 border border-slate-300 rounded-lg text-sm font-mono bg-white disabled:bg-slate-100"
                      />
                    </div>
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                    Plan de intervención personalizado
                  </label>
                  <input
                    type="text"
                    required
                    minLength={3}
                    maxLength={300}
                    disabled={isTerminalRetained}
                    value={editPlan}
                    onChange={(e) => setEditPlan(e.target.value)}
                    className="w-full min-h-[44px] px-3 border border-slate-300 rounded-lg text-sm disabled:bg-slate-100"
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                    Notas clínicas de evolución (sin PII real)
                  </label>
                  <textarea
                    rows={3}
                    maxLength={1000}
                    disabled={isTerminalRetained}
                    value={editNotes}
                    onChange={(e) => setEditNotes(e.target.value)}
                    className="w-full p-3 border border-slate-300 rounded-lg text-sm disabled:bg-slate-100"
                  />
                </div>

                {caseFeedback && (
                  <div className="flex items-center gap-2 text-xs text-emerald-700 font-medium">
                    <CheckCircle2 className="w-4 h-4" />
                    <span>{caseFeedback}</span>
                  </div>
                )}

                {!isTerminalRetained && (
                  <button
                    type="submit"
                    disabled={savingCase}
                    className="px-4 py-2.5 min-h-[44px] bg-slate-900 hover:bg-slate-800 disabled:opacity-60 text-white text-xs font-semibold rounded-lg transition-colors"
                  >
                    {savingCase ? 'Guardando cambios...' : 'Guardar plan de seguimiento'}
                  </button>
                )}
              </form>

              <div className="pt-5 border-t border-slate-200 space-y-4">
                <h3 className="text-sm font-semibold text-slate-900">
                  Bitácora de intervenciones y contactos ({entries.length})
                </h3>

                <form
                  onSubmit={handleAddEntry}
                  className="p-4 bg-slate-50 border border-slate-200 rounded-lg space-y-3"
                >
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <select
                      value={entryAction}
                      onChange={(e) => setEntryAction(e.target.value as ActionType)}
                      className="min-h-[44px] px-3 bg-white border border-slate-300 rounded-lg text-xs font-medium"
                    >
                      {ACTION_TYPES.map((act) => (
                        <option key={act} value={act}>
                          {act}
                        </option>
                      ))}
                    </select>
                    <select
                      value={entryOutcome}
                      onChange={(e) => setEntryOutcome(e.target.value as EntryOutcome)}
                      className="min-h-[44px] px-3 bg-white border border-slate-300 rounded-lg text-xs font-medium"
                    >
                      {ENTRY_OUTCOMES.map((out) => (
                        <option key={out} value={out}>
                          Resultado: {out}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="flex flex-col sm:flex-row gap-2">
                    <input
                      type="text"
                      required
                      minLength={3}
                      maxLength={500}
                      value={entryNote}
                      onChange={(e) => setEntryNote(e.target.value)}
                      placeholder="Ej. Asignación de apoyo psicológico al ingreso; confirma seguimiento..."
                      className="flex-1 min-h-[44px] px-3 bg-white border border-slate-300 rounded-lg text-sm"
                    />
                    <button
                      type="submit"
                      disabled={savingEntry || entryNote.trim().length < 3}
                      className="inline-flex items-center justify-center gap-1.5 px-4 min-h-[44px] bg-teal-700 hover:bg-teal-800 disabled:opacity-60 text-white text-xs font-semibold rounded-lg whitespace-nowrap"
                    >
                      <Send className="w-3.5 h-3.5" />
                      Registrar
                    </button>
                  </div>
                </form>

                {entries.length === 0 ? (
                  <p className="text-xs text-slate-500">
                    Aún no hay intervenciones registradas en la bitácora de este caso sintético.
                  </p>
                ) : (
                  <div className="space-y-2.5">
                    {entries.map((entry) => (
                      <div
                        key={entry.id}
                        className="p-3.5 bg-white border border-slate-200 rounded-lg"
                      >
                        <div className="flex items-center justify-between gap-2 text-xs">
                          <span className="font-semibold text-slate-900">
                            {entry.actionType} ·{' '}
                            <span className="text-teal-800">{entry.outcome}</span>
                          </span>
                        </div>
                        <p className="text-xs text-slate-600 mt-1 leading-relaxed">{entry.note}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </section>
      </div>

      {/* Modal: New T0 Synthetic Case */}
      {showNewModal && (
        <div className="fixed inset-0 z-50 bg-slate-950/50 flex items-center justify-center p-4 overflow-y-auto">
          <div className="bg-white border border-slate-200 rounded-xl max-w-xl w-full p-6 space-y-5 my-8">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div>
                <h2 className="text-lg font-bold text-slate-900">
                  Nuevo expediente sintético (Escenario T0)
                </h2>
                <p className="text-xs text-slate-500">
                  Evaluado en tiempo real con `modelo_t0`. Sin nombres reales ni PII.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setShowNewModal(false)}
                className="p-2 text-slate-500 hover:text-slate-800 rounded-lg"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleCreateCase} className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Código seudonimizado (`PAC-SYN-XXXX`)
                  </label>
                  <input
                    type="text"
                    required
                    pattern="^PAC-SYN-[0-9]{4}$"
                    value={newCode}
                    onChange={(e) => setNewCode(e.target.value.toUpperCase())}
                    className="w-full min-h-[44px] px-3 border border-slate-300 rounded-lg text-sm font-mono"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Próxima cita
                  </label>
                  <input
                    type="date"
                    required
                    value={newDate}
                    onChange={(e) => setNewDate(e.target.value)}
                    className="w-full min-h-[44px] px-3 border border-slate-300 rounded-lg text-sm font-mono"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Tipo de fisura (`tipo_fisura`)
                  </label>
                  <select
                    value={newFisura}
                    onChange={(e) => setNewFisura(e.target.value as FissureType)}
                    className="w-full min-h-[44px] px-3 border border-slate-300 rounded-lg text-sm bg-white"
                  >
                    {FISSURE_TYPES.map((f) => (
                      <option key={f.value} value={f.value}>
                        {f.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Nivel SES (`nivel_socioeconomico_ord`)
                  </label>
                  <select
                    value={newSes}
                    onChange={(e) => setNewSes(Number(e.target.value) as SesLevel)}
                    className="w-full min-h-[44px] px-3 border border-slate-300 rounded-lg text-sm bg-white"
                  >
                    {SES_OPTIONS.map((s) => (
                      <option key={s.value} value={s.value}>
                        {s.short}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Edad inicio (meses)
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={216}
                    value={newEdad}
                    onChange={(e) => setNewEdad(Number(e.target.value))}
                    className="w-full min-h-[44px] px-3 border border-slate-300 rounded-lg text-sm font-mono"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Distancia (km)
                  </label>
                  <input
                    type="number"
                    min={0}
                    max={1500}
                    step={0.5}
                    value={newDist}
                    onChange={(e) => setNewDist(Number(e.target.value))}
                    className="w-full min-h-[44px] px-3 border border-slate-300 rounded-lg text-sm font-mono"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Ingresos (USD/mes)
                  </label>
                  <input
                    type="number"
                    min={0}
                    max={50000}
                    value={newIngresos}
                    onChange={(e) => setNewIngresos(Number(e.target.value))}
                    className="w-full min-h-[44px] px-3 border border-slate-300 rounded-lg text-sm font-mono"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={() => setNewApoyo((v) => !v)}
                  className={`w-full min-h-[44px] px-3.5 rounded-lg border text-xs font-medium flex items-center justify-between ${
                    newApoyo
                      ? 'bg-emerald-50 border-emerald-300 text-emerald-900'
                      : 'bg-red-50 border-red-200 text-red-900'
                  }`}
                >
                  <span>Apoyo psicológico</span>
                  <span className="font-mono font-bold">{newApoyo ? 'Sí (1)' : 'No (0)'}</span>
                </button>
                <button
                  type="button"
                  onClick={() => setNewAlim((v) => !v)}
                  className={`w-full min-h-[44px] px-3.5 rounded-lg border text-xs font-medium flex items-center justify-between ${
                    newAlim
                      ? 'bg-amber-50 border-amber-300 text-amber-900'
                      : 'bg-slate-50 border-slate-200 text-slate-700'
                  }`}
                >
                  <span>Dificultad alimentación preop</span>
                  <span className="font-mono font-bold">{newAlim ? 'Sí (1)' : 'No (0)'}</span>
                </button>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Observaciones clínicas iniciales
                </label>
                <textarea
                  rows={2}
                  maxLength={1000}
                  value={newNotes}
                  onChange={(e) => setNewNotes(e.target.value)}
                  className="w-full p-3 border border-slate-300 rounded-lg text-sm"
                />
              </div>

              <div className="flex justify-end gap-2.5 pt-2">
                <button
                  type="button"
                  onClick={() => setShowNewModal(false)}
                  className="px-4 py-2.5 min-h-[44px] border border-slate-300 text-slate-700 text-xs font-medium rounded-lg"
                >
                  Cancelar
                </button>
                <button
                  type="submit"
                  disabled={creating}
                  className="px-5 py-2.5 min-h-[44px] bg-teal-700 hover:bg-teal-800 text-white text-xs font-semibold rounded-lg"
                >
                  {creating ? 'Creando...' : 'Guardar expediente T0'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
