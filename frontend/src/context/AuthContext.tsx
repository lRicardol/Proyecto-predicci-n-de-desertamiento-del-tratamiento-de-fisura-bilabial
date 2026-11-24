import React, { createContext, useCallback, useContext, useEffect, useState } from 'react';
import {
  User,
  onAuthStateChanged,
  signInWithPopup,
  signOut as firebaseSignOut,
} from 'firebase/auth';
import {
  collection,
  deleteDoc,
  doc,
  getDoc,
  onSnapshot,
  query,
  serverTimestamp,
  setDoc,
  updateDoc,
  where,
  writeBatch,
} from 'firebase/firestore';
import {
  auth,
  db,
  googleProvider,
  handleFirestoreError,
  OperationType,
} from '../lib/firebase';
import { buildInitialSyntheticFollowUps } from '../data/syntheticCohort';
import {
  ActionType,
  EntryOutcome,
  FissureType,
  FollowUpEntry,
  FollowUpStatus,
  PatientFollowUp,
  PriorityTier,
  SesLevel,
  SpecialistProfile,
} from '../types/clinical';

interface NewFollowUpPayload {
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
}

interface UpdateFollowUpClinicalPayload {
  id: string;
  status: FollowUpStatus;
  interventionPlan: string;
  clinicalNotes: string;
  nextAppointmentDate: string;
}

interface AuthContextValue {
  user: User | null;
  profile: SpecialistProfile | null;
  followUps: PatientFollowUp[];
  authReady: boolean;
  dataLoading: boolean;
  authError: string | null;
  signInWithGoogle: () => Promise<void>;
  signOutSpecialist: () => Promise<void>;
  updateSpecialistProfile: (data: {
    displayName: string;
    specialty: string;
    institution: string;
  }) => Promise<void>;
  createFollowUp: (payload: NewFollowUpPayload) => Promise<string>;
  updateFollowUpClinical: (payload: UpdateFollowUpClinicalPayload) => Promise<void>;
  deleteFollowUp: (id: string) => Promise<void>;
  addFollowUpEntry: (
    followUpId: string,
    payload: { actionType: ActionType; outcome: EntryOutcome; note: string }
  ) => Promise<void>;
  subscribeFollowUpEntries: (
    followUpId: string,
    onData: (entries: FollowUpEntry[]) => void
  ) => () => void;
  clearAuthError: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

function sanitizeCode(raw: string): string {
  const trimmed = raw.trim().toUpperCase();
  if (/^PAC-SYN-[0-9]{4}$/.test(trimmed)) {
    return trimmed;
  }
  const randomDigits = Math.floor(2000 + Math.random() * 2999);
  return `PAC-SYN-${randomDigits}`;
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [profile, setProfile] = useState<SpecialistProfile | null>(null);
  const [followUps, setFollowUps] = useState<PatientFollowUp[]>([]);
  const [authReady, setAuthReady] = useState(false);
  const [dataLoading, setDataLoading] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);

  const ensureSpecialistProfile = useCallback(async (firebaseUser: User) => {
    const specialistRef = doc(db, 'specialists', firebaseUser.uid);
    try {
      const snap = await getDoc(specialistRef);
      if (!snap.exists()) {
        const displayName = (firebaseUser.displayName || 'Especialista Clínico').slice(0, 100);
        const email = (firebaseUser.email || 'especialista@clinica.local').slice(0, 160);
        const initialProfile: SpecialistProfile = {
          uid: firebaseUser.uid,
          displayName: displayName.length >= 2 ? displayName : 'Especialista Clínico',
          email,
          specialty: 'Cirugía Craneofacial y Seguimiento',
          institution: 'Unidad Multidisciplinaria de Fisura Labiopalatina',
          role: 'specialist',
        };

        const batch = writeBatch(db);
        batch.set(specialistRef, {
          ...initialProfile,
          createdAt: serverTimestamp(),
          updatedAt: serverTimestamp(),
        });

        const seedCases = buildInitialSyntheticFollowUps(firebaseUser.uid);
        for (const item of seedCases) {
          const { id, ...fields } = item;
          const caseRef = doc(db, 'followUps', id + '_' + firebaseUser.uid.slice(0, 8));
          batch.set(caseRef, {
            ...fields,
            createdAt: serverTimestamp(),
            updatedAt: serverTimestamp(),
          });
        }

        await batch.commit();
        setProfile(initialProfile);
      } else {
        setProfile(snap.data() as SpecialistProfile);
      }
    } catch (error) {
      handleFirestoreError(error, OperationType.WRITE, `specialists/${firebaseUser.uid}`);
    }
  }, []);

  useEffect(() => {
    const unsubscribeAuth = onAuthStateChanged(auth, async (firebaseUser) => {
      setUser(firebaseUser);
      if (!firebaseUser) {
        setProfile(null);
        setFollowUps([]);
        setAuthReady(true);
        setDataLoading(false);
        return;
      }

      setDataLoading(true);
      try {
        await ensureSpecialistProfile(firebaseUser);
      } catch (err) {
        setAuthError(
          err instanceof Error && /insufficient permissions/i.test(err.message)
            ? 'Tu cuenta de Google no está autorizada para usar el seguimiento. Pide al administrador que agregue tu correo.'
            : err instanceof Error
            ? 'No fue posible inicializar el perfil clínico del especialista.'
            : 'Error al conectar con el perfil de especialista.'
        );
      } finally {
        setAuthReady(true);
      }
    });

    return () => unsubscribeAuth();
  }, [ensureSpecialistProfile]);

  useEffect(() => {
    if (!authReady || !user) {
      return;
    }

    setDataLoading(true);
    const q = query(
      collection(db, 'followUps'),
      where('specialistId', '==', user.uid)
    );

    const unsubscribeSnap = onSnapshot(
      q,
      (snapshot) => {
        const rows: PatientFollowUp[] = snapshot.docs
          .map((d) => ({
            id: d.id,
            ...(d.data() as Omit<PatientFollowUp, 'id'>),
          }))
          .sort((a, b) => b.riskScore - a.riskScore);
        setFollowUps(rows);
        setDataLoading(false);
      },
      (error) => {
        setDataLoading(false);
        handleFirestoreError(error, OperationType.LIST, 'followUps');
      }
    );

    return () => unsubscribeSnap();
  }, [authReady, user]);

  const signInWithGoogle = useCallback(async () => {
    setAuthError(null);
    try {
      await signInWithPopup(auth, googleProvider);
    } catch (error: unknown) {
      const code = (error as { code?: string })?.code;
      if (code === 'auth/popup-closed-by-user') {
        setAuthError('Inicio de sesión cancelado antes de completar la verificación.');
      } else if (code === 'auth/popup-blocked') {
        setAuthError('El navegador bloqueó la ventana de autenticación. Habilita ventanas emergentes e intenta nuevamente.');
      } else {
        setAuthError('No fue posible completar la autenticación con Google. Intenta nuevamente.');
      }
    }
  }, []);

  const signOutSpecialist = useCallback(async () => {
    setAuthError(null);
    await firebaseSignOut(auth);
  }, []);

  const updateSpecialistProfile = useCallback(
    async (data: { displayName: string; specialty: string; institution: string }) => {
      if (!user || !profile) return;
      const cleanName = data.displayName.trim().slice(0, 100);
      const cleanSpecialty = data.specialty.trim().slice(0, 80);
      const cleanInstitution = data.institution.trim().slice(0, 120);
      if (cleanName.length < 2 || cleanSpecialty.length < 2 || cleanInstitution.length < 2) {
        throw new Error('Todos los campos profesionales deben tener al menos 2 caracteres.');
      }

      const path = `specialists/${user.uid}`;
      try {
        await updateDoc(doc(db, 'specialists', user.uid), {
          displayName: cleanName,
          specialty: cleanSpecialty,
          institution: cleanInstitution,
          updatedAt: serverTimestamp(),
        });
        setProfile({
          ...profile,
          displayName: cleanName,
          specialty: cleanSpecialty,
          institution: cleanInstitution,
        });
      } catch (error) {
        handleFirestoreError(error, OperationType.UPDATE, path);
      }
    },
    [user, profile]
  );

  const createFollowUp = useCallback(
    async (payload: NewFollowUpPayload): Promise<string> => {
      if (!user) throw new Error('Debes iniciar sesión como especialista.');
      const docId = `case_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
      const path = `followUps/${docId}`;

      const cleanPlan = payload.interventionPlan.trim().slice(0, 300);
      const cleanNotes = payload.clinicalNotes.trim().slice(0, 1000);
      const validDate = /^[0-9]{4}-[0-9]{2}-[0-9]{2}$/.test(payload.nextAppointmentDate)
        ? payload.nextAppointmentDate
        : '2026-12-15';

      try {
        await setDoc(doc(db, 'followUps', docId), {
          specialistId: user.uid,
          syntheticCode: sanitizeCode(payload.syntheticCode),
          tipoFisura: payload.tipoFisura,
          edadInicioMeses: Math.max(0, Math.min(216, Math.round(payload.edadInicioMeses))),
          distanciaHospitalKm: Math.max(0, Math.min(1500, Number(payload.distanciaHospitalKm.toFixed(1)))),
          nivelSocioeconomicoOrd: payload.nivelSocioeconomicoOrd,
          ingresosMensualesUsd: Math.max(0, Math.min(50000, Math.round(payload.ingresosMensualesUsd))),
          apoyoPsicologico: Boolean(payload.apoyoPsicologico),
          dificultadAlimentacionPreop: Boolean(payload.dificultadAlimentacionPreop),
          riskScore: Math.max(0, Math.min(100, Number(payload.riskScore.toFixed(1)))),
          prioridad: payload.prioridad,
          status: payload.status,
          nextAppointmentDate: validDate,
          interventionPlan: cleanPlan.length >= 3 ? cleanPlan : 'Seguimiento clínico estándar.',
          clinicalNotes: cleanNotes,
          isSynthetic: true,
          createdAt: serverTimestamp(),
          updatedAt: serverTimestamp(),
        });
        return docId;
      } catch (error) {
        handleFirestoreError(error, OperationType.CREATE, path);
      }
    },
    [user]
  );

  const updateFollowUpClinical = useCallback(
    async (payload: UpdateFollowUpClinicalPayload) => {
      if (!user) throw new Error('Debes iniciar sesión como especialista.');
      const path = `followUps/${payload.id}`;
      const cleanPlan = payload.interventionPlan.trim().slice(0, 300);
      const cleanNotes = payload.clinicalNotes.trim().slice(0, 1000);
      const validDate = /^[0-9]{4}-[0-9]{2}-[0-9]{2}$/.test(payload.nextAppointmentDate)
        ? payload.nextAppointmentDate
        : '2026-12-15';

      try {
        await updateDoc(doc(db, 'followUps', payload.id), {
          status: payload.status,
          interventionPlan: cleanPlan.length >= 3 ? cleanPlan : 'Seguimiento clínico activo.',
          clinicalNotes: cleanNotes,
          nextAppointmentDate: validDate,
          updatedAt: serverTimestamp(),
        });
      } catch (error) {
        handleFirestoreError(error, OperationType.UPDATE, path);
      }
    },
    [user]
  );

  const deleteFollowUp = useCallback(
    async (id: string) => {
      if (!user) throw new Error('Debes iniciar sesión como especialista.');
      const path = `followUps/${id}`;
      try {
        await deleteDoc(doc(db, 'followUps', id));
      } catch (error) {
        handleFirestoreError(error, OperationType.DELETE, path);
      }
    },
    [user]
  );

  const addFollowUpEntry = useCallback(
    async (
      followUpId: string,
      payload: { actionType: ActionType; outcome: EntryOutcome; note: string }
    ) => {
      if (!user) throw new Error('Debes iniciar sesión como especialista.');
      const entryId = `entry_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
      const path = `followUps/${followUpId}/entries/${entryId}`;
      const cleanNote = payload.note.trim().slice(0, 500);
      if (cleanNote.length < 3) {
        throw new Error('La nota clínica debe tener al menos 3 caracteres.');
      }

      try {
        await setDoc(doc(db, 'followUps', followUpId, 'entries', entryId), {
          followUpId,
          specialistId: user.uid,
          actionType: payload.actionType,
          outcome: payload.outcome,
          note: cleanNote,
          createdAt: serverTimestamp(),
        });
      } catch (error) {
        handleFirestoreError(error, OperationType.CREATE, path);
      }
    },
    [user]
  );

  const subscribeFollowUpEntries = useCallback(
    (followUpId: string, onData: (entries: FollowUpEntry[]) => void) => {
      if (!user) {
        onData([]);
        return () => {};
      }
      const path = `followUps/${followUpId}/entries`;
      const q = query(
        collection(db, 'followUps', followUpId, 'entries'),
        where('specialistId', '==', user.uid)
      );

      return onSnapshot(
        q,
        (snapshot) => {
          const items: FollowUpEntry[] = snapshot.docs
            .map((d) => ({
              id: d.id,
              ...(d.data() as Omit<FollowUpEntry, 'id'>),
            }))
            .sort((a, b) => {
              const aSec = (a.createdAt as { seconds?: number } | undefined)?.seconds ?? 0;
              const bSec = (b.createdAt as { seconds?: number } | undefined)?.seconds ?? 0;
              return bSec - aSec;
            });
          onData(items);
        },
        (error) => {
          handleFirestoreError(error, OperationType.LIST, path);
        }
      );
    },
    [user]
  );

  const clearAuthError = useCallback(() => setAuthError(null), []);

  return (
    <AuthContext.Provider
      value={{
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
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth debe utilizarse dentro de AuthProvider.');
  }
  return ctx;
}
