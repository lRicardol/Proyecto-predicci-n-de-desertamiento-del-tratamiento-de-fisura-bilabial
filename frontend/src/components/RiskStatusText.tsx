import React from 'react';
import { AlertCircle, CheckCircle2, ShieldAlert } from 'lucide-react';
import { FollowUpStatus, PriorityTier } from '../types/clinical';

export const RiskLabel: React.FC<{ level: PriorityTier; score?: number }> = ({
  level,
  score,
}) => {
  if (level === 'Alta') {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-red-700 whitespace-nowrap">
        <ShieldAlert className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
        <span>
          Prioridad Alta{typeof score === 'number' ? ` · ${score}%` : ''}
        </span>
      </span>
    );
  }
  if (level === 'Media') {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-amber-700 whitespace-nowrap">
        <AlertCircle className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
        <span>
          Prioridad Media{typeof score === 'number' ? ` · ${score}%` : ''}
        </span>
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-emerald-700 whitespace-nowrap">
      <CheckCircle2 className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
      <span>
        Prioridad Habitual{typeof score === 'number' ? ` · ${score}%` : ''}
      </span>
    </span>
  );
};

export const StatusLabel: React.FC<{ status: FollowUpStatus }> = ({ status }) => {
  const colorMap: Record<FollowUpStatus, string> = {
    'Alerta Crítica': 'text-red-700',
    'En Intervención': 'text-amber-700',
    Activo: 'text-teal-800',
    Retenido: 'text-emerald-700',
  };

  return (
    <span className={`text-xs font-semibold whitespace-nowrap ${colorMap[status]}`}>
      {status}
    </span>
  );
};
