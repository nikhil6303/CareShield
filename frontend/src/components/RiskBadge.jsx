import React from 'react';

/**
 * RiskBadge
 * Renders a color-coded badge based on the risk level in dark theme.
 */
const RiskBadge = ({ level }) => {
  const getBadgeClasses = (lvl) => {
    const base = 'inline-flex items-center px-3 py-1 rounded-full text-xs font-bold shadow-sm';
    
    switch (String(lvl).toLowerCase()) {
      case 'low':
        return `${base} bg-emerald-950/60 text-emerald-400 border border-emerald-500/30`;
      case 'medium':
        return `${base} bg-amber-950/60 text-amber-400 border border-amber-500/30`;
      case 'high':
        return `${base} bg-rose-950/60 text-rose-400 border border-rose-500/30`;
      case 'critical':
        return `${base} bg-rose-950 text-rose-300 border border-rose-500/50 animate-pulse`;
      default:
        return `${base} bg-slate-900 text-slate-400 border border-slate-700`;
    }
  };

  return (
    <span className={getBadgeClasses(level)}>
      {level || 'N/A'}
    </span>
  );
};

export default RiskBadge;
