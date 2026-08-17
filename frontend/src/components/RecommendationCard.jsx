import React from 'react';
import RiskBadge from './RiskBadge';
import { ShieldAlert, FileText, CheckCircle, HelpCircle, Activity } from 'lucide-react';

/**
 * RecommendationCard
 * Displays a single decision-support outreach suggestion with details in dark theme.
 */
const RecommendationCard = ({ action, priority, reason, evidence, suggestedNextStep, category }) => {
  const getIcon = (cat) => {
    switch (cat) {
      case 'Service Recovery':
        return <ShieldAlert size={20} className="text-rose-400" />;
      case 'Care Outreach':
        return <Activity size={20} className="text-teal-400" />;
      case 'Benefit Education':
        return <FileText size={20} className="text-indigo-400" />;
      default:
        return <HelpCircle size={20} className="text-slate-400" />;
    }
  };

  const getPriorityClasses = (prio) => {
    switch (String(prio).toLowerCase()) {
      case 'critical':
      case 'high':
        return 'border-l-4 border-l-rose-500';
      case 'medium':
        return 'border-l-4 border-l-amber-500';
      default:
        return 'border-l-4 border-l-emerald-500';
    }
  };

  return (
    <div className={`bg-[#111827]/80 backdrop-blur-md rounded-2xl border border-slate-800 p-6 flex flex-col gap-4 shadow-xl ${getPriorityClasses(priority)}`}>
      {/* Header (Action & Priority) */}
      <div className="flex justify-between items-start gap-4 flex-wrap">
        <div className="flex items-center gap-3">
          <span className="p-2.5 rounded-xl bg-slate-900 border border-slate-800">
            {getIcon(category)}
          </span>
          <h4 className="text-base font-bold text-white font-heading">{action}</h4>
        </div>
        <RiskBadge level={priority} />
      </div>

      {/* Description / Reason */}
      <div className="flex flex-col gap-1">
        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Reason for Action</span>
        <p className="text-xs text-slate-300 leading-relaxed">{reason}</p>
      </div>

      {/* Evidence */}
      {evidence && evidence.length > 0 && (
        <div className="flex flex-col gap-2">
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Observable Evidence</span>
          <ul className="list-inside list-disc pl-2 flex flex-col gap-1">
            {evidence.map((ev, idx) => (
              <li key={idx} className="text-xs text-slate-300">{ev}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Suggested Next Step */}
      <div className="bg-emerald-950/20 rounded-xl p-3.5 border border-emerald-500/20 flex items-start gap-2.5 mt-1">
        <CheckCircle size={16} className="text-emerald-400 mt-0.5 flex-shrink-0" />
        <div className="flex flex-col gap-0.5">
          <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider">Suggested Next Step</span>
          <p className="text-xs text-slate-200 leading-relaxed">{suggestedNextStep}</p>
        </div>
      </div>
    </div>
  );
};

export default RecommendationCard;
