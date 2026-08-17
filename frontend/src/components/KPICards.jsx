import React from 'react';
import { Users, AlertTriangle, Activity, HeartHandshake } from 'lucide-react';

/**
 * KPICards
 * Displays top dashboard statistics as standard enterprise SaaS metric cards in dark theme.
 */
const KPICards = ({ total, highRisk, avgRisk, opportunities }) => {
  const cards = [
    {
      title: 'Total Members',
      value: total ? total.toLocaleString() : '0',
      icon: Users,
      colorClass: 'text-indigo-400',
      bgIcon: 'bg-indigo-500/10 border-indigo-500/20',
      borderColor: 'border-l-indigo-500'
    },
    {
      title: 'High / Critical Risk',
      value: highRisk ? highRisk.toLocaleString() : '0',
      icon: AlertTriangle,
      colorClass: 'text-rose-400',
      bgIcon: 'bg-rose-500/10 border-rose-500/20',
      borderColor: 'border-l-rose-500'
    },
    {
      title: 'Average Churn Risk',
      value: typeof avgRisk === 'number' ? (avgRisk * 100).toFixed(1) + '%' : '0.0%',
      icon: Activity,
      colorClass: 'text-teal-400',
      bgIcon: 'bg-teal-500/10 border-teal-500/20',
      borderColor: 'border-l-teal-500'
    },
    {
      title: 'Retention Opportunities',
      value: opportunities ? opportunities.toLocaleString() : '0',
      icon: HeartHandshake,
      colorClass: 'text-emerald-400',
      bgIcon: 'bg-emerald-500/10 border-emerald-500/20',
      borderColor: 'border-l-emerald-500'
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-4">
      {cards.map((card, idx) => {
        const Icon = card.icon;
        return (
          <div 
            key={idx} 
            className={`bg-[#111827]/80 backdrop-blur-md rounded-2xl border border-slate-800 border-l-4 ${card.borderColor} p-6 flex justify-between items-center transition-all duration-300 hover:-translate-y-1 shadow-xl`}
          >
            <div className="flex flex-col gap-1">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                {card.title}
              </span>
              <span className="text-3xl font-extrabold text-white font-heading">
                {card.value}
              </span>
            </div>
            <div className={`p-3.5 rounded-xl border ${card.bgIcon} ${card.colorClass}`}>
              <Icon size={24} />
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default KPICards;
