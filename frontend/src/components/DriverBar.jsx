import React from 'react';

/**
 * DriverBar
 * Displays a single SHAP driver impact strength and direction as a professional horizontal bar in dark theme.
 */
const DriverBar = ({ label, value, shapValue, maxShap = 1.0 }) => {
  const isPositive = shapValue > 0;
  const divisor = maxShap || 1.0;
  const percentage = Math.min(Math.round((Math.abs(shapValue) / divisor) * 100), 100);

  return (
    <div className="flex flex-col md:flex-row md:items-center justify-between py-3 border-b border-slate-800/80 last:border-b-0 gap-4">
      {/* Label and Value */}
      <div className="flex-1">
        <span className="text-sm font-semibold text-slate-200 block">{label}</span>
        <span className="text-xs text-slate-400">Actual Value: <strong className="text-white font-bold">{value}</strong></span>
      </div>

      {/* Visual Bar showing impact */}
      <div className="flex items-center gap-3 w-full md:w-72">
        {/* Decrease indicator (Green, aligned right) */}
        <div className="w-1/2 flex justify-end">
          {!isPositive && (
            <div className="flex items-center gap-1.5 justify-end w-full">
              <span className="text-[10px] font-bold text-emerald-400">-{Math.abs(shapValue).toFixed(3)}</span>
              <div 
                className="h-3 rounded-l bg-emerald-500 transition-all duration-500 shadow-sm shadow-emerald-500/50" 
                style={{ width: `${percentage}%` }}
              />
            </div>
          )}
        </div>

        {/* Center line dividing positive/negative */}
        <div className="w-[2px] h-6 bg-slate-700 relative z-10" />

        {/* Increase indicator (Red, aligned left) */}
        <div className="w-1/2 flex justify-start">
          {isPositive && (
            <div className="flex items-center gap-1.5 justify-start w-full">
              <div 
                className="h-3 rounded-r bg-rose-500 transition-all duration-500 shadow-sm shadow-rose-500/50" 
                style={{ width: `${percentage}%` }}
              />
              <span className="text-[10px] font-bold text-rose-400">+{Math.abs(shapValue).toFixed(3)}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DriverBar;
