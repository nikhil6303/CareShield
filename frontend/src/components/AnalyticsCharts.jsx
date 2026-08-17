import React from 'react';
import { 
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, Cell 
} from 'recharts';

/**
 * CustomTooltip
 * Dark theme tooltip matching obsidian visual design.
 */
const CustomTooltip = ({ active, payload, label, formatter }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-[#1e293b] border border-slate-700 p-3 rounded-xl shadow-xl text-xs text-white">
        <p className="font-bold text-slate-200 mb-1">{label}</p>
        {payload.map((item, idx) => (
          <p key={idx} className="font-semibold" style={{ color: item.color || item.fill }}>
            {item.name}: {formatter ? formatter(item.value) : item.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

/**
 * RiskDistributionChart
 * Displays Risk Distribution across Low, Medium, High, and Critical classes in dark theme.
 */
export const RiskDistributionChart = ({ distribution }) => {
  const data = [
    { name: 'Low Risk', count: distribution.Low || 0, color: '#10b981' },
    { name: 'Medium Risk', count: distribution.Medium || 0, color: '#f59e0b' },
    { name: 'High Risk', count: distribution.High || 0, color: '#f43f5e' },
    { name: 'Critical Risk', count: distribution.Critical || 0, color: '#e11d48' }
  ];

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
        <XAxis 
          dataKey="name" 
          tick={{ fill: '#94a3b8', fontSize: 10, fontWeight: 500 }}
          axisLine={{ stroke: '#334155' }}
          tickLine={false}
        />
        <YAxis 
          tick={{ fill: '#94a3b8', fontSize: 10 }}
          axisLine={{ stroke: '#334155' }}
          tickLine={false}
        />
        <Tooltip content={<CustomTooltip formatter={(val) => val.toLocaleString() + ' members'} />} />
        <Bar dataKey="count" radius={[6, 6, 0, 0]}>
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
};

/**
 * RiskTrendChart
 * Displays the Average Probability curve vs Tenure months cohorts in dark theme.
 */
export const RiskTrendChart = ({ trend }) => {
  const data = trend ? trend.map(t => ({
    name: t.cohort,
    'Average Churn Risk': t.avg_probability
  })) : [];

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data} margin={{ top: 10, right: 20, left: -20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis 
          dataKey="name" 
          tick={{ fill: '#94a3b8', fontSize: 10, fontWeight: 500 }}
          axisLine={{ stroke: '#334155' }}
          tickLine={false}
        />
        <YAxis 
          tick={{ fill: '#94a3b8', fontSize: 10 }}
          axisLine={{ stroke: '#334155' }}
          tickLine={false}
          tickFormatter={(val) => (val * 100).toFixed(0) + '%'}
        />
        <Tooltip content={<CustomTooltip formatter={(val) => (val * 100).toFixed(1) + '%'} />} />
        <Line 
          type="monotone" 
          dataKey="Average Churn Risk" 
          stroke="#0d9488" 
          strokeWidth={3}
          activeDot={{ r: 6 }}
          dot={{ r: 3, stroke: '#0d9488', strokeWidth: 2, fill: '#0b0f19' }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

/**
 * TopDriversChart
 * Displays the Top 5 Global SHAP drivers horizontally in dark theme.
 */
export const TopDriversChart = ({ drivers }) => {
  const data = drivers ? [...drivers].reverse().map(d => ({
    name: d.label,
    'Impact Strength': d.mean_importance
  })) : [];

  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart 
        layout="vertical" 
        data={data} 
        margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
        <XAxis 
          type="number" 
          tick={{ fill: '#94a3b8', fontSize: 10 }}
          axisLine={{ stroke: '#334155' }}
          tickLine={false}
        />
        <YAxis 
          type="category" 
          dataKey="name" 
          tick={{ fill: '#cbd5e1', fontSize: 10, fontWeight: 500 }}
          axisLine={{ stroke: '#334155' }}
          tickLine={false}
          width={180}
        />
        <Tooltip content={<CustomTooltip formatter={(val) => val.toFixed(4)} />} />
        <Bar dataKey="Impact Strength" fill="#6366f1" radius={[0, 4, 4, 0]} barSize={20} />
      </BarChart>
    </ResponsiveContainer>
  );
};
