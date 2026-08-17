import React, { useState, useEffect } from 'react';
import RiskBadge from './RiskBadge';
import DriverBar from './DriverBar';
import RecommendationCard from './RecommendationCard';
import { 
  UserCheck, ArrowLeft, ShieldAlert, Sparkles, Activity, 
  Calendar, MapPin, DollarSign, Clock, CheckCircle2, User, Search
} from 'lucide-react';

const RetentionAdvisor = ({ members = [], selectedMemberId, onSelectMember, API_URL }) => {
  const [detailLoading, setDetailLoading] = useState(false);
  const [memberDetail, setMemberDetail] = useState(null);
  const [error, setError] = useState('');

  // Fetch member details from API or cache whenever selectedMemberId changes
  useEffect(() => {
    if (!selectedMemberId) {
      setMemberDetail(null);
      setError('');
      return;
    }

    const fetchDetail = async () => {
      setDetailLoading(true);
      setError('');
      try {
        const res = await fetch(`${API_URL}/member/${selectedMemberId}`);
        if (!res.ok) {
          throw new Error(`Failed to load member details for ${selectedMemberId}`);
        }
        const data = await res.json();
        setMemberDetail(data);
      } catch (err) {
        console.error("Error fetching member details:", err);
        setError(`Could not load retention assessment for member ${selectedMemberId}. Ensure backend is running.`);
      } finally {
        setDetailLoading(false);
      }
    };

    fetchDetail();
  }, [selectedMemberId, API_URL]);

  return (
    <div className="flex flex-col gap-6 max-w-6xl mx-auto text-white select-none">
      {/* Overview Banner Header */}
      <div className="bg-[#111827]/80 backdrop-blur-md border border-slate-800 rounded-2xl p-6 flex flex-col md:flex-row justify-between items-start md:items-center gap-4 shadow-xl">
        <div>
          <div className="flex items-center gap-2 text-indigo-400 font-bold text-xs uppercase tracking-wider mb-1">
            <Sparkles size={16} />
            <span>Retention Advisor & Decision Support</span>
          </div>
          <h2 className="text-xl font-extrabold text-white tracking-tight font-heading">
            Individual Member Retention Assessment
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Prioritized evidence-based retention actions, risk drivers, and churn probability analysis.
          </p>
        </div>

        {/* Member Selector Bar */}
        <div className="flex items-center gap-3 bg-slate-900/90 border border-slate-700/80 rounded-xl p-2.5 shadow-inner">
          <UserCheck size={18} className="text-teal-400 ml-1" />
          <span className="text-xs font-bold text-slate-300 uppercase tracking-wide">Select Member:</span>
          <select 
            value={selectedMemberId || ''} 
            onChange={(e) => onSelectMember(e.target.value || null)}
            className="bg-[#0b0f19] border border-slate-700 text-white text-xs font-bold rounded-lg px-3 py-2 outline-none focus:border-indigo-500 cursor-pointer min-w-[200px]"
          >
            <option value="">-- Choose Member ID --</option>
            {members.map(m => (
              <option key={m.member_id} value={m.member_id}>
                {m.member_id} ({(m.churn_probability * 100).toFixed(0)}% - {m.risk_level})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* State A: No Member Selected */}
      {!selectedMemberId && (
        <div className="bg-[#111827]/60 border border-slate-800 border-dashed rounded-2xl p-16 flex flex-col items-center justify-center text-center gap-4 shadow-xl">
          <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 text-indigo-400 flex items-center justify-center border border-indigo-500/20 shadow-lg">
            <Search size={32} />
          </div>
          <div>
            <h3 className="text-lg font-bold text-white font-heading">Select a member to view their retention assessment</h3>
            <p className="text-xs text-slate-400 mt-1 max-w-md">
              Choose any member ID from the dropdown above or click a member row in the Strategic Dashboard table to view their actual SHAP drivers and prioritized 1-3 retention recommendations.
            </p>
          </div>
        </div>
      )}

      {/* State B: Loading Member Detail */}
      {selectedMemberId && detailLoading && (
        <div className="bg-[#111827]/80 border border-slate-800 rounded-2xl p-24 flex flex-col items-center justify-center gap-3 text-slate-400">
          <div className="w-10 h-10 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm font-semibold text-slate-300">Evaluating SHAP risk drivers and consolidated actions for member {selectedMemberId}...</span>
        </div>
      )}

      {/* State C: Error Loading Member */}
      {selectedMemberId && !detailLoading && error && (
        <div className="bg-rose-950/40 border border-rose-500/40 text-rose-300 rounded-2xl p-6 text-sm font-semibold flex items-center gap-3">
          <ShieldAlert size={20} className="text-rose-400 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* State D: Loaded Member Assessment */}
      {selectedMemberId && !detailLoading && memberDetail && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column: Member Profile & Risk Score */}
          <div className="lg:col-span-1 flex flex-col gap-6">
            {/* Member Info Profile Card */}
            <div className="bg-[#111827]/80 border border-slate-800 rounded-2xl p-6 flex flex-col gap-5 shadow-xl">
              <div>
                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Healthcare Member ID</span>
                <h3 className="text-2xl font-extrabold text-white tracking-tight font-heading mt-0.5">
                  {memberDetail.member_info.PatientID}
                </h3>
              </div>

              <div className="grid grid-cols-2 gap-4 border-t border-b border-slate-800 py-4 text-xs">
                <div>
                  <span className="text-[10px] text-slate-400 uppercase block font-semibold">Insurance Plan</span>
                  <strong className="text-white font-bold">{memberDetail.member_info.Insurance_Type}</strong>
                </div>
                <div>
                  <span className="text-[10px] text-slate-400 uppercase block font-semibold">Tenure</span>
                  <strong className="text-white font-bold">{memberDetail.member_info.Tenure_Months} months</strong>
                </div>
                <div>
                  <span className="text-[10px] text-slate-400 uppercase block font-semibold">Specialty Care</span>
                  <strong className="text-white font-bold truncate block" title={memberDetail.member_info.Specialty}>
                    {memberDetail.member_info.Specialty}
                  </strong>
                </div>
                <div>
                  <span className="text-[10px] text-slate-400 uppercase block font-semibold">Age / Gender</span>
                  <strong className="text-white font-bold">
                    {memberDetail.member_info.Age} yrs / {memberDetail.member_info.Gender}
                  </strong>
                </div>
              </div>

              <div className="flex flex-col gap-1 text-xs">
                <span className="text-[10px] text-slate-400 uppercase block font-semibold">Last Recorded Clinical Visit</span>
                <span className="text-slate-300 font-semibold">
                  {memberDetail.member_info.Days_Since_Last_Visit} days ago ({memberDetail.member_info.Last_Interaction_Date})
                </span>
              </div>
            </div>

            {/* Risk Gauge Card */}
            <div className="bg-[#111827]/80 border border-slate-800 rounded-2xl p-6 flex flex-col items-center justify-center text-center gap-4 relative overflow-hidden shadow-xl">
              <div className={`absolute top-0 left-0 w-full h-1.5 ${
                memberDetail.risk_level === 'Critical' || memberDetail.risk_level === 'High' ? 'bg-rose-500' : 
                (memberDetail.risk_level === 'Medium' ? 'bg-amber-500' : 'bg-emerald-500')
              }`} />

              <span className="text-xs font-bold text-slate-400 uppercase tracking-wider block">Predicted Churn Risk</span>
              <div className="relative flex items-center justify-center">
                <span className={`text-6xl font-black font-heading ${
                  memberDetail.risk_level === 'Critical' || memberDetail.risk_level === 'High' ? 'text-rose-400' : 
                  (memberDetail.risk_level === 'Medium' ? 'text-amber-400' : 'text-emerald-400')
                }`}>
                  {(memberDetail.churn_probability * 100).toFixed(0)}%
                </span>
              </div>
              
              <RiskBadge level={memberDetail.risk_level} />
              <p className="text-xs text-slate-400 leading-relaxed px-2">
                Evaluated from XGBoost ensemble ML model and SHAP risk driver weights.
              </p>
            </div>
          </div>

          {/* Right Column: SHAP Drivers & Prioritized Recommendations */}
          <div className="lg:col-span-2 flex flex-col gap-6">
            {/* Top Churn Drivers Card */}
            <section className="bg-[#111827]/80 border border-slate-800 rounded-2xl p-6 flex flex-col gap-4 shadow-xl">
              <div>
                <h3 className="text-sm font-bold text-white uppercase tracking-wide font-heading">Why is this member at risk?</h3>
                <p className="text-xs text-slate-400">SHAP feature contributions pushing risk towards higher risk (+ Red) or retention (- Green)</p>
              </div>

              <div className="flex flex-col mt-1">
                {memberDetail.drivers.map((driver, idx) => (
                  <DriverBar 
                    key={idx}
                    label={driver.label}
                    value={driver.value}
                    shapValue={driver.shap_value}
                    maxShap={Math.max(...memberDetail.drivers.map(d => Math.abs(d.shap_value)))}
                  />
                ))}
              </div>
            </section>

            {/* Consolidated Prioritized Retention Actions */}
            <section className="flex flex-col gap-4">
              <div>
                <h3 className="text-sm font-bold text-white uppercase tracking-wide font-heading">Prioritized Retention Actions</h3>
                <p className="text-xs text-slate-400">Targeted 1 to 3 consolidated retention strategies grouping observable risk signals</p>
              </div>

              <div className="flex flex-col gap-4">
                {memberDetail.recommendations.map((rec, idx) => (
                  <RecommendationCard 
                    key={idx}
                    action={rec.action}
                    priority={rec.priority}
                    reason={rec.reason}
                    evidence={rec.evidence}
                    suggestedNextStep={rec.suggested_next_step}
                    category={rec.category}
                  />
                ))}
              </div>
            </section>
          </div>
        </div>
      )}
    </div>
  );
};

export default RetentionAdvisor;
