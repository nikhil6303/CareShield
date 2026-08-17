import React, { useState, useEffect, useCallback } from 'react';
import { 
  LayoutDashboard, Users, ArrowLeft, RefreshCw, HeartHandshake,
  ShieldCheck, UploadCloud, UserCheck, Sliders, Activity
} from 'lucide-react';

import KPICards from './components/KPICards';
import RiskBadge from './components/RiskBadge';
import MemberTable from './components/MemberTable';
import DriverBar from './components/DriverBar';
import RecommendationCard from './components/RecommendationCard';
import DatasetUpload from './components/DatasetUpload';
import RetentionAdvisor from './components/RetentionAdvisor';
import { RiskDistributionChart, RiskTrendChart, TopDriversChart } from './components/AnalyticsCharts';

// Base API URL: Points to local Flask backend during local Vite dev (port 3000/5173), and defaults to relative paths in production
const API_URL = (import.meta.env.VITE_API_URL !== undefined && import.meta.env.VITE_API_URL !== '')
  ? import.meta.env.VITE_API_URL
  : (typeof window !== 'undefined' && (window.location.port === '3000' || window.location.port === '5173'))
    ? 'http://127.0.0.1:5000'
    : '';

function App() {
  const [currentPage, setCurrentPage] = useState('upload'); // 'upload' | 'dashboard' | 'members' | 'details' | 'simulator' | 'analytics'
  const [selectedMemberId, setSelectedMemberId] = useState(null);
  const [selectedMemberDetail, setSelectedMemberDetail] = useState(null);
  const [members, setMembers] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [apiOnline, setApiOnline] = useState(false);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [appError, setAppError] = useState('');

  const fetchInitialData = useCallback(async () => {
    setLoading(true);
    setAppError('');
    try {
      // 1. Check API health
      const healthRes = await fetch(`${API_URL}/health`);
      const healthData = await healthRes.json();
      
      if (healthData.status === 'ok') {
        setApiOnline(true);
        
        // 2. Fetch members list
        const membersRes = await fetch(`${API_URL}/members`);
        const membersData = await membersRes.json();
        setMembers(membersData);

        // 3. Fetch analytics
        const analyticsRes = await fetch(`${API_URL}/analytics`);
        const analyticsData = await analyticsRes.json();
        setAnalytics(analyticsData);
      } else {
        throw new Error('API offline');
      }
    } catch (e) {
      console.warn('Flask API server offline.', e);
      setApiOnline(false);
      setMembers([]);
      setAnalytics(null);
      setAppError(`Flask API server (${API_URL}) is offline or connecting. Ensure the backend is running, then upload a dataset.`);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchMemberDetail = useCallback(async (memberId) => {
    setDetailLoading(true);
    try {
      if (apiOnline) {
        const res = await fetch(`${API_URL}/member/${memberId}`);
        const data = await res.json();
        setSelectedMemberDetail(data);
      } else {
        throw new Error('Flask API is offline.');
      }
    } catch (e) {
      console.error('Error fetching member details:', e);
      setSelectedMemberDetail(null);
      setAppError('Could not load member details from Flask. Confirm the backend is running and the dataset has been analyzed.');
    } finally {
      setDetailLoading(false);
    }
  }, [apiOnline]);

  // Fetch all data from API on mount and check URL query param
  useEffect(() => {
    fetchInitialData();
    const params = new URLSearchParams(window.location.search);
    const memberParam = params.get('member');
    if (memberParam) {
      setSelectedMemberId(memberParam);
      setCurrentPage('simulator');
    }
  }, [fetchInitialData]);

  // Fetch detailed information when a member is selected
  useEffect(() => {
    if (selectedMemberId) {
      fetchMemberDetail(selectedMemberId);
    }
  }, [selectedMemberId, fetchMemberDetail]);

  // Navigation handlers: Clicking a member row navigates directly to Retention Advisor
  const handleSelectMember = (memberId) => {
    setSelectedMemberId(memberId);
    setCurrentPage('simulator');
    if (memberId) {
      const newUrl = `${window.location.pathname}?member=${memberId}`;
      window.history.pushState({ memberId }, '', newUrl);
    } else {
      window.history.pushState({}, '', window.location.pathname);
    }
  };

  const handleBackToMembers = () => {
    setCurrentPage('members');
    setSelectedMemberId(null);
    setSelectedMemberDetail(null);
    window.history.pushState({}, '', window.location.pathname);
  };

  // High risk members for Dashboard Table
  const highRiskMembers = members.filter(m => m.risk_level === 'High' || m.risk_level === 'Critical').slice(0, 5);

  return (
    <div className="flex h-screen overflow-hidden bg-[#0b0f19] text-[#f3f4f6] antialiased font-sans select-none">
      {/* Sidebar Navigation matching exact dark design */}
      <aside className="w-72 h-screen sticky top-0 bg-[#070a13] border-r border-white/10 flex flex-col justify-between p-6 flex-shrink-0 text-white select-none overflow-y-auto">
        <div className="flex flex-col gap-8">
          {/* Logo Area */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-teal-500 flex items-center justify-center text-white shadow-lg shadow-indigo-500/30 border border-indigo-400/20">
              <HeartHandshake size={22} className="animate-pulse" />
            </div>
            <div className="flex flex-col">
              <span className="font-bold text-lg leading-tight tracking-wide font-heading bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">CareShield</span>
              <span className="text-[10px] text-slate-400 font-bold tracking-widest uppercase">ADVISOR SUITE</span>
            </div>
          </div>

          {/* Navigation Items */}
          <nav className="flex flex-col gap-2">
            <button 
              onClick={() => { setCurrentPage('upload'); setSelectedMemberId(null); }}
              className={`flex items-center gap-3.5 px-4 py-3 rounded-xl text-sm font-semibold transition-all duration-200 cursor-pointer ${currentPage === 'upload' ? 'bg-[#6366f1] text-white shadow-lg shadow-indigo-600/30' : 'text-slate-400 hover:bg-white/5 hover:text-white'}`}
            >
              <UploadCloud size={18} />
              <span>Dataset Upload</span>
            </button>
            
            <button 
              onClick={() => { setCurrentPage('dashboard'); setSelectedMemberId(null); }}
              className={`flex items-center gap-3.5 px-4 py-3 rounded-xl text-sm font-semibold transition-all duration-200 cursor-pointer ${currentPage === 'dashboard' ? 'bg-[#6366f1] text-white shadow-lg shadow-indigo-600/30' : 'text-slate-400 hover:bg-white/5 hover:text-white'}`}
            >
              <LayoutDashboard size={18} />
              <span>Strategic Dashboard</span>
            </button>

            <button 
              onClick={() => { setCurrentPage('members'); setSelectedMemberId(null); }}
              className={`flex items-center gap-3.5 px-4 py-3 rounded-xl text-sm font-semibold transition-all duration-200 cursor-pointer ${currentPage === 'members' || currentPage === 'details' ? 'bg-[#6366f1] text-white shadow-lg shadow-indigo-600/30' : 'text-slate-400 hover:bg-white/5 hover:text-white'}`}
            >
              <Users size={18} />
              <span>Member Details</span>
            </button>

            <button 
              onClick={() => { setCurrentPage('simulator'); setSelectedMemberId(null); }}
              className={`flex items-center gap-3.5 px-4 py-3 rounded-xl text-sm font-semibold transition-all duration-200 cursor-pointer ${currentPage === 'simulator' ? 'bg-[#6366f1] text-white shadow-lg shadow-indigo-600/30' : 'text-slate-400 hover:bg-white/5 hover:text-white'}`}
            >
              <Sliders size={18} />
              <span>Retention Advisor</span>
            </button>
          </nav>
        </div>

        {/* Sidebar Footer Cards */}
        <div className="flex flex-col gap-6">
          {/* Strategic Card matching exact screenshot */}
          <div className="bg-gradient-to-br from-indigo-950/40 to-teal-950/20 border border-indigo-500/20 rounded-xl p-4 flex flex-col gap-2.5 shadow-xl">
            <div className="flex items-center gap-2 text-white text-xs font-bold font-heading">
              <ShieldCheck size={14} className="text-teal-400" />
              <span>Strategic Goal</span>
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Retention is a core health plan priority. Prevent revenue loss, maintain care continuity, and elevate quality performance measures (HEDIS/Star ratings).
            </p>
          </div>
        </div>
      </aside>

      {/* Main Content Pane */}
      <main className="flex-1 flex flex-col min-w-0 h-screen overflow-y-auto bg-[#0b0f19]">
        {/* Top Main Header matching screenshot */}
        <header className="bg-[#0b0f19] border-b border-white/10 py-5 px-8 flex justify-between items-center flex-shrink-0">
          <div className="flex flex-col">
            <h1 className="text-2xl font-extrabold tracking-tight font-heading bg-gradient-to-r from-white via-slate-100 to-indigo-200 bg-clip-text text-transparent">
              Member Churn Prediction & Retention Advisor
            </h1>
            <p className="text-xs text-slate-400 mt-0.5 font-medium">
              Health Plan Retention Decision Support
            </p>
          </div>

          <div className="flex items-center gap-3">
            {/* Dark status badge matching screenshot */}
            <div className="flex items-center gap-2 px-3 py-1.5 bg-[#111827] border border-slate-800 rounded-full text-xs font-medium text-slate-300 shadow-inner">
              <span className={`w-2 h-2 rounded-full ${apiOnline ? 'bg-emerald-400 shadow-sm shadow-emerald-400/80 animate-pulse' : 'bg-rose-500'}`} />
              <span>{apiOnline ? 'Live API Online' : 'Connecting...'}</span>
            </div>

            <button
              onClick={() => setCurrentPage('upload')}
              className="flex items-center gap-1.5 px-4 py-2 bg-[#6366f1] hover:bg-[#4f46e5] text-white rounded-xl text-xs font-bold transition-all shadow-md shadow-indigo-600/30 cursor-pointer"
            >
              <UploadCloud size={14} />
              <span>Upload Dataset</span>
            </button>

            <button 
              onClick={fetchInitialData}
              className="flex items-center gap-1.5 px-3 py-2 border border-slate-700 bg-slate-900/60 rounded-xl text-xs font-bold text-slate-300 hover:bg-slate-800 transition-colors cursor-pointer"
            >
              <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
              <span>Sync API</span>
            </button>
          </div>
        </header>

        {/* Pages Content View */}
        <div className="flex-1 p-8">
          {loading ? (
            <div className="h-96 flex flex-col justify-center items-center gap-3 text-slate-400">
              <div className="w-10 h-10 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin" />
              <span className="text-sm font-semibold text-slate-400">Loading dataset and ML models cache...</span>
            </div>
          ) : (
            <>
              {appError && (
                <div className="mb-6 bg-rose-950/40 border border-rose-500/30 text-rose-300 rounded-xl p-4 text-sm font-semibold">
                  {appError}
                </div>
              )}

              {/* TAB 1: DATASET UPLOAD */}
              {currentPage === 'upload' && (
                <DatasetUpload 
                  onUploadSuccess={(data) => {
                    setMembers(data.members);
                    setAnalytics(data.summary);
                    setApiOnline(true);
                    setAppError('');
                    setCurrentPage('dashboard');
                  }}
                  onNavigateToDashboard={() => setCurrentPage('dashboard')}
                />
              )}

              {/* TAB 2: DASHBOARD */}
              {currentPage === 'dashboard' && analytics && (
                <div className="flex flex-col gap-6">
                  {/* Intro Banner */}
                  <div className="bg-[#111827]/80 border border-slate-800 rounded-2xl p-6 flex flex-col gap-2">
                    <h3 className="text-lg font-bold text-white font-heading">Strategic Retention Dashboard</h3>
                    <p className="text-xs text-slate-400">
                      High-level clinical and business health plan metrics. Target high-risk cohorts, monitor trends across tenure stages, and identify global churn drivers to guide outreach campaigns.
                    </p>
                  </div>

                  {/* KPI Cards Component */}
                  <KPICards 
                    total={analytics.total_members}
                    highRisk={(analytics.risk_distribution.High || 0) + (analytics.risk_distribution.Critical || 0)}
                    avgRisk={analytics.average_risk}
                    opportunities={(analytics.risk_distribution.Medium || 0) + (analytics.risk_distribution.High || 0) + (analytics.risk_distribution.Critical || 0)}
                  />

                  {/* Charts Grid */}
                  <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 mb-4">
                    <section className="bg-[#111827]/80 rounded-2xl border border-slate-800 p-6 flex flex-col gap-4 shadow-xl">
                      <div>
                        <h3 className="text-sm font-bold text-white uppercase tracking-wide font-heading">Risk Distribution</h3>
                        <p className="text-xs text-slate-400">Count of membership distributed across churn risk bands</p>
                      </div>
                      <div className="mt-2">
                        <RiskDistributionChart distribution={analytics.risk_distribution} />
                      </div>
                    </section>

                    <section className="bg-[#111827]/80 rounded-2xl border border-slate-800 p-6 flex flex-col gap-4 shadow-xl">
                      <div>
                        <h3 className="text-sm font-bold text-white uppercase tracking-wide font-heading">Risk Trend by Tenure</h3>
                        <p className="text-xs text-slate-400">Average churn risk probability plotted against member tenure cohorts</p>
                      </div>
                      <div className="mt-2">
                        <RiskTrendChart trend={analytics.tenure_risk_trend} />
                      </div>
                    </section>
                  </div>

                  {/* Top Drivers Chart */}
                  <section className="bg-[#111827]/80 rounded-2xl border border-slate-800 p-6 flex flex-col gap-4 mb-4 shadow-xl">
                    <div>
                      <h3 className="text-sm font-bold text-white uppercase tracking-wide font-heading">Top Factors Influencing Member Retention</h3>
                      <p className="text-xs text-slate-400 font-medium">Key factors that have the highest impact on overall member risk across the dataset</p>
                    </div>
                    <div className="mt-2">
                      <TopDriversChart drivers={analytics.top_risk_drivers} />
                    </div>
                  </section>

                  {/* High Risk Members Priority Table */}
                  <section className="bg-[#111827]/80 rounded-2xl border border-slate-800 overflow-hidden shadow-xl">
                    <div className="p-5 border-b border-slate-800 bg-slate-900/50 flex justify-between items-center">
                      <div>
                        <h3 className="text-sm font-bold text-white uppercase tracking-wide font-heading">High-Risk Members Priority Action List</h3>
                        <p className="text-xs text-slate-400">Members with High or Critical Churn Risk needing immediate coordination</p>
                      </div>
                      <button 
                        onClick={() => setCurrentPage('members')}
                        className="text-xs font-bold text-indigo-400 hover:text-indigo-300 transition-colors"
                      >
                        View All Members →
                      </button>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="min-w-full divide-y divide-slate-800">
                        <thead className="bg-slate-900/80">
                          <tr>
                            <th className="px-6 py-3.5 text-left text-xs font-bold text-slate-400 uppercase tracking-wider">Member ID</th>
                            <th className="px-6 py-3.5 text-left text-xs font-bold text-slate-400 uppercase tracking-wider">Risk Score</th>
                            <th className="px-6 py-3.5 text-left text-xs font-bold text-slate-400 uppercase tracking-wider">Risk Level</th>
                            <th className="px-6 py-3.5 text-left text-xs font-bold text-slate-400 uppercase tracking-wider">Primary Driver</th>
                            <th className="px-6 py-3.5 text-left text-xs font-bold text-slate-400 uppercase tracking-wider">Recommended Action</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/60 bg-[#111827]/60">
                          {highRiskMembers.map((m) => (
                            <tr 
                              key={m.member_id}
                              onClick={() => handleSelectMember(m.member_id)}
                              className="hover:bg-slate-800/60 cursor-pointer transition-colors duration-150 group"
                            >
                              <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-indigo-400 group-hover:text-indigo-300">
                                {m.member_id}
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-white">
                                {(m.churn_probability * 100).toFixed(1)}%
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm">
                                <RiskBadge level={m.risk_level} />
                              </td>
                              <td className="px-6 py-4 text-sm text-slate-300 truncate max-w-[220px]">
                                {m.primary_driver}
                              </td>
                              <td className="px-6 py-4 text-sm text-slate-300 font-medium">
                                {m.recommended_action}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </section>
                </div>
              )}

              {/* TAB 3: MEMBERS REGISTRY */}
              {currentPage === 'members' && (
                <div className="flex flex-col gap-6">
                  <div>
                    <h2 className="text-lg font-bold text-white tracking-tight font-heading uppercase">
                      Members Registry
                    </h2>
                    <p className="text-xs text-slate-400">
                      Search and segment plan members by risk, coverage types, or primary drivers
                    </p>
                  </div>
                  {/* Reusable MemberTable component */}
                  <MemberTable members={members} onSelectMember={handleSelectMember} />
                </div>
              )}

              {/* TAB 4: RETENTION ADVISOR */}
              {currentPage === 'simulator' && (
                <RetentionAdvisor 
                  members={members}
                  selectedMemberId={selectedMemberId}
                  onSelectMember={handleSelectMember}
                  API_URL={API_URL}
                />
              )}

              {/* MEMBER DETAILS SPECIFIC PROFILE VIEW */}
              {currentPage === 'details' && (
                <div className="flex flex-col gap-6">
                  {/* Breadcrumb Back link */}
                  <div className="flex items-center gap-2">
                    <button 
                      onClick={handleBackToMembers}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 border border-slate-700 bg-slate-900/60 rounded-xl text-xs font-bold text-slate-300 hover:bg-slate-800 transition-colors"
                    >
                      <ArrowLeft size={14} />
                      <span>Back to Members</span>
                    </button>
                    <span className="text-xs text-slate-500 font-medium">/ Member Detail Profile</span>
                  </div>

                  {detailLoading || !selectedMemberDetail ? (
                    <div className="h-96 flex flex-col justify-center items-center gap-3">
                      <div className="w-10 h-10 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                      <span className="text-sm font-semibold text-slate-400">Calculating risk drivers and recommendations...</span>
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                      {/* Left Side: Member Summary & Gauges */}
                      <div className="lg:col-span-1 flex flex-col gap-6">
                        {/* Member Bio Card */}
                        <div className="bg-[#111827]/80 border border-slate-800 rounded-2xl p-6 flex flex-col gap-5 shadow-xl">
                          <div>
                            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Plan Member ID</span>
                            <h3 className="text-xl font-extrabold text-white tracking-tight font-heading">{selectedMemberDetail.member_info.PatientID}</h3>
                          </div>
                          
                          <div className="grid grid-cols-2 gap-4 border-t border-b border-slate-800 py-4 text-sm">
                            <div>
                              <span className="text-[10px] text-slate-400 uppercase block font-semibold">Insurance Plan</span>
                              <strong className="text-white font-bold">{selectedMemberDetail.member_info.Insurance_Type}</strong>
                            </div>
                            <div>
                              <span className="text-[10px] text-slate-400 uppercase block font-semibold">Membership Tenure</span>
                              <strong className="text-white font-bold">{selectedMemberDetail.member_info.Tenure_Months} months</strong>
                            </div>
                            <div>
                              <span className="text-[10px] text-slate-400 uppercase block font-semibold">Primary Specialty</span>
                              <strong className="text-white font-bold truncate block" title={selectedMemberDetail.member_info.Specialty}>
                                {selectedMemberDetail.member_info.Specialty}
                              </strong>
                            </div>
                            <div>
                              <span className="text-[10px] text-slate-400 uppercase block font-semibold">Age / Gender</span>
                              <strong className="text-white font-bold">
                                {selectedMemberDetail.member_info.Age} / {selectedMemberDetail.member_info.Gender}
                              </strong>
                            </div>
                          </div>

                          <div className="flex flex-col gap-1 text-xs">
                            <span className="text-[10px] text-slate-400 uppercase block font-semibold">Last Recorded Visit</span>
                            <span className="text-slate-300 font-semibold">
                              {selectedMemberDetail.member_info.Days_Since_Last_Visit} days ago ({selectedMemberDetail.member_info.Last_Interaction_Date})
                            </span>
                          </div>
                        </div>

                        {/* Large Churn Risk Gauge Card */}
                        <div className="bg-[#111827]/80 border border-slate-800 rounded-2xl p-6 flex flex-col items-center justify-center text-center gap-4 relative overflow-hidden shadow-xl">
                          {/* Accent Color Band */}
                          <div className={`absolute top-0 left-0 w-full h-1.5 ${
                            selectedMemberDetail.risk_level === 'Critical' || selectedMemberDetail.risk_level === 'High' ? 'bg-rose-500' : 
                            (selectedMemberDetail.risk_level === 'Medium' ? 'bg-amber-500' : 'bg-emerald-500')
                          }`} />

                          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">Estimated Churn Risk</span>
                          <div className="relative flex items-center justify-center">
                            <span className={`text-6xl font-black font-heading ${
                              selectedMemberDetail.risk_level === 'Critical' || selectedMemberDetail.risk_level === 'High' ? 'text-rose-400' : 
                              (selectedMemberDetail.risk_level === 'Medium' ? 'text-amber-400' : 'text-emerald-400')
                            }`}>
                              {(selectedMemberDetail.churn_probability * 100).toFixed(0)}%
                            </span>
                          </div>
                          
                          <RiskBadge level={selectedMemberDetail.risk_level} />
                          <p className="text-xs text-slate-400 leading-relaxed px-4">
                            Action triggers are enabled based on risk priority tier.
                          </p>
                        </div>
                      </div>

                      {/* Right Side: Drivers and Interventions */}
                      <div className="lg:col-span-2 flex flex-col gap-6">
                        {/* Why is member high risk? */}
                        <section className="bg-[#111827]/80 border border-slate-800 rounded-2xl p-6 flex flex-col gap-4 shadow-xl">
                          <div>
                            <h3 className="text-sm font-bold text-white uppercase tracking-wide font-heading">Why is this member at risk?</h3>
                            <p className="text-xs text-slate-400">Member factors pushing risk towards higher risk (+ Red) or retention (- Green)</p>
                          </div>
                          
                          {/* Drivers List */}
                          <div className="flex flex-col mt-2">
                            {selectedMemberDetail.drivers.map((driver, idx) => (
                              <DriverBar 
                                key={idx}
                                label={driver.label}
                                value={driver.value}
                                shapValue={driver.shap_value}
                                maxShap={Math.max(...selectedMemberDetail.drivers.map(d => Math.abs(d.shap_value)))}
                              />
                            ))}
                          </div>

                          {/* Evidence list */}
                          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 flex flex-col gap-3 mt-2">
                            <span className="text-[11px] font-bold text-indigo-300 uppercase tracking-wide">Observable Plan Evidence & Signals</span>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-2 text-xs">
                              <div className="flex justify-between border-b border-slate-800/80 py-1">
                                <span className="text-slate-400">Billing Issues Reported:</span>
                                <strong className="text-slate-200 font-bold">{selectedMemberDetail.member_info.Billing_Issues === 1 ? 'Yes' : 'None'}</strong>
                              </div>
                              <div className="flex justify-between border-b border-slate-800/80 py-1">
                                <span className="text-slate-400">Patient Portal Enrollment:</span>
                                <strong className="text-slate-200 font-bold">{selectedMemberDetail.member_info.Portal_Usage === 1 ? 'Active' : 'Inactive'}</strong>
                              </div>
                              <div className="flex justify-between border-b border-slate-800/80 py-1">
                                <span className="text-slate-400">Average Out of Pocket Spend:</span>
                                <strong className="text-slate-200 font-bold">${selectedMemberDetail.member_info.Avg_Out_Of_Pocket_Cost.toLocaleString()}</strong>
                              </div>
                              <div className="flex justify-between border-b border-slate-800/80 py-1">
                                <span className="text-slate-400">Missed Appointments Count:</span>
                                <strong className="text-slate-200 font-bold">{selectedMemberDetail.member_info.Missed_Appointments} / 5</strong>
                              </div>
                              <div className="flex justify-between border-b border-slate-800/80 py-1">
                                <span className="text-slate-400">Annual Visits Count:</span>
                                <strong className="text-slate-200 font-bold">{selectedMemberDetail.member_info.Visits_Last_Year} visits</strong>
                              </div>
                              <div className="flex justify-between border-b border-slate-800/80 py-1">
                                <span className="text-slate-400">Overall Plan Satisfaction:</span>
                                <strong className="text-slate-200 font-bold">{selectedMemberDetail.member_info.Overall_Satisfaction.toFixed(1)} / 5.0</strong>
                              </div>
                            </div>
                          </div>
                        </section>

                        {/* Retention Advisor Suggestions */}
                        <section className="flex flex-col gap-4">
                          <div>
                            <h3 className="text-sm font-bold text-white uppercase tracking-wide font-heading">Retention Advisor Suggestions (Rule Engine)</h3>
                            <p className="text-xs text-slate-400">Personalized outreach suggestions generated by mapping feature evidence to strategic plan categories</p>
                          </div>
                          
                          <div className="flex flex-col gap-4">
                            {selectedMemberDetail.recommendations.map((rec, idx) => (
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
              )}
            </>
          )}
        </div>

        {/* Global Footer Disclaimer */}
        <footer className="bg-[#070a13] border-t border-white/10 py-4 px-8 text-center text-[11px] text-slate-500 font-medium tracking-wide flex-shrink-0">
          Risk scores are model estimates and do not guarantee disenrollment. Recommendations are decision-support suggestions and require human review.
        </footer>
      </main>
    </div>
  );
}

export default App;
