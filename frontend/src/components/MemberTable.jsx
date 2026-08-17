import React, { useState, useMemo } from 'react';
import RiskBadge from './RiskBadge';

/**
 * MemberTable
 * Renders a searchable and filterable list of members in dark theme.
 * Supports sorting on Churn Probability/Tenure/ID.
 */
const MemberTable = ({ members, onSelectMember }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [riskFilter, setRiskFilter] = useState('');
  const [planFilter, setPlanFilter] = useState('');
  const [driverFilter, setDriverFilter] = useState('');
  const [sortField, setSortField] = useState('churn_probability');
  const [sortDirection, setSortDirection] = useState('desc');

  // Page index for pagination (chunking large sets of 2,000 members for performance)
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 20;

  // Compute unique plans and primary driver labels for drop-downs
  const uniquePlans = useMemo(() => {
    const plans = members.map(m => m.plan).filter(Boolean);
    return [...new Set(plans)].sort();
  }, [members]);

  const uniqueDrivers = useMemo(() => {
    const drivers = members.map(m => {
      if (m.primary_driver && m.primary_driver.includes(' (')) {
        return m.primary_driver.split(' (')[0];
      }
      return m.primary_driver;
    }).filter(Boolean);
    return [...new Set(drivers)].sort();
  }, [members]);

  // Filter and Sort members list
  const processedMembers = useMemo(() => {
    let result = [...members];

    // Search filter (Member ID)
    if (searchTerm.trim()) {
      const term = searchTerm.toLowerCase();
      result = result.filter(m => m.member_id.toLowerCase().includes(term));
    }

    // Risk level filter
    if (riskFilter) {
      result = result.filter(m => m.risk_level === riskFilter);
    }

    // Plan filter
    if (planFilter) {
      result = result.filter(m => m.plan === planFilter);
    }

    // Driver filter
    if (driverFilter) {
      result = result.filter(m => m.primary_driver && m.primary_driver.startsWith(driverFilter));
    }

    // Sort
    result.sort((a, b) => {
      let aVal = a[sortField];
      let bVal = b[sortField];

      if (typeof aVal === 'string') {
        aVal = aVal.toLowerCase();
        bVal = bVal.toLowerCase();
      }

      if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1;
      if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1;
      return 0;
    });

    return result;
  }, [members, searchTerm, riskFilter, planFilter, driverFilter, sortField, sortDirection]);

  // Pagination slice
  const paginatedMembers = useMemo(() => {
    const startIdx = (currentPage - 1) * itemsPerPage;
    return processedMembers.slice(startIdx, startIdx + itemsPerPage);
  }, [processedMembers, currentPage]);

  const totalPages = Math.ceil(processedMembers.length / itemsPerPage) || 1;

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
    setCurrentPage(1); // Reset page on sort
  };

  return (
    <div className="bg-[#111827]/80 backdrop-blur-md rounded-2xl border border-slate-800 overflow-hidden shadow-xl text-white">
      {/* Filters Area */}
      <div className="p-5 border-b border-slate-800 bg-slate-900/60 flex flex-col md:flex-row gap-4 items-center justify-between">
        {/* Search */}
        <div className="w-full md:w-80">
          <label htmlFor="search-input" className="sr-only">Search by Member ID</label>
          <input 
            id="search-input"
            type="text" 
            placeholder="Search Member ID..." 
            value={searchTerm}
            onChange={(e) => { setSearchTerm(e.target.value); setCurrentPage(1); }}
            className="w-full border border-slate-700 rounded-xl px-4 py-2.5 text-xs bg-slate-900 text-slate-200 outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all placeholder-slate-500"
          />
        </div>

        {/* Dropdown Filters */}
        <div className="flex flex-wrap items-center gap-3 w-full md:w-auto justify-end">
          {/* Risk Filter */}
          <select 
            value={riskFilter} 
            onChange={(e) => { setRiskFilter(e.target.value); setCurrentPage(1); }}
            className="border border-slate-700 rounded-xl px-3 py-2.5 text-xs font-semibold bg-slate-900 text-slate-300 outline-none focus:border-indigo-500"
          >
            <option value="">All Risk Levels</option>
            <option value="Low">Low Risk</option>
            <option value="Medium">Medium Risk</option>
            <option value="High">High Risk</option>
            <option value="Critical">Critical Risk</option>
          </select>

          {/* Plan Filter */}
          <select 
            value={planFilter} 
            onChange={(e) => { setPlanFilter(e.target.value); setCurrentPage(1); }}
            className="border border-slate-700 rounded-xl px-3 py-2.5 text-xs font-semibold bg-slate-900 text-slate-300 outline-none focus:border-indigo-500"
          >
            <option value="">All Plans</option>
            {uniquePlans.map(plan => (
              <option key={plan} value={plan}>{plan}</option>
            ))}
          </select>

          {/* Driver Filter */}
          <select 
            value={driverFilter} 
            onChange={(e) => { setDriverFilter(e.target.value); setCurrentPage(1); }}
            className="border border-slate-700 rounded-xl px-3 py-2.5 text-xs font-semibold bg-slate-900 text-slate-300 outline-none focus:border-indigo-500 max-w-[200px]"
          >
            <option value="">All Primary Drivers</option>
            {uniqueDrivers.map(driver => (
              <option key={driver} value={driver}>{driver}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Table Area */}
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-slate-800">
          <thead className="bg-slate-900/80">
            <tr>
              <th 
                className="px-6 py-3.5 text-left text-xs font-bold text-slate-400 uppercase tracking-wider cursor-pointer hover:text-white transition-colors"
                onClick={() => handleSort('member_id')}
              >
                Member ID {sortField === 'member_id' && (sortDirection === 'asc' ? '▲' : '▼')}
              </th>
              <th className="px-6 py-3.5 text-left text-xs font-bold text-slate-400 uppercase tracking-wider">Plan</th>
              <th 
                className="px-6 py-3.5 text-left text-xs font-bold text-slate-400 uppercase tracking-wider cursor-pointer hover:text-white transition-colors"
                onClick={() => handleSort('tenure')}
              >
                Tenure (Mo) {sortField === 'tenure' && (sortDirection === 'asc' ? '▲' : '▼')}
              </th>
              <th 
                className="px-6 py-3.5 text-left text-xs font-bold text-slate-400 uppercase tracking-wider cursor-pointer hover:text-white transition-colors"
                onClick={() => handleSort('churn_probability')}
              >
                Risk Score {sortField === 'churn_probability' && (sortDirection === 'asc' ? '▲' : '▼')}
              </th>
              <th className="px-6 py-3.5 text-left text-xs font-bold text-slate-400 uppercase tracking-wider">Risk Level</th>
              <th className="px-6 py-3.5 text-left text-xs font-bold text-slate-400 uppercase tracking-wider">Primary Driver</th>
              <th className="px-6 py-3.5 text-left text-xs font-bold text-slate-400 uppercase tracking-wider">Recommended Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 bg-[#111827]/60">
            {paginatedMembers.length > 0 ? (
              paginatedMembers.map((member) => (
                <tr 
                  key={member.member_id}
                  onClick={() => onSelectMember(member.member_id)}
                  className="hover:bg-slate-800/60 cursor-pointer transition-colors duration-150 group"
                >
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-indigo-400 group-hover:text-indigo-300">
                    {member.member_id}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-300">
                    {member.plan}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-300">
                    {member.tenure} mo
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-white">
                    {(member.churn_probability * 100).toFixed(1)}%
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">
                    <RiskBadge level={member.risk_level} />
                  </td>
                  <td className="px-6 py-4 text-sm text-slate-300 truncate max-w-[220px]">
                    {member.primary_driver}
                  </td>
                  <td className="px-6 py-4 text-sm text-slate-300 font-medium">
                    {member.recommended_action}
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="7" className="px-6 py-8 text-center text-sm text-slate-500">
                  No member records found matching current filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Footer */}
      <div className="px-6 py-4 border-t border-slate-800 bg-slate-900/60 flex items-center justify-between text-xs text-slate-400">
        <span>
          Showing <strong className="text-white font-bold">{processedMembers.length > 0 ? (currentPage - 1) * itemsPerPage + 1 : 0}</strong> to <strong className="text-white font-bold">{Math.min(currentPage * itemsPerPage, processedMembers.length)}</strong> of <strong className="text-white font-bold">{processedMembers.length.toLocaleString()}</strong> members
        </span>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
            disabled={currentPage === 1}
            className="px-3 py-1.5 border border-slate-700 bg-slate-900 rounded-lg text-xs font-semibold disabled:opacity-40 hover:bg-slate-800 text-slate-300 transition-colors"
          >
            Previous
          </button>
          <span className="font-semibold text-slate-300">Page {currentPage} of {totalPages}</span>
          <button
            onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
            disabled={currentPage === totalPages}
            className="px-3 py-1.5 border border-slate-700 bg-slate-900 rounded-lg text-xs font-semibold disabled:opacity-40 hover:bg-slate-800 text-slate-300 transition-colors"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
};

export default MemberTable;
