import React, { useState } from 'react';
import { UploadCloud, FileSpreadsheet, CheckCircle2, AlertCircle, Loader2, ArrowRight, X, Sparkles, Database } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';

const DatasetUpload = ({ onUploadSuccess, onNavigateToDashboard }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [status, setStatus] = useState('idle'); // 'idle' | 'uploading' | 'processing' | 'success' | 'error'
  const [errorMessage, setErrorMessage] = useState('');
  const [progressStep, setProgressStep] = useState(0);
  const [summaryData, setSummaryData] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      validateAndSetFile(file);
    }
  };

  const validateAndSetFile = (file) => {
    const validExtensions = ['.csv', '.xlsx', '.xls'];
    const fileName = file.name.toLowerCase();
    const isValid = validExtensions.some(ext => fileName.endsWith(ext));

    if (!isValid) {
      setStatus('error');
      setErrorMessage('Unsupported file format. Please upload a .csv, .xlsx, or .xls file.');
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
    setStatus('idle');
    setErrorMessage('');
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setStatus('idle');
    setErrorMessage('');
    setSummaryData(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setStatus('error');
      setErrorMessage('Please select a dataset.');
      return;
    }

    setStatus('uploading');
    setProgressStep(1);
    setErrorMessage('');
    setSummaryData(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch(`${API_URL}/predict-file`, {
        method: 'POST',
        body: formData
      });

      setStatus('processing');
      setProgressStep(3);

      if (!res.ok) {
        let errorText = 'Upload failed.';
        try {
          const errData = await res.json();
          errorText = errData.error || errData.message || errorText;
        } catch {
          errorText = `Server returned status ${res.status}`;
        }
        throw new Error(errorText);
      }

      const data = await res.json();
      
      if (data.status === 'success' || data.success) {
        setProgressStep(4);
        setStatus('success');
        setSummaryData(data.summary || {
          total_members: data.total_members,
          risk_distribution: data.risk_summary
        });

        if (onUploadSuccess) {
          onUploadSuccess(data);
        }
      } else {
        throw new Error(data.error || 'Dataset analysis failed on server.');
      }
    } catch (err) {
      console.error('Dataset upload error:', err);
      setStatus('error');
      setErrorMessage(err.message || `Upload failed. Confirm Flask API is accessible at ${API_URL}.`);
    }
  };

  const handleLoadDemo = async () => {
    setStatus('processing');
    setProgressStep(2);
    setErrorMessage('');
    try {
      const res = await fetch(`${API_URL}/load-demo`);
      if (!res.ok) throw new Error('Failed to load demo dataset from backend.');
      const data = await res.json();
      if (data.status === 'success' || data.success) {
        setProgressStep(4);
        setStatus('success');
        setSummaryData(data.summary || {
          total_members: data.total_members,
          risk_distribution: data.risk_summary
        });
        if (onUploadSuccess) onUploadSuccess(data);
      } else {
        throw new Error(data.error || 'Failed to load demo data.');
      }
    } catch (err) {
      setStatus('error');
      setErrorMessage(err.message);
    }
  };

  const statusText = (() => {
    if (status === 'uploading') return 'Uploading dataset...';
    if (status === 'processing') return selectedFile ? 'Analyzing dataset...' : 'Analyzing members...';
    if (status === 'success' && summaryData?.total_members) {
      return `Analysis completed for ${summaryData.total_members.toLocaleString()} members.`;
    }
    if (status === 'success') return 'Analysis completed.';
    if (status === 'error') return `Upload failed: ${errorMessage}`;
    return selectedFile ? `Selected file: ${selectedFile.name}` : 'Please select a dataset.';
  })();

  return (
    <div className="flex flex-col gap-6 max-w-4xl mx-auto text-white">
      {/* Intro Banner Card */}
      <div className="bg-[#111827]/80 backdrop-blur-md rounded-2xl border border-slate-800 p-6 flex flex-col gap-2 shadow-2xl">
        <h2 className="text-xl font-bold text-white tracking-tight font-heading">Upload Member Dataset</h2>
        <p className="text-sm text-slate-400 leading-relaxed">
          Upload a CSV or Excel file containing health plan member metrics to run predictive churn analysis, identify individual risk drivers, and compile rule-based retention advisor suggestions.
        </p>
      </div>

      {/* Main Upload Panel */}
      <div className="bg-[#111827]/80 backdrop-blur-md rounded-2xl border border-slate-800 p-8 flex flex-col gap-6 shadow-2xl">
        <div className="flex items-center gap-3 border-b border-slate-800 pb-4">
          <div className="w-10 h-10 rounded-xl bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold border border-indigo-500/30">
            <FileSpreadsheet size={22} />
          </div>
          <div>
            <h3 className="text-base font-bold text-white font-heading">Import Member Data</h3>
            <p className="text-xs text-slate-400">Supported Formats: .csv, .xlsx, .xls</p>
          </div>
        </div>

        {/* Drop Zone */}
        {!selectedFile ? (
          <div 
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-2xl p-12 flex flex-col items-center justify-center text-center gap-4 transition-all ${
              dragActive 
                ? 'border-indigo-500 bg-indigo-950/30' 
                : 'border-slate-700/80 bg-slate-900/40 hover:border-slate-600 hover:bg-slate-900/60'
            }`}
          >
            <div className="w-16 h-16 rounded-2xl bg-teal-500/10 text-teal-400 flex items-center justify-center border border-teal-500/20 shadow-lg shadow-teal-950/50">
              <UploadCloud size={32} />
            </div>
            <div>
              <p className="text-base font-semibold text-slate-200">Drag and drop your member CSV file here, or click to browse.</p>
              <p className="text-xs text-slate-400 mt-1">Accepts CSV, XLSX, and XLS format member datasets</p>
            </div>

            <label className="px-6 py-2.5 bg-[#0d9488] hover:bg-[#0f766e] text-white rounded-xl text-xs font-bold cursor-pointer transition-all shadow-md shadow-teal-950/50 inline-flex items-center gap-2 mt-2">
              <UploadCloud size={16} />
              <span>Browse File</span>
              <input
                type="file"
                accept=".csv,.xlsx,.xls,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel"
                onChange={handleFileChange}
                className="hidden" 
              />
            </label>
          </div>
        ) : (
          /* File Selected Row */
          <div className="bg-teal-950/20 border border-teal-500/30 rounded-xl p-5 flex items-center justify-between gap-4">
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-10 h-10 rounded-lg bg-teal-500/20 text-teal-400 flex items-center justify-center flex-shrink-0 border border-teal-500/30">
                <FileSpreadsheet size={22} />
              </div>
              <div className="flex flex-col min-w-0">
                <span className="text-sm font-bold text-white truncate">{selectedFile.name}</span>
                <span className="text-xs text-teal-300/80">{(selectedFile.size / 1024).toFixed(1)} KB • Ready to analyze</span>
              </div>
            </div>

            {status !== 'uploading' && status !== 'processing' && (
              <button 
                onClick={handleRemoveFile}
                className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
                title="Remove file"
              >
                <X size={18} />
              </button>
            )}
          </div>
        )}

        {/* Buttons & Actions */}
        <div className="flex flex-col gap-4">
          <button
            onClick={handleAnalyze}
            disabled={!selectedFile || status === 'uploading' || status === 'processing'}
            className="w-full py-4 bg-[#6366f1] hover:bg-[#4f46e5] disabled:opacity-40 text-white rounded-xl font-bold text-sm transition-all shadow-lg shadow-indigo-600/30 flex items-center justify-center gap-2 cursor-pointer"
          >
            {status === 'uploading' || status === 'processing' ? (
              <>
                <Loader2 size={18} className="animate-spin" />
                <span>Analyzing Dataset...</span>
              </>
            ) : (
              <>
                <Sparkles size={18} />
                <span>Analyze Dataset</span>
              </>
            )}
          </button>

          {!selectedFile && (
            <div className="flex items-center justify-between gap-4 pt-2">
              <span className="text-xs text-slate-500">Need sample data?</span>
              <button
                onClick={handleLoadDemo}
                className="inline-flex items-center gap-2 px-4 py-2 border border-teal-500/40 hover:bg-teal-950/30 text-teal-400 rounded-xl text-xs font-semibold transition-all cursor-pointer"
              >
                <Database size={14} />
                <span>Load Demo Dataset</span>
              </button>
            </div>
          )}

          {/* Progress / Status Steps */}
          {(status === 'uploading' || status === 'processing' || status === 'success') && (
            <div className="bg-slate-900/80 rounded-xl p-4 border border-slate-800 flex flex-col gap-2 text-xs">
              <p className="font-bold text-slate-200 mb-1">{statusText}</p>
              <div className="flex items-center gap-2">
                {progressStep >= 1 ? <CheckCircle2 size={14} className="text-emerald-400" /> : <Loader2 size={14} className="animate-spin text-indigo-400" />}
                <span className={progressStep >= 1 ? 'font-semibold text-slate-200' : 'text-slate-500'}>Uploading dataset file...</span>
              </div>
              <div className="flex items-center gap-2">
                {progressStep >= 3 ? <CheckCircle2 size={14} className="text-emerald-400" /> : <Loader2 size={14} className="animate-spin text-indigo-400" />}
                <span className={progressStep >= 3 ? 'font-semibold text-slate-200' : 'text-slate-500'}>Validating columns and cleaning values...</span>
              </div>
              <div className="flex items-center gap-2">
                {progressStep >= 3 ? <CheckCircle2 size={14} className="text-emerald-400" /> : <Loader2 size={14} className="animate-spin text-indigo-400" />}
                <span className={progressStep >= 3 ? 'font-semibold text-slate-200' : 'text-slate-500'}>Evaluating XGBoost churn predictions & SHAP drivers...</span>
              </div>
            </div>
          )}

          {/* Success Banner */}
          {status === 'success' && summaryData && (
            <div className="bg-emerald-950/30 border border-emerald-500/30 rounded-xl p-5 flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-emerald-400 font-bold text-sm">
                  <CheckCircle2 size={18} />
                  <span>Analysis Completed Successfully!</span>
                </div>
                <span className="text-xs bg-emerald-500/20 text-emerald-300 font-semibold px-3 py-1 rounded-full border border-emerald-500/30">
                  {summaryData.total_members ? summaryData.total_members.toLocaleString() : '2,000'} Members Analyzed
                </span>
              </div>

              <p className="text-xs text-slate-300">
                Member churn scores, risk level tiers, top risk drivers, and personalized retention suggestions have been calculated.
              </p>

              <button
                onClick={onNavigateToDashboard}
                className="self-start inline-flex items-center gap-2 px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs rounded-xl transition-all shadow-md shadow-emerald-950/50 mt-1 cursor-pointer"
              >
                <span>View Strategic Dashboard</span>
                <ArrowRight size={14} />
              </button>
            </div>
          )}

          {/* Error Banner */}
          {status === 'error' && (
            <div className="bg-rose-950/30 border border-rose-500/30 rounded-xl p-4 flex items-start gap-3 text-xs text-rose-300">
              <AlertCircle size={18} className="text-rose-400 flex-shrink-0 mt-0.5" />
              <div className="flex flex-col gap-1">
                <span className="font-bold text-rose-200">Upload Failed</span>
                <p>{statusText}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DatasetUpload;
