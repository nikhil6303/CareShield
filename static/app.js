// Global objects and charts
let rocChart = null;
let globalDriversChart = null;
let individualBreakdownChart = null;
let dashRiskDistChart = null;
let dashRiskTrendChart = null;
let dashDriversChart = null;
let detailShapChart = null;
const API_URL = (typeof window !== 'undefined' && window.location && window.location.protocol.startsWith('http')) 
    ? window.location.origin 
    : 'http://127.0.0.1:5000';
let isApiOnline = false;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    checkApiStatus();
    initTabs();
    initSliders();
    initDiagnostics();
    initSimulator();
    initUploadFlow();
});

// 1. Tab Navigation
function initTabs() {
    const navItems = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetTab = item.getAttribute('data-tab');
            
            navItems.forEach(nav => nav.classList.remove('active'));
            tabContents.forEach(tab => tab.classList.remove('active'));
            
            item.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
            
            // Re-render charts on tab switch to resolve sizing issues
            if (targetTab === 'diagnostics-tab') {
                setTimeout(() => {
                    if (rocChart) rocChart.resize();
                    if (globalDriversChart) globalDriversChart.resize();
                }, 50);
            } else if (targetTab === 'simulator-tab') {
                setTimeout(() => {
                    if (individualBreakdownChart) individualBreakdownChart.resize();
                }, 50);
            } else if (targetTab === 'dashboard-tab') {
                setTimeout(() => {
                    if (dashRiskDistChart) dashRiskDistChart.resize();
                    if (dashRiskTrendChart) dashRiskTrendChart.resize();
                    if (dashDriversChart) dashDriversChart.resize();
                }, 50);
            } else if (targetTab === 'member-details-tab') {
                setTimeout(() => {
                    if (detailShapChart) detailShapChart.resize();
                }, 50);
            }
        });
    });
}

// 2. Slider Value Sync
function initSliders() {
    const rangeInputs = document.querySelectorAll('input[type="range"]');
    rangeInputs.forEach(input => {
        const id = input.id.replace('input-', 'val-');
        const valEl = document.getElementById(id);
        if (valEl) {
            valEl.textContent = parseFloat(input.value).toFixed(1);
        }
        
        input.addEventListener('input', () => {
            if (valEl) {
                valEl.textContent = parseFloat(input.value).toFixed(1);
            }
            // Trigger dynamic prediction on slider input
            updatePrediction();
        });
    });

    // Also trigger on change for form elements
    const formElements = document.querySelectorAll('#simulator-form input, #simulator-form select');
    formElements.forEach(el => {
        if (el.type !== 'range') {
            el.addEventListener('change', updatePrediction);
        }
    });
    // Add real-time typing listeners for number inputs
    const numberInputs = document.querySelectorAll('#simulator-form input[type="number"]');
    numberInputs.forEach(el => {
        el.addEventListener('input', updatePrediction);
    });
}

// 3. Initialize Model Analytics & Diagnostics Tab
function initDiagnostics() {
    if (!document.getElementById('diag-lr-accuracy')) return;
    // Populate Logistic Regression metrics
    const lrMetrics = MODEL_DATA.logistic_regression.metrics;
    document.getElementById('diag-lr-accuracy').textContent = (lrMetrics.accuracy * 100).toFixed(1) + '%';
    document.getElementById('diag-lr-precision').textContent = lrMetrics.precision.toFixed(3);
    document.getElementById('diag-lr-recall').textContent = lrMetrics.recall.toFixed(3);
    document.getElementById('diag-lr-f1').textContent = lrMetrics.f1.toFixed(3);
    document.getElementById('diag-lr-auc').textContent = lrMetrics.auc.toFixed(3);
    document.getElementById('lr-cm-tn').textContent = lrMetrics.cm[0][0];
    document.getElementById('lr-cm-fp').textContent = lrMetrics.cm[0][1];
    document.getElementById('lr-cm-fn').textContent = lrMetrics.cm[1][0];
    document.getElementById('lr-cm-tp').textContent = lrMetrics.cm[1][1];

    // Populate Decision Tree metrics
    const dtMetrics = MODEL_DATA.decision_tree.metrics;
    document.getElementById('diag-dt-accuracy').textContent = (dtMetrics.accuracy * 100).toFixed(1) + '%';
    document.getElementById('diag-dt-precision').textContent = dtMetrics.precision.toFixed(3);
    document.getElementById('diag-dt-recall').textContent = dtMetrics.recall.toFixed(3);
    document.getElementById('diag-dt-f1').textContent = dtMetrics.f1.toFixed(3);
    document.getElementById('diag-dt-auc').textContent = dtMetrics.auc.toFixed(3);
    document.getElementById('dt-cm-tn').textContent = dtMetrics.cm[0][0];
    document.getElementById('dt-cm-fp').textContent = dtMetrics.cm[0][1];
    document.getElementById('dt-cm-fn').textContent = dtMetrics.cm[1][0];
    document.getElementById('dt-cm-tp').textContent = dtMetrics.cm[1][1];

    // Populate AdaBoost metrics
    if (MODEL_DATA.adaboost) {
        const adaMetrics = MODEL_DATA.adaboost.metrics;
        document.getElementById('diag-ada-accuracy').textContent = (adaMetrics.accuracy * 100).toFixed(1) + '%';
        document.getElementById('diag-ada-precision').textContent = adaMetrics.precision.toFixed(3);
        document.getElementById('diag-ada-recall').textContent = adaMetrics.recall.toFixed(3);
        document.getElementById('diag-ada-f1').textContent = adaMetrics.f1.toFixed(3);
        document.getElementById('diag-ada-auc').textContent = adaMetrics.auc.toFixed(3);
        document.getElementById('ada-cm-tn').textContent = adaMetrics.cm[0][0];
        document.getElementById('ada-cm-fp').textContent = adaMetrics.cm[0][1];
        document.getElementById('ada-cm-fn').textContent = adaMetrics.cm[1][0];
        document.getElementById('ada-cm-tp').textContent = adaMetrics.cm[1][1];
    }

    // Render ROC Curve Chart
    const ctxRoc = document.getElementById('roc-curve-chart').getContext('2d');
    const rocDatasets = [
        {
            label: `Logistic Regression (AUC: ${lrMetrics.auc.toFixed(3)})`,
            data: lrMetrics.roc_curve.fpr.map((fpr, i) => ({ x: fpr, y: lrMetrics.roc_curve.tpr[i] })),
            borderColor: '#6366f1', borderWidth: 2, fill: false, tension: 0.1, pointRadius: 2
        },
        {
            label: `Decision Tree (AUC: ${dtMetrics.auc.toFixed(3)})`,
            data: dtMetrics.roc_curve.fpr.map((fpr, i) => ({ x: fpr, y: dtMetrics.roc_curve.tpr[i] })),
            borderColor: '#0d9488', borderWidth: 2, fill: false, tension: 0.1, pointRadius: 2
        },
        {
            label: 'Random Guess',
            data: [{x: 0, y: 0}, {x: 1, y: 1}],
            borderColor: 'rgba(255,255,255,0.2)', borderWidth: 1,
            borderDash: [5, 5], fill: false, pointRadius: 0
        }
    ];
    if (MODEL_DATA.adaboost) {
        const adaMetrics = MODEL_DATA.adaboost.metrics;
        rocDatasets.splice(2, 0, {
            label: `AdaBoost (AUC: ${adaMetrics.auc.toFixed(3)})`,
            data: adaMetrics.roc_curve.fpr.map((fpr, i) => ({ x: fpr, y: adaMetrics.roc_curve.tpr[i] })),
            borderColor: '#f59e0b', borderWidth: 2, fill: false, tension: 0.1, pointRadius: 2
        });
    }
    rocChart = new Chart(ctxRoc, {
        type: 'line',
        data: { datasets: rocDatasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: { display: true, text: 'False Positive Rate (FPR)', color: '#9ca3af' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' }
                },
                y: {
                    title: { display: true, text: 'True Positive Rate (TPR)', color: '#9ca3af' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#f3f4f6' }
                }
            }
        }
    });

    // Render Global Feature Drivers Chart
    drawGlobalDrivers('coefs');
    
    document.getElementById('btn-coefs').addEventListener('click', (e) => {
        document.querySelectorAll('.tab-sub-btn').forEach(btn => btn.classList.remove('active'));
        e.target.classList.add('active');
        drawGlobalDrivers('coefs');
    });
    
    document.getElementById('btn-importances').addEventListener('click', (e) => {
        document.querySelectorAll('.tab-sub-btn').forEach(btn => btn.classList.remove('active'));
        e.target.classList.add('active');
        drawGlobalDrivers('importances');
    });

    const btnAda = document.getElementById('btn-ada-importances');
    if (btnAda) {
        btnAda.addEventListener('click', (e) => {
            document.querySelectorAll('.tab-sub-btn').forEach(btn => btn.classList.remove('active'));
            e.target.classList.add('active');
            drawGlobalDrivers('ada-importances');
        });
    }
}

function drawGlobalDrivers(type) {
    let dataLabel = '';
    let labels = [];
    let values = [];
    let colors = [];

    if (type === 'coefs') {
        dataLabel = 'Feature Impact Strength on Member Risk';
        const coefsObj = MODEL_DATA.logistic_regression.model.coefficients;
        // Sort features by coefficient magnitude
        const sortedCoefs = Object.entries(coefsObj).sort((a, b) => b[1] - a[1]);
        
        // Take top positive and top negative features to keep chart readable
        const topPositive = sortedCoefs.slice(0, 7);
        const topNegative = sortedCoefs.slice(-7);
        const selected = [...topPositive, ...topNegative];
        
        labels = selected.map(item => item[0]);
        values = selected.map(item => item[1]);
        colors = values.map(val => val > 0 ? 'rgba(244, 63, 94, 0.7)' : 'rgba(16, 185, 129, 0.7)');
    } else if (type === 'ada-importances') {
        dataLabel = 'AdaBoost Feature Importance';
        const importancesObj = MODEL_DATA.adaboost ? MODEL_DATA.adaboost.importances : {};
        const sortedImp = Object.entries(importancesObj)
            .sort((a, b) => b[1] - a[1])
            .filter(item => item[1] > 0.005);
        labels = sortedImp.map(item => item[0]);
        values = sortedImp.map(item => item[1]);
        colors = 'rgba(245, 158, 11, 0.7)'; // Amber
    } else {
        dataLabel = 'Decision Tree Feature Importance';
        const importancesObj = MODEL_DATA.decision_tree.importances;
        // Sort by importance
        const sortedImp = Object.entries(importancesObj)
            .sort((a, b) => b[1] - a[1])
            .filter(item => item[1] > 0.005); // Filter out tiny importances
            
        labels = sortedImp.map(item => item[0]);
        values = sortedImp.map(item => item[1]);
        colors = 'rgba(13, 148, 136, 0.7)'; // Teal
    }

    const ctxDrivers = document.getElementById('global-importance-chart').getContext('2d');
    if (globalDriversChart) {
        globalDriversChart.destroy();
    }
    
    globalDriversChart = new Chart(ctxDrivers, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: dataLabel,
                data: values,
                backgroundColor: colors,
                borderColor: Array.isArray(colors) ? colors.map(c => c.replace('0.7', '1')) : '#2dd4bf',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#9ca3af', font: { size: 10 } }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#f3f4f6' }
                }
            }
        }
    });
}

// 4. Initialize Member Simulator
function initSimulator() {
    // Generate initial prediction
    updatePrediction();
}

// Helper to calculate sigmoid
function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
}

// Core function to read form inputs, preprocess, predict, explain, and recommend
function updatePrediction() {
    const form = document.getElementById('simulator-form');
    if (!form.checkValidity()) {
        return; // Wait until numbers are valid
    }

    // A. Gather raw inputs from form
    const inputs = {
        Age: parseFloat(document.getElementById('input-Age').value) || 0,
        Gender: document.getElementById('input-Gender').value,
        State: document.getElementById('input-State').value,
        Tenure_Months: parseFloat(document.getElementById('input-Tenure_Months').value) || 0,
        Distance_To_Facility_Miles: parseFloat(document.getElementById('input-Distance_To_Facility_Miles').value) || 0,
        Visits_Last_Year: parseFloat(document.getElementById('input-Visits_Last_Year').value) || 0,
        Missed_Appointments: parseFloat(document.getElementById('input-Missed_Appointments').value) || 0,
        Days_Since_Last_Visit: parseFloat(document.getElementById('input-Days_Since_Last_Visit').value) || 0,
        Referrals_Made: parseFloat(document.getElementById('input-Referrals_Made').value) || 0,
        Avg_Out_Of_Pocket_Cost: parseFloat(document.getElementById('input-Avg_Out_Of_Pocket_Cost').value) || 0,
        Overall_Satisfaction: parseFloat(document.getElementById('input-Overall_Satisfaction').value) || 0,
        Wait_Time_Satisfaction: parseFloat(document.getElementById('input-Wait_Time_Satisfaction').value) || 0,
        Staff_Satisfaction: parseFloat(document.getElementById('input-Staff_Satisfaction').value) || 0,
        Provider_Rating: parseFloat(document.getElementById('input-Provider_Rating').value) || 0,
        Specialty: document.getElementById('input-Specialty').value,
        Insurance_Type: document.getElementById('input-Insurance_Type').value,
        Billing_Issues: document.getElementById('input-Billing_Issues').checked ? 1 : 0,
        Portal_Usage: document.getElementById('input-Portal_Usage').checked ? 1 : 0
    };

    // Calculate engineered features dynamically for the simulation
    inputs.Total_Satisfaction = inputs.Overall_Satisfaction + inputs.Wait_Time_Satisfaction + inputs.Staff_Satisfaction + inputs.Provider_Rating;
    inputs.Avg_Satisfaction = inputs.Total_Satisfaction / 4;
    inputs.Min_Satisfaction = Math.min(inputs.Overall_Satisfaction, inputs.Wait_Time_Satisfaction, inputs.Staff_Satisfaction, inputs.Provider_Rating);
    
    // Variance calculation (sample variance with Bessel's correction, divisor = 3)
    const mean_sat = inputs.Avg_Satisfaction;
    const diffs = [
        inputs.Overall_Satisfaction - mean_sat,
        inputs.Wait_Time_Satisfaction - mean_sat,
        inputs.Staff_Satisfaction - mean_sat,
        inputs.Provider_Rating - mean_sat
    ];
    inputs.Sat_Variance = diffs.reduce((acc, val) => acc + val*val, 0) / 3;
    
    inputs.Missed_Ratio = inputs.Missed_Appointments / (inputs.Visits_Last_Year + 1);
    inputs.Visit_Per_Month = inputs.Visits_Last_Year / (inputs.Tenure_Months / 12 + 1);
    inputs.Engagement_Score = (inputs.Portal_Usage * inputs.Referrals_Made) + inputs.Visits_Last_Year - inputs.Missed_Appointments;
    
    inputs.Log_Cost = Math.log1p(inputs.Avg_Out_Of_Pocket_Cost);
    inputs.Log_Distance = Math.log1p(inputs.Distance_To_Facility_Miles);
    inputs.Log_Days = Math.log1p(inputs.Days_Since_Last_Visit);
    inputs.Sqrt_Days = Math.sqrt(inputs.Days_Since_Last_Visit);
    inputs.Days_Per_Tenure = inputs.Days_Since_Last_Visit / (inputs.Tenure_Months + 1);
    
    inputs.Long_Absent = inputs.Days_Since_Last_Visit > 300 ? 1 : 0;
    inputs.Very_Long_Absent = inputs.Days_Since_Last_Visit > 500 ? 1 : 0;
    
    inputs.Cost_Per_Visit = inputs.Avg_Out_Of_Pocket_Cost / (inputs.Visits_Last_Year + 1);
    inputs.Cost_Tenure = inputs.Avg_Out_Of_Pocket_Cost / (inputs.Tenure_Months + 1);
    inputs.High_Cost_Low_Sat = inputs.Avg_Out_Of_Pocket_Cost * (5.0 - inputs.Overall_Satisfaction);
    
    inputs.Billing_Dissatisfaction = inputs.Billing_Issues * (5.0 - inputs.Overall_Satisfaction);
    
    inputs.Distance_Missed = inputs.Distance_To_Facility_Miles * inputs.Missed_Appointments;
    inputs.Provider_Overall_Dissatisfaction = (5 - inputs.Provider_Rating) * (5 - inputs.Overall_Satisfaction);
    inputs.Sat_Times_Portal = inputs.Overall_Satisfaction * inputs.Portal_Usage;
    inputs.Age_Tenure = inputs.Age * inputs.Tenure_Months;
    inputs.Recency_Missed = inputs.Days_Since_Last_Visit * inputs.Missed_Appointments;
    inputs.Sat_Tenure = inputs.Overall_Satisfaction * inputs.Tenure_Months;
    inputs.Cost_Sat = inputs.Avg_Out_Of_Pocket_Cost * inputs.Overall_Satisfaction;
    inputs.Visits_Sat = inputs.Visits_Last_Year * inputs.Overall_Satisfaction;
    
    const isPrivateOrSelfPay = (inputs.Insurance_Type === 'Private' || inputs.Insurance_Type === 'Self-Pay') ? 1 : 0;
    inputs.Private_SelfPay_Cost = inputs.Avg_Out_Of_Pocket_Cost * isPrivateOrSelfPay;

    // B. Transform inputs into scaled vector X_vec matching MODEL_DATA.feature_names
    const featureNames = MODEL_DATA.feature_names;
    const X_vec = {};
    const rawValues = {}; // Store raw values for Decision Tree rule comparisons

    featureNames.forEach(feature => {
        // Check if numeric column (needs scaling)
        if (MODEL_DATA.numeric_cols.includes(feature)) {
            const rawVal = inputs[feature];
            rawValues[feature] = rawVal;
            const mean = MODEL_DATA.scaler_params[feature].mean;
            const std = MODEL_DATA.scaler_params[feature].std;
            X_vec[feature] = (rawVal - mean) / std;
        } 
        // Check if basic binary column
        else if (MODEL_DATA.binary_cols.includes(feature)) {
            if (feature === 'Gender_Male') {
                const val = inputs.Gender === 'Male' ? 1 : 0;
                X_vec[feature] = val;
                rawValues[feature] = val;
            } else if (feature.startsWith('State_') || feature.startsWith('Specialty_') || feature.startsWith('Insurance_Type_')) {
                // Categorical one-hot encoded columns (handled in the next block)
                X_vec[feature] = 0;
                rawValues[feature] = 0;
            } else {
                const val = inputs[feature] !== undefined ? inputs[feature] : 0;
                X_vec[feature] = val;
                rawValues[feature] = val;
            }
        }
        // One-hot encoded categorical columns (State, Specialty, Insurance_Type)
        else {
            const isState = feature.startsWith('State_');
            const isSpecialty = feature.startsWith('Specialty_');
            const isInsurance = feature.startsWith('Insurance_Type_');

            if (isState) {
                const stateVal = feature.substring(6);
                const val = inputs.State === stateVal ? 1 : 0;
                X_vec[feature] = val;
                rawValues[feature] = val;
            } else if (isSpecialty) {
                const specVal = feature.substring(10);
                const val = inputs.Specialty === specVal ? 1 : 0;
                X_vec[feature] = val;
                rawValues[feature] = val;
            } else if (isInsurance) {
                const insVal = feature.substring(15);
                const val = inputs.Insurance_Type === insVal ? 1 : 0;
                X_vec[feature] = val;
                rawValues[feature] = val;
            } else {
                X_vec[feature] = 0;
                rawValues[feature] = 0;
            }
        }
    });

    // C. Calculate Logistic Regression Probability
    let lrLogit = MODEL_DATA.logistic_regression.model.intercept;
    const coefficients = MODEL_DATA.logistic_regression.model.coefficients;
    const lrContributions = {};

    featureNames.forEach(feature => {
        const val = X_vec[feature];
        const coef = coefficients[feature] || 0;
        const contrib = val * coef;
        lrLogit += contrib;
        lrContributions[feature] = contrib;
    });

    const lrProb = sigmoid(lrLogit);

    // D. Calculate Decision Tree Probability and Traverse Path
    const nodes = MODEL_DATA.decision_tree.nodes;
    let currentNodeId = 0;
    const dtPath = [0];

    while (!nodes[currentNodeId].is_leaf) {
        const node = nodes[currentNodeId];
        const feature = node.feature;
        const threshold = node.threshold;
        const scaledVal = X_vec[feature];

        if (scaledVal <= threshold) {
            currentNodeId = node.left_child;
        } else {
            currentNodeId = node.right_child;
        }
        dtPath.push(currentNodeId);
    }
    const dtProb = nodes[currentNodeId].churn_probability;

    // E. Update Gauges in the UI
    updateGauge('lr', lrProb);
    updateGauge('dt', dtProb);

    // F. Render Individual Contribution Breakdown Chart (explainable AI)
    updateIndividualBreakdownChart(lrContributions);

    // G. Generate advisory console recommendations
    updateAdvisoryConsole(inputs, lrProb);

    // H. Render Decision Tree active path tracing
    renderDecisionTreePath(dtPath, X_vec, rawValues);

    // I. Send inputs to Flask API for dynamic prediction & SHAP if online
    if (isApiOnline) {
        sendDynamicPredictPayload(inputs);
    }
}

// Function to animate and color-code gauge fill
function updateGauge(modelPrefix, prob) {
    const valueEl = document.getElementById(`${modelPrefix}-churn-prob`);
    const fillEl = document.getElementById(`${modelPrefix}-gauge-fill`);
    if (!valueEl || !fillEl) return;
    
    const percentage = (prob * 100).toFixed(1) + '%';
    valueEl.textContent = percentage;

    // Map probability (0 to 1) to dashoffset (125.6 to 0)
    // 125.6 corresponds to full semi-circle length
    const offset = 125.6 * (1 - prob);
    fillEl.style.strokeDashoffset = offset;

    // Dynamic Color-coding
    let color = 'var(--risk-low)';
    if (prob >= 0.4 && prob < 0.7) {
        color = 'var(--risk-medium)';
    } else if (prob >= 0.7) {
        color = 'var(--risk-high)';
    }
    fillEl.style.stroke = color;
}

// Render dynamic explainable risk factors chart
function updateIndividualBreakdownChart(contributions) {
    const canvas = document.getElementById('individual-breakdown-chart');
    if (!canvas) return;

    // Sort contributions by magnitude
    const entries = Object.entries(contributions);
    const sorted = entries.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    
    // Pick the top 7 contributors to avoid clutter
    const topContributors = sorted.slice(0, 7);

    // Human-friendly names helper
    const getFriendlyName = (feat) => {
        if (feat.startsWith('State_')) return `State: ${feat.substring(6)}`;
        if (feat.startsWith('Specialty_')) return `Provider Specialty: ${feat.substring(10)}`;
        if (feat.startsWith('Insurance_Type_')) return `Insurance: ${feat.substring(15)}`;
        if (feat === 'Gender_Male') return 'Gender: Male';
        return feat.replace(/_/g, ' ');
    };

    const labels = topContributors.map(item => getFriendlyName(item[0]));
    const values = topContributors.map(item => item[1]);
    const colors = values.map(val => val > 0 ? 'rgba(244, 63, 94, 0.7)' : 'rgba(16, 185, 129, 0.7)');

    const ctx = canvas.getContext('2d');
    
    if (individualBreakdownChart) {
        individualBreakdownChart.destroy();
    }

    individualBreakdownChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Log-Odds Churn Contribution (Direction & Strength)',
                data: values,
                backgroundColor: colors,
                borderColor: values.map(v => v > 0 ? 'var(--risk-high)' : 'var(--risk-low)'),
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'Negative Impact (Lowers Risk) ◀  ▶ Positive Impact (Increases Risk)', color: '#9ca3af', font: { size: 10 } },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#f3f4f6', font: { size: 11, weight: '500' } }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const val = context.raw;
                            const direction = val > 0 ? 'increases risk' : 'reduces risk';
                            return `Impact: ${val.toFixed(2)} (${direction})`;
                        }
                    }
                }
            }
        }
    });
}

// Generate tailored retention strategies based on risk levels and parameters
function updateAdvisoryConsole(inputs, prob) {
    const summaryEl = document.getElementById('advisory-summary');
    const listEl = document.getElementById('intervention-list');
    if (!summaryEl || !listEl) return;
    
    // Clear list
    listEl.innerHTML = '';

    // A. Determine overall risk level
    let riskClass = 'low-risk';
    let riskLabel = 'LOW RISK';
    let summaryText = 'Member is currently at Low Churn Risk. Standard engagement and preventative health outreach recommended.';

    if (prob >= 0.4 && prob < 0.7) {
        riskClass = 'medium-risk';
        riskLabel = 'MEDIUM RISK';
        summaryText = 'Member is at Moderate Risk. Targeted operational and outreach workflows are recommended below to reduce churn drivers.';
    } else if (prob >= 0.7) {
        riskClass = 'high-risk';
        riskLabel = 'HIGH RISK';
        summaryText = 'Member is at Critical Churn Risk. Immediate proactive interventions from member care team and administrative concierges are vital.';
    }

    summaryEl.className = `advisory-summary ${riskClass}`;
    summaryEl.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Churn Risk Profile: <strong>${riskLabel} (${(prob * 100).toFixed(1)}%)</strong><br><span style="font-size:0.8rem; font-weight:normal; color:var(--text-secondary);">${summaryText}</span>`;

    // B. Build interventions array based on patient profile
    const interventions = [];

    // Intervention 1: Billing issues reported
    if (inputs.Billing_Issues === 1) {
        interventions.push({
            priority: 'high',
            icon: 'fa-solid fa-file-invoice-dollar',
            title: 'Deploy Billing Concierge outreach',
            desc: 'Member reported unresolved billing issues. Assign an account concierge to contact the member within 24 hours, audit prior claims, resolve disputes, and configure automated billing/payment setups.'
        });
    }

    // Intervention 2: Missed appointments
    if (inputs.Missed_Appointments >= 2) {
        interventions.push({
            priority: 'high',
            icon: 'fa-solid fa-calendar-times',
            title: 'Schedule Transportation Check & Active Reminders',
            desc: `Member missed ${inputs.Missed_Appointments} appointments. Proactively check if transportation barriers exist (Social Determinants of Health). Integrate automated interactive voice response (IVR) calls and SMS reminders.`
        });
    }

    // Intervention 3: Poor satisfaction scores
    if (inputs.Overall_Satisfaction < 3.0) {
        interventions.push({
            priority: 'high',
            icon: 'fa-solid fa-headset',
            title: 'Member Advocacy Care Call',
            desc: `Member satisfaction is extremely low (${inputs.Overall_Satisfaction}/5). Trigger a priority outreach from a Member Advocate to identify primary clinical/administrative points of friction and resolve care access barriers.`
        });
    }

    // Intervention 4: Long days since last visit
    if (inputs.Days_Since_Last_Visit > 180) {
        const priority = inputs.Days_Since_Last_Visit > 300 ? 'high' : 'medium';
        interventions.push({
            priority: priority,
            icon: 'fa-solid fa-clock-rotate-left',
            title: 'Preventative Re-engagement Campaign',
            desc: `It has been ${inputs.Days_Since_Last_Visit} days since the member visited a provider. Initiate a routine wellness campaign check-in, assist with booking an annual wellness exam, and coordinate routine screenings.`
        });
    }

    // Intervention 5: Wait time dissatisfaction
    if (inputs.Wait_Time_Satisfaction < 3.0) {
        interventions.push({
            priority: 'medium',
            icon: 'fa-solid fa-hourglass-half',
            title: 'Offer Telehealth Options & Provider Matching',
            desc: `Member is dissatisfied with clinic wait times (${inputs.Wait_Time_Satisfaction}/5). Walk through CareShield Digital Telehealth registration to bypass in-person wait times for routine follow-ups, or offer faster wait-time provider recommendations.`
        });
    }

    // Intervention 6: Long distance
    if (inputs.Distance_To_Facility_Miles > 20) {
        interventions.push({
            priority: 'medium',
            icon: 'fa-solid fa-route',
            title: 'Promote Virtual Care & Telehealth Portal',
            desc: `Member lives ${inputs.Distance_To_Facility_Miles} miles away from the primary care facility. Send instructions for setting up virtual primary care visits and mail order prescription services to limit transport friction.`
        });
    }

    // Intervention 7: No portal usage
    if (inputs.Portal_Usage === 0) {
        interventions.push({
            priority: 'low',
            icon: 'fa-solid fa-laptop-medical',
            title: 'Assist in Patient Portal Onboarding',
            desc: 'Member portal is inactive. Email detailed portal setup guides and provide phone assistance to register. Digital connectivity improves communication transparency and billing satisfaction.'
        });
    }

    // Intervention 8: High out of pocket cost
    if (inputs.Avg_Out_Of_Pocket_Cost > 1000) {
        interventions.push({
            priority: 'medium',
            icon: 'fa-solid fa-scale-balanced',
            title: 'Financial Counseling & Prescription Tier Substitution',
            desc: `Avg out-of-pocket cost is high ($${inputs.Avg_Out_Of_Pocket_Cost}). Connect the member with a financial counselor to explore subsidy eligibility, plan tier alternatives, and check for cost-effective generic medication alternatives.`
        });
    }

    // Fallback: If no custom intervention matches and member has high/medium risk, add default
    if (interventions.length === 0) {
        interventions.push({
            priority: 'low',
            icon: 'fa-solid fa-circle-info',
            title: 'Routine Care Coordination Check-in',
            desc: 'Initiate a standard care coordination review call to update member details, verify correct primary care assignments, and confirm satisfaction with plan network.'
        });
    }

    // Sort by priority (high first, then medium, then low)
    const priorityOrder = { 'high': 1, 'medium': 2, 'low': 3 };
    interventions.sort((a, b) => priorityOrder[a.priority] - priorityOrder[b.priority]);

    // Render items
    interventions.forEach(item => {
        const itemEl = document.createElement('div');
        itemEl.className = `intervention-item ${item.priority}`;
        itemEl.innerHTML = `
            <div class="intervention-icon">
                <i class="${item.icon}"></i>
            </div>
            <div class="intervention-details">
                <h5>${item.title} <span class="node-tag ${item.priority === 'high' ? 'churn' : (item.priority === 'medium' ? 'medium-text' : 'safe')}" style="margin-top:0; font-size:0.55rem; padding: 1px 4px;">${item.priority.toUpperCase()}</span></h5>
                <p>${item.desc}</p>
            </div>
        `;
        listEl.appendChild(itemEl);
    });
}

// Visual decision tree path tracer. Displays step-by-step logic path with unscaled variables
function renderDecisionTreePath(activePath, X_vec, rawValues) {
    const container = document.getElementById('tree-container');
    if (!container) return;
    try {
        container.innerHTML = ''; // Clear fallback

        const nodes = MODEL_DATA.decision_tree.nodes;

        // Create container element
        const pathWrapper = document.createElement('div');
        pathWrapper.className = 'tree-graph';
        
        // We will render the nodes sequentially as a timeline/flowcard layout
        // which looks incredibly premium and is highly readable on all screens!
        activePath.forEach((nodeId, idx) => {
            const node = nodes[nodeId];
            const nodeCard = document.createElement('div');
            nodeCard.className = 'tree-node active-path';
            
            // Is it a leaf node?
            if (node.is_leaf) {
                nodeCard.classList.add('leaf-node');
                const isSafe = node.churn_probability < 0.4;
                if (isSafe) nodeCard.classList.add('safe');
                
                nodeCard.innerHTML = `
                    <div class="node-rule"><i class="fa-solid fa-flag"></i> Leaf Node REACHED</div>
                    <div style="font-size:0.75rem; color:var(--text-secondary); margin:6px 0;">
                        Node Sample Size: <strong>n = ${node.samples}</strong>
                    </div>
                    <span class="node-tag ${isSafe ? 'safe' : 'churn'}">
                        Churn rate: ${(node.churn_probability * 100).toFixed(1)}%
                    </span>
                `;
            } else {
                // Unscale the split threshold for business user readability
                const feature = node.feature;
                const threshold = node.threshold;
                
                let ruleText = '';
                let valText = '';
                
                let rawVal = rawValues[feature];
                if (rawVal === undefined || rawVal === null) {
                    rawVal = 0;
                }
                
                if (MODEL_DATA.numeric_cols.includes(feature)) {
                    // Unscale threshold: thresh * std + mean
                    const mean = MODEL_DATA.scaler_params[feature].mean;
                    const std = MODEL_DATA.scaler_params[feature].std;
                    const unscaledThreshold = threshold * std + mean;
                    
                    // Formulate rule descriptions based on feature names
                    let featureLabel = feature.replace(/_/g, ' ');
                    if (feature === 'Tenure_Months') featureLabel = 'Tenure';
                    
                    ruleText = `${featureLabel} &le; ${unscaledThreshold.toFixed(1)}`;
                    valText = `Member value: <strong>${rawVal.toFixed(1)}</strong>`;
                } else {
                    // Binary features (Billing_Issues, Portal_Usage, etc.)
                    let featureLabel = feature.replace(/_/g, ' ');
                    if (feature.startsWith('State_')) featureLabel = `State is ${feature.substring(6)}`;
                    else if (feature.startsWith('Specialty_')) featureLabel = `Specialty is ${feature.substring(10)}`;
                    else if (feature.startsWith('Insurance_Type_')) featureLabel = `Insurance is ${feature.substring(15)}`;
                    
                    ruleText = `${featureLabel} = No`;
                    valText = `Member value: <strong>${rawVal === 1 ? 'Yes' : 'No'}</strong>`;
                }
                
                // Check next node in path to see which direction we took
                const nextNodeId = activePath[idx + 1];
                const directionText = nextNodeId === node.left_child ? 
                    '<span class="risk-low-text"><i class="fa-solid fa-circle-check"></i> Rule Matches (Yes)</span>' : 
                    '<span class="risk-high-text"><i class="fa-solid fa-circle-xmark"></i> Rule Fails (No)</span>';

                nodeCard.innerHTML = `
                    <div class="node-rule" style="font-family:var(--font-heading);">${ruleText}</div>
                    <div style="font-size:0.7rem; color:var(--text-muted);">${valText}</div>
                    <div class="node-stats">
                        <span>n = ${node.samples}</span>
                        <span>Rate: ${(node.churn_probability * 100).toFixed(0)}%</span>
                    </div>
                    <div class="node-prob-bar">
                        <div class="node-prob-fill" style="width: ${node.churn_probability * 100}%"></div>
                    </div>
                    <div style="font-size:0.65rem; font-weight:700; margin-top:8px;">
                        Decision: ${directionText}
                    </div>
                `;
            }

            pathWrapper.appendChild(nodeCard);
            
            // Add vertical arrows between steps (except for the last leaf node)
            if (idx < activePath.length - 1) {
                const arrow = document.createElement('div');
                arrow.className = 'path-arrow';
                arrow.style.cssText = 'color: var(--color-secondary); font-size: 1.25rem; margin: -10px 0; z-index: 10; animation: pulse 2s infinite;';
                arrow.innerHTML = '<i class="fa-solid fa-angles-down"></i>';
                pathWrapper.appendChild(arrow);
            }
        });

        container.appendChild(pathWrapper);
    } catch (e) {
        console.error("Error tracing decision path:", e);
        container.innerHTML = `<div class="tree-error" style="color:var(--risk-high); font-size:0.9rem; text-align:center; padding:20px; border: 1px dashed rgba(244,63,94,0.3); border-radius:10px; background:var(--risk-high-bg);"><i class="fa-solid fa-triangle-exclamation"></i> Error tracing active decision path: ${e.message}</div>`;
    }
}

// Utility to handle floating point comparisons safely
function scaledValLessThanEqual(val, threshold) {
    // Standard margin for machine precision
    return val <= threshold + 1e-9;
}

// REST API Integration Functions
async function checkApiStatus() {
    const statusEl = document.getElementById('api-status-text') || document.getElementById('load-status');
    const indicatorEl = document.querySelector('.status-indicator');
    if (!statusEl) return;
    statusEl.textContent = 'Connecting to REST API...';
    try {
        const res = await fetch(`${API_URL}/health`);
        const data = await res.json();
        if (data.status === 'ok') {
            isApiOnline = true;
            statusEl.textContent = 'REST API: Online';
            if (indicatorEl) {
                indicatorEl.style.backgroundColor = '#10b981';
                indicatorEl.style.boxShadow = '0 0 8px #10b981';
            }
            
            // Load dashboard and selector details
            loadDashboardData().catch(e => console.warn('Dashboard data warning:', e));
            loadSelectorMembers().catch(e => console.warn('Selector members warning:', e));
        } else {
            throw new Error('Health check failed');
        }
    } catch (e) {
        console.error('REST API server offline. Falling back to local data.', e);
        isApiOnline = false;
        statusEl.textContent = 'REST API: Offline';
        if (indicatorEl) {
            indicatorEl.style.backgroundColor = '#f43f5e';
            indicatorEl.style.boxShadow = 'none';
        }
        
        // Populate local fallback dashboard stats
        loadLocalDashboardFallback();
    }
}

async function loadDashboardData() {
    try {
        const res = await fetch(`${API_URL}/analytics`);
        const data = await res.json();
        
        // Populate KPIs
        document.getElementById('kpi-total-members').textContent = data.total_members.toLocaleString();
        
        const dist = data.risk_distribution;
        const highRiskCount = (dist.High || 0) + (dist.Critical || 0);
        document.getElementById('kpi-high-risk').textContent = highRiskCount.toLocaleString();
        
        const avgProbPct = (data.average_risk * 100).toFixed(1) + '%';
        document.getElementById('kpi-avg-risk').textContent = avgProbPct;
        
        const oppCount = (dist.Medium || 0) + (dist.High || 0) + (dist.Critical || 0);
        document.getElementById('kpi-opportunities').textContent = oppCount.toLocaleString();
        
        // Render Charts
        renderRiskDistChart(dist);
        renderRiskTrendChart(data.tenure_risk_trend);
        renderDashboardDriversChart(data.top_risk_drivers);
        
    } catch (e) {
        console.error('Error loading dashboard analytics:', e);
        loadLocalDashboardFallback();
    }
}

function loadLocalDashboardFallback() {
    // Falls back to hardcoded numbers based on our actual dataset counts
    document.getElementById('kpi-total-members').textContent = '2,000';
    document.getElementById('kpi-high-risk').textContent = '650';
    document.getElementById('kpi-avg-risk').textContent = '56.3%';
    document.getElementById('kpi-opportunities').textContent = '1,476';
    
    // Offline local distribution
    const mockDist = { "Low": 524, "Medium": 826, "High": 486, "Critical": 164 };
    renderRiskDistChart(mockDist);
    
    // Offline local trend
    const mockTrend = [
        {"avg_probability": 0.6277, "cohort": "0-12m"},
        {"avg_probability": 0.5773, "cohort": "13-24m"},
        {"avg_probability": 0.6061, "cohort": "25-36m"},
        {"avg_probability": 0.6323, "cohort": "37-48m"},
        {"avg_probability": 0.5667, "cohort": "49-60m"},
        {"avg_probability": 0.5682, "cohort": "61-72m"},
        {"avg_probability": 0.5414, "cohort": "73-84m"},
        {"avg_probability": 0.4892, "cohort": "85-96m"},
        {"avg_probability": 0.4936, "cohort": "97-108m"},
        {"avg_probability": 0.5398, "cohort": "109-120m"}
    ];
    renderRiskTrendChart(mockTrend);
    
    // Offline local drivers
    const mockDrivers = [
        {"feature": "Overall_Satisfaction", "label": "Overall satisfaction score", "mean_importance": 0.28217},
        {"feature": "Days_Since_Last_Visit", "label": "Days since last clinical visit", "mean_importance": 0.25842},
        {"feature": "Age", "label": "Age (years)", "mean_importance": 0.21928},
        {"feature": "Distance_To_Facility_Miles", "label": "Distance to facility (miles)", "mean_importance": 0.21423},
        {"feature": "composite_satisfaction", "label": "Composite satisfaction score", "mean_importance": 0.17696}
    ];
    renderDashboardDriversChart(mockDrivers);
}

function renderRiskDistChart(dist) {
    const ctx = document.getElementById('dashboard-risk-dist-chart').getContext('2d');
    if (dashRiskDistChart) dashRiskDistChart.destroy();
    
    dashRiskDistChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'],
            datasets: [{
                label: 'Member Count',
                data: [dist.Low || 0, dist.Medium || 0, dist.High || 0, dist.Critical || 0],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.65)',
                    'rgba(245, 158, 11, 0.65)',
                    'rgba(244, 63, 94, 0.65)',
                    'rgba(224, 30, 90, 0.85)'
                ],
                borderColor: [
                    '#10b981', '#f59e0b', '#f43f5e', '#e11d48'
                ],
                borderWidth: 1.5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#9ca3af' } },
                y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#9ca3af' } }
            }
        }
    });
}

function renderRiskTrendChart(trend) {
    const ctx = document.getElementById('dashboard-risk-trend-chart').getContext('2d');
    if (dashRiskTrendChart) dashRiskTrendChart.destroy();
    
    dashRiskTrendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: trend.map(t => t.cohort),
            datasets: [{
                label: 'Average Churn Probability',
                data: trend.map(t => t.avg_probability),
                borderColor: '#0d9488',
                backgroundColor: 'rgba(13, 148, 136, 0.1)',
                borderWidth: 2.5,
                fill: true,
                tension: 0.35,
                pointRadius: 3,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { grid: { color: 'rgba(255, 255, 255, 0.03)' }, ticks: { color: '#9ca3af' } },
                y: { 
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }, 
                    ticks: { 
                        color: '#9ca3af',
                        callback: function(value) { return (value * 100).toFixed(0) + '%'; }
                    } 
                }
            }
        }
    });
}

function renderDashboardDriversChart(drivers) {
    const ctx = document.getElementById('dashboard-drivers-chart').getContext('2d');
    if (dashDriversChart) dashDriversChart.destroy();
    
    dashDriversChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: drivers.map(d => d.label),
            datasets: [{
                label: 'Impact Strength',
                data: drivers.map(d => d.mean_importance),
                backgroundColor: 'rgba(99, 102, 241, 0.65)',
                borderColor: '#6366f1',
                borderWidth: 1.5
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { 
                    title: { display: true, text: 'Average Impact on Churn Probability', color: '#9ca3af', font: { size: 10 } },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }, 
                    ticks: { color: '#9ca3af' } 
                },
                y: { grid: { display: false }, ticks: { color: '#f3f4f6', font: { size: 11, weight: '500' } } }
            }
        }
    });
}

async function loadSelectorMembers() {
    const selector = document.getElementById('member-selector');
    if (!selector) return;
    try {
        const res = await fetch(`${API_URL}/members`);
        const members = await res.json();
        
        // Sort member IDs naturally
        members.sort((a, b) => a.member_id.localeCompare(b.member_id, undefined, {numeric: true, sensitivity: 'base'}));
        
        selector.innerHTML = '<option value="">-- Load Simulator Form --</option>';
        members.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.member_id;
            opt.textContent = `${m.member_id} (${(m.churn_probability * 100).toFixed(0)}% - ${m.risk_level})`;
            selector.appendChild(opt);
        });
        
        selector.addEventListener('change', async (e) => {
            const memberId = e.target.value;
            if (!memberId) return;
            
            const statusEl = document.getElementById('load-status');
            statusEl.textContent = `Loading member ${memberId}...`;
            statusEl.style.color = 'var(--risk-medium)';
            
            try {
                const memberRes = await fetch(`${API_URL}/member/${memberId}`);
                const detail = await memberRes.json();
                
                // Populate the inputs
                populateFormWithMember(detail.member_info);
                
                statusEl.textContent = `Member ${memberId} loaded successfully!`;
                statusEl.style.color = 'var(--risk-low)';
                
                // Trigger dynamic predictions
                updatePrediction();
                
                // Update gauges to show dynamic result (XGBoost)
                if (isApiOnline) {
                    updateGauge('lr', detail.churn_probability);
                    updateGauge('dt', detail.churn_probability); // Use the second gauge for XGBoost when online
                    
                    document.querySelector('#simulator-tab .gauge-wrapper:nth-child(2) .gauge-title').textContent = 'Ensemble Model';
                    document.querySelector('#simulator-tab .gauge-wrapper:nth-child(2) .gauge-description').textContent = 'Actual ensemble probability';
                    
                    populateAdvisorConsoleFromApi(detail.recommendations, detail.churn_probability, detail.risk_level);
                    populateDriversFromApi(detail.drivers);
                }
            } catch (err) {
                console.error(err);
                statusEl.textContent = 'Error loading member details.';
                statusEl.style.color = 'var(--risk-high)';
            }
        });
    } catch (e) {
        console.error('Error loading selector list:', e);
    }
}

function populateFormWithMember(info) {
    document.getElementById('input-Age').value = info.Age;
    document.getElementById('input-Gender').value = info.Gender;
    document.getElementById('input-State').value = info.State;
    
    document.getElementById('input-Tenure_Months').value = info.Tenure_Months;
    document.getElementById('input-Insurance_Type').value = info.Insurance_Type;
    document.getElementById('input-Distance_To_Facility_Miles').value = info.Distance_To_Facility_Miles;
    document.getElementById('input-Visits_Last_Year').value = info.Visits_Last_Year;
    document.getElementById('input-Missed_Appointments').value = info.Missed_Appointments;
    document.getElementById('input-Days_Since_Last_Visit').value = info.Days_Since_Last_Visit;
    document.getElementById('input-Referrals_Made').value = info.Referrals_Made;
    document.getElementById('input-Avg_Out_Of_Pocket_Cost').value = info.Avg_Out_Of_Pocket_Cost;
    document.getElementById('input-Specialty').value = info.Specialty;
    
    document.getElementById('input-Billing_Issues').checked = info.Billing_Issues === 1;
    document.getElementById('input-Portal_Usage').checked = info.Portal_Usage === 1;
    
    document.getElementById('input-Overall_Satisfaction').value = info.Overall_Satisfaction;
    document.getElementById('input-Wait_Time_Satisfaction').value = info.Wait_Time_Satisfaction;
    document.getElementById('input-Staff_Satisfaction').value = info.Staff_Satisfaction;
    document.getElementById('input-Provider_Rating').value = info.Provider_Rating;
    
    document.getElementById('val-Overall_Satisfaction').textContent = info.Overall_Satisfaction.toFixed(1);
    document.getElementById('val-Wait_Time_Satisfaction').textContent = info.Wait_Time_Satisfaction.toFixed(1);
    document.getElementById('val-Staff_Satisfaction').textContent = info.Staff_Satisfaction.toFixed(1);
    document.getElementById('val-Provider_Rating').textContent = info.Provider_Rating.toFixed(1);
}

function populateAdvisorConsoleFromApi(recommendations, prob, riskLevel) {
    const summaryEl = document.getElementById('advisory-summary');
    const listEl = document.getElementById('intervention-list');
    listEl.innerHTML = '';
    
    let riskClass = riskLevel.toLowerCase() + '-risk';
    let riskLabel = riskLevel.toUpperCase();
    let summaryText = 'Standard routine monitoring and preventive care workflows recommended.';
    
    if (riskLevel === 'Medium') {
        summaryText = 'Member is at Moderate Risk. Targeted operational and outreach workflows recommended below.';
    } else if (riskLevel === 'High' || riskLevel === 'Critical') {
        summaryText = 'Immediate proactive interventions from administrative concierges are vital to address drivers.';
    }
    
    summaryEl.className = `advisory-summary ${riskClass}`;
    summaryEl.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Churn Risk Profile: <strong>${riskLabel} (${(prob * 100).toFixed(1)}%)</strong><br><span style="font-size:0.8rem; font-weight:normal; color:var(--text-secondary);">${summaryText}</span>`;
    
    if (recommendations.length === 0) {
        listEl.innerHTML = '<div style="color:var(--text-secondary); font-size:0.85rem; padding:10px; text-align:center;">No interventions generated for this profile.</div>';
        return;
    }
    
    recommendations.forEach(item => {
        const itemEl = document.createElement('div');
        const priorityClass = item.priority.toLowerCase();
        itemEl.className = `intervention-item ${priorityClass}`;
        
        let icon = 'fa-solid fa-circle-info';
        if (item.category === 'Service Recovery') icon = 'fa-solid fa-file-invoice-dollar';
        else if (item.category === 'Care Outreach') icon = 'fa-solid fa-calendar-times';
        else if (item.category === 'Benefit Education') icon = 'fa-solid fa-laptop-medical';
        else if (item.category === 'Pharmacy Support') icon = 'fa-solid fa-scale-balanced';
        
        itemEl.innerHTML = `
            <div class="intervention-icon">
                <i class="${icon}"></i>
            </div>
            <div class="intervention-details">
                <h5>${item.action} <span class="node-tag ${priorityClass === 'critical' || priorityClass === 'high' ? 'churn' : (priorityClass === 'medium' ? 'medium-text' : 'safe')}" style="margin-top:0; font-size:0.55rem; padding: 1px 4px;">${item.priority.toUpperCase()}</span></h5>
                <p><strong>Reason:</strong> ${item.reason}</p>
                <p style="font-size:0.75rem; color:var(--text-secondary); margin-top:4px;"><strong>Evidence:</strong> ${item.evidence.join('; ')}</p>
                <p style="font-size:0.75rem; color:var(--text-secondary); margin-top:2px;"><strong>Next Step:</strong> ${item.suggested_next_step}</p>
            </div>
        `;
        listEl.appendChild(itemEl);
    });
}

function populateDriversFromApi(drivers) {
    const labels = drivers.map(d => d.label);
    const values = drivers.map(d => d.shap_value);
    const colors = values.map(val => val > 0 ? 'rgba(244, 63, 94, 0.7)' : 'rgba(16, 185, 129, 0.7)');
    
    const ctx = document.getElementById('individual-breakdown-chart').getContext('2d');
    if (individualBreakdownChart) individualBreakdownChart.destroy();
    
    individualBreakdownChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Impact on Risk (Increases / Decreases)',
                data: values,
                backgroundColor: colors,
                borderColor: values.map(v => v > 0 ? 'var(--risk-high)' : 'var(--risk-low)'),
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'Negative Impact (Lowers Risk) ◀  ▶ Positive Impact (Increases Risk)', color: '#9ca3af', font: { size: 10 } },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' }
                },
                y: { grid: { display: false }, ticks: { color: '#f3f4f6', font: { size: 11, weight: '500' } } }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

async function sendDynamicPredictPayload(inputs) {
    try {
        const res = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputs)
        });
        const detail = await res.json();
        
        // Update Gauges to the actual XGBoost and LR probabilities
        updateGauge('lr', detail.churn_probability);
        updateGauge('dt', detail.churn_probability);
        
        // Populate recommendations and drivers
        populateAdvisorConsoleFromApi(detail.recommendations, detail.churn_probability, detail.risk_level);
        populateDriversFromApi(detail.drivers);
    } catch (e) {
        console.error('Dynamic predict API call failed:', e);
    }
}


// Global reset helper for file upload UI
window.resetUploadArea = function() {
    const fileInput = document.getElementById('csv-file-input');
    const dropZone = document.getElementById('upload-drop-zone');
    const fileSelectedRow = document.getElementById('file-selected-row');
    const btnAnalyze = document.getElementById('btn-analyze-dataset');
    const errorMessage = document.getElementById('upload-error-message');
    const progressContainer = document.getElementById('upload-progress-container');

    if (fileInput) fileInput.value = '';
    if (dropZone) dropZone.style.display = 'block';
    if (fileSelectedRow) fileSelectedRow.style.display = 'none';
    if (btnAnalyze) btnAnalyze.disabled = true;
    if (errorMessage) errorMessage.style.display = 'none';
    if (progressContainer) progressContainer.style.display = 'none';
};

// Initialize Upload Tab Logic
function initUploadFlow() {
    const fileInput = document.getElementById('csv-file-input');
    const dropZone = document.getElementById('upload-drop-zone');
    const fileSelectedRow = document.getElementById('file-selected-row');
    const selectedFileName = document.getElementById('selected-file-name');
    const btnAnalyze = document.getElementById('btn-analyze-dataset');
    const progressContainer = document.getElementById('upload-progress-container');
    const errorMessage = document.getElementById('upload-error-message');
    const successActions = document.getElementById('upload-success-actions');
    const spinner = document.getElementById('upload-spinner');

    let selectedFile = null;

    // Handle file selection
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Drag and drop event listeners
    if (dropZone) {
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = 'var(--color-primary)';
            dropZone.style.background = 'rgba(255, 255, 255, 0.05)';
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.style.borderColor = 'var(--border-card)';
            dropZone.style.background = 'rgba(255, 255, 255, 0.02)';
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = 'var(--border-card)';
            dropZone.style.background = 'rgba(255, 255, 255, 0.02)';
            if (e.dataTransfer.files.length > 0) {
                handleFileSelect(e.dataTransfer.files[0]);
            }
        });
    }

    function handleFileSelect(file) {
        selectedFile = file;
        if (selectedFileName) {
            selectedFileName.textContent = `File Uploaded: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        }
        if (dropZone) dropZone.style.display = 'none';
        if (fileSelectedRow) fileSelectedRow.style.display = 'flex';
        btnAnalyze.disabled = false;
        errorMessage.style.display = 'none';
        progressContainer.style.display = 'none';
    }

    function showError(text) {
        errorMessage.textContent = text;
        errorMessage.style.display = 'block';
        progressContainer.style.display = 'none';
        successActions.style.display = 'none';
    }

    // Handle "Analyze Dataset" click
    btnAnalyze.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Reset progress steps UI
        const steps = ['step-validate', 'step-predict', 'step-shap', 'step-recommend'];
        steps.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.className = 'progress-step';
                el.querySelector('i').className = 'fa-regular fa-circle';
            }
        });

        errorMessage.style.display = 'none';
        successActions.style.display = 'none';
        progressContainer.style.display = 'block';
        spinner.style.display = 'inline-block';
        btnAnalyze.disabled = true;

        const formData = new FormData();
        formData.append('file', selectedFile);

        // Step 1: Validation in progress
        updateStepStatus('step-validate', 'active-step', 'fa-solid fa-spinner fa-spin');

        try {
            await sleep(600);
            
            const res = await fetch(`${API_URL}/predict-file`, {
                method: 'POST',
                body: formData
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.error || 'Server processing failed.');
            }

            const data = await res.json();

            // Transition validation to success, start prediction
            updateStepStatus('step-validate', 'success', 'fa-solid fa-circle-check');
            updateStepStatus('step-predict', 'active-step', 'fa-solid fa-spinner fa-spin');
            await sleep(600);

            // Transition prediction to success, start shap
            updateStepStatus('step-predict', 'success', 'fa-solid fa-circle-check');
            updateStepStatus('step-shap', 'active-step', 'fa-solid fa-spinner fa-spin');
            await sleep(600);

            // Transition shap to success, start recommendation
            updateStepStatus('step-shap', 'success', 'fa-solid fa-circle-check');
            updateStepStatus('step-recommend', 'active-step', 'fa-solid fa-spinner fa-spin');
            await sleep(600);

            // Complete recommendations
            updateStepStatus('step-recommend', 'success', 'fa-solid fa-circle-check');
            spinner.style.display = 'none';

            // Show dashboard transition button
            successActions.style.display = 'block';
            isApiOnline = true;
            applyAnalysisResult(data);

            // Automatically navigate to Strategic Dashboard
            setTimeout(() => {
                const dashNav = document.getElementById('nav-dashboard');
                if (dashNav) dashNav.click();
            }, 300);

        } catch (err) {
            console.error("Upload error:", err);
            showError(err.message || "An error occurred while analyzing the dataset.");
            spinner.style.display = 'none';
            btnAnalyze.disabled = false;
        }
    });

    // Handle "Load Demo Data" button — calls /load-demo (no file needed)
    const btnDemo = document.getElementById('btn-load-demo');
    if (btnDemo) {
        btnDemo.addEventListener('click', async () => {
            const steps = ['step-validate', 'step-predict', 'step-shap', 'step-recommend'];
            steps.forEach(id => {
                const el = document.getElementById(id);
                if (el) {
                    el.className = 'progress-step';
                    el.querySelector('i').className = 'fa-regular fa-circle';
                }
            });

            errorMessage.style.display = 'none';
            successActions.style.display = 'none';
            progressContainer.style.display = 'block';
            spinner.style.display = 'inline-block';
            btnDemo.disabled = true;

            updateStepStatus('step-validate', 'active-step', 'fa-solid fa-spinner fa-spin');

            try {
                await sleep(500);
                const res = await fetch(`${API_URL}/load-demo`);
                if (!res.ok) {
                    const errData = await res.json();
                    throw new Error(errData.error || 'Demo load failed on server.');
                }
                const data = await res.json();

                updateStepStatus('step-validate', 'success', 'fa-solid fa-circle-check');
                updateStepStatus('step-predict', 'active-step', 'fa-solid fa-spinner fa-spin');
                await sleep(500);
                updateStepStatus('step-predict', 'success', 'fa-solid fa-circle-check');
                updateStepStatus('step-shap', 'active-step', 'fa-solid fa-spinner fa-spin');
                await sleep(500);
                updateStepStatus('step-shap', 'success', 'fa-solid fa-circle-check');
                updateStepStatus('step-recommend', 'active-step', 'fa-solid fa-spinner fa-spin');
                await sleep(500);
                updateStepStatus('step-recommend', 'success', 'fa-solid fa-circle-check');
                spinner.style.display = 'none';
                successActions.style.display = 'block';
                isApiOnline = true;

                // Show file uploaded confirmation
                if (selectedFileName) selectedFileName.textContent = 'File Uploaded: Demo Dataset';
                if (dropZone) dropZone.style.display = 'none';
                if (fileSelectedRow) fileSelectedRow.style.display = 'flex';

                applyAnalysisResult(data);

                // Automatically navigate to Strategic Dashboard
                setTimeout(() => {
                    const dashNav = document.getElementById('nav-dashboard');
                    if (dashNav) dashNav.click();
                }, 300);

            } catch (err) {
                console.error("Demo load error:", err);
                showError(err.message || "Could not load demo data. Is the Flask server running?");
                spinner.style.display = 'none';
            } finally {
                btnDemo.disabled = false;
            }
        });
    }
}

// Shared: apply API response data to KPIs, charts, table and selectors
function applyAnalysisResult(data) {
    document.getElementById('kpi-total-members').textContent = data.summary.total_members.toLocaleString();
    const dist = data.summary.risk_distribution;
    const highRiskCount = (dist.High || 0) + (dist.Critical || 0);
    document.getElementById('kpi-high-risk').textContent = highRiskCount.toLocaleString();
    document.getElementById('kpi-avg-risk').textContent = (data.summary.average_risk * 100).toFixed(1) + '%';
    document.getElementById('kpi-opportunities').textContent = ((dist.Medium || 0) + (dist.High || 0) + (dist.Critical || 0)).toLocaleString();

    renderRiskDistChart(dist);
    renderRiskTrendChart(data.summary.tenure_risk_trend);
    renderDashboardDriversChart(data.summary.top_risk_drivers);
    populateUploadedSelectors(data.members);
    populateHighRiskTable(data.members);
}

function updateStepStatus(id, stepClass, iconClass) {
    const el = document.getElementById(id);
    if (el) {
        el.className = `progress-step ${stepClass}`;
        const icon = el.querySelector('i');
        if (icon) icon.className = iconClass;
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Populate selectors with the uploaded dataset
function populateUploadedSelectors(members) {
    const simulatorSelector = document.getElementById('member-selector');
    const detailSelector = document.getElementById('detail-member-selector');

    // Sort member IDs naturally
    members.sort((a, b) => a.member_id.localeCompare(b.member_id, undefined, {numeric: true, sensitivity: 'base'}));

    const simulatorOpts = ['<option value="">-- Load Simulator Form --</option>'];
    const detailOpts = ['<option value="">-- Choose Member ID --</option>'];

    members.forEach(m => {
        const text = `${m.member_id} (${(m.churn_probability * 100).toFixed(0)}% - ${m.risk_level})`;
        simulatorOpts.push(`<option value="${m.member_id}">${text}</option>`);
        detailOpts.push(`<option value="${m.member_id}">${text}</option>`);
    });

    if (simulatorSelector) simulatorSelector.innerHTML = simulatorOpts.join('');
    if (detailSelector) detailSelector.innerHTML = detailOpts.join('');
}

// Populate the High-Risk members table
function populateHighRiskTable(members) {
    const tbody = document.getElementById('high-risk-table-body');
    if (!tbody) return;
    tbody.innerHTML = '';

    // Filter to High and Critical risk levels, then sort by probability desc
    const highRiskMembers = members.filter(m => m.risk_level === 'High' || m.risk_level === 'Critical');
    highRiskMembers.sort((a, b) => b.churn_probability - a.churn_probability);

    if (highRiskMembers.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="7" style="padding: 20px; text-align: center; color: var(--text-muted);">
                    No High or Critical Risk members found in the uploaded dataset.
                </td>
            </tr>
        `;
        return;
    }

    highRiskMembers.forEach(m => {
        const tr = document.createElement('tr');
        const probText = (m.churn_probability * 100).toFixed(1) + '%';
        const driverText = m.primary_driver || 'N/A';

        tr.innerHTML = `
            <td style="padding: 12px; font-weight: bold; color: white;">${m.member_id}</td>
            <td style="padding: 12px;">${m.plan}</td>
            <td style="padding: 12px;">${m.tenure} m</td>
            <td style="padding: 12px; font-weight: 500;">${probText}</td>
            <td style="padding: 12px;"><span class="node-tag ${m.risk_level === 'Critical' ? 'critical' : 'churn'}" style="padding: 2px 6px; font-size: 0.65rem;">${m.risk_level.toUpperCase()}</span></td>
            <td style="padding: 12px; font-size: 0.85rem; color: var(--text-secondary);">${driverText}</td>
            <td style="padding: 12px; text-align: center;">
                <button type="button" class="btn-view-profile" onclick="navigateToMemberProfile('${m.member_id}')">
                    View Profile
                </button>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

// Function to handle "View Profile" click in dashboard table
function navigateToMemberProfile(memberId) {
    // Switch to Member Details tab
    const navDetails = document.getElementById('nav-member-details');
    if (navDetails) navDetails.click();
    
    // Select the member in the dropdown
    const selector = document.getElementById('detail-member-selector');
    if (selector) {
        selector.value = memberId;
        // Trigger change event to load details
        const event = new Event('change');
        selector.dispatchEvent(event);
    }
}

// Initialize Member Details Tab Logic
function initMemberDetailsTab() {
    const selector = document.getElementById('detail-member-selector');
    const resultsArea = document.getElementById('member-details-results-area');
    const fallbackArea = document.getElementById('member-details-fallback-area');

    if (!selector) return;

    selector.addEventListener('change', async (e) => {
        const memberId = e.target.value;
        if (!memberId) {
            if (resultsArea) resultsArea.style.display = 'none';
            if (fallbackArea) fallbackArea.style.display = 'block';
            return;
        }

        try {
            const res = await fetch(`${API_URL}/member/${memberId}`);
            const data = await res.json();

            // Populate summary cards
            document.getElementById('detail-id').textContent = data.member_info.PatientID;
            document.getElementById('detail-plan').textContent = data.member_info.Insurance_Type;
            document.getElementById('detail-tenure').textContent = `${data.member_info.Tenure_Months} months`;
            
            const probPct = (data.churn_probability * 100).toFixed(1) + '%';
            document.getElementById('detail-churn-prob').textContent = `${probPct} (${data.risk_level})`;
            
            // Style the risk card border and text
            const riskCard = document.getElementById('detail-risk-lvl-card');
            const riskIcon = document.getElementById('detail-risk-icon');
            
            let riskColor = 'var(--risk-low)';
            let riskIconHtml = '<i class="fa-solid fa-circle-check"></i>';
            if (data.risk_level === 'Medium') {
                riskColor = 'var(--risk-medium)';
                riskIconHtml = '<i class="fa-solid fa-circle-exclamation"></i>';
            } else if (data.risk_level === 'High') {
                riskColor = 'var(--risk-high)';
                riskIconHtml = '<i class="fa-solid fa-triangle-exclamation"></i>';
            } else if (data.risk_level === 'Critical') {
                riskColor = '#e11d48';
                riskIconHtml = '<i class="fa-solid fa-radiation"></i>';
            }
            
            if (riskCard) riskCard.style.borderLeft = `4px solid ${riskColor}`;
            if (riskIcon) {
                riskIcon.innerHTML = riskIconHtml;
                riskIcon.style.color = riskColor;
            }

            // Populate explanations list
            const listEl = document.getElementById('detail-drivers-text-list');
            if (listEl) {
                listEl.innerHTML = '';
                data.drivers.forEach(d => {
                    const li = document.createElement('li');
                    li.style.marginBottom = '12px';
                    li.style.display = 'flex';
                    li.style.alignItems = 'start';
                    li.style.gap = '8px';
                    
                    const icon = d.direction === 'increases_risk' ? 
                        '<i class="fa-solid fa-arrow-trend-up" style="color:var(--risk-high); margin-top:3px;"></i>' : 
                        '<i class="fa-solid fa-arrow-trend-down" style="color:var(--risk-low); margin-top:3px;"></i>';
                    
                    li.innerHTML = `${icon} <span><strong>${d.label}:</strong> ${d.value} &mdash; ${d.impact}</span>`;
                    listEl.appendChild(li);
                });
            }

            // Populate recommendations list
            populateRecommendationsList(data.recommendations, data.churn_probability, data.risk_level);

            // Populate SHAP chart
            renderDetailShapChart(data.drivers);

            if (fallbackArea) fallbackArea.style.display = 'none';
            if (resultsArea) resultsArea.style.display = 'block';

        } catch (err) {
            console.error("Error loading member details:", err);
            if (resultsArea) resultsArea.style.display = 'none';
            if (fallbackArea) fallbackArea.style.display = 'block';
        }
    });
}

function populateRecommendationsList(recommendations, prob, riskLevel) {
    const summaryEl = document.getElementById('detail-advisory-summary');
    const listEl = document.getElementById('detail-recommendations-list');
    if (!listEl) return;
    listEl.innerHTML = '';
    
    let riskClass = riskLevel.toLowerCase() + '-risk';
    if (riskLevel === 'Critical') riskClass = 'critical-risk';
    let riskLabel = riskLevel.toUpperCase();
    let summaryText = 'Standard routine monitoring and preventive care workflows recommended.';
    
    if (riskLevel === 'Medium') {
        summaryText = 'Member is at Moderate Risk. Targeted operational and outreach workflows recommended below.';
    } else if (riskLevel === 'High' || riskLevel === 'Critical') {
        summaryText = 'Immediate proactive interventions from administrative concierges are vital to address drivers.';
    }
    
    if (summaryEl) {
        summaryEl.className = `advisory-summary ${riskClass}`;
        summaryEl.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Churn Risk Profile: <strong>${riskLabel} (${(prob * 100).toFixed(1)}%)</strong><br><span style="font-size:0.8rem; font-weight:normal; color:var(--text-secondary);">${summaryText}</span>`;
    }
    
    if (recommendations.length === 0) {
        listEl.innerHTML = '<div style="color:var(--text-secondary); font-size:0.85rem; padding:10px; text-align:center;">No interventions generated for this profile.</div>';
        return;
    }
    
    recommendations.forEach(item => {
        const itemEl = document.createElement('div');
        const priorityClass = item.priority.toLowerCase();
        itemEl.className = `intervention-item ${priorityClass}`;
        
        let icon = 'fa-solid fa-circle-info';
        if (item.category === 'Service Recovery') icon = 'fa-solid fa-file-invoice-dollar';
        else if (item.category === 'Care Outreach') icon = 'fa-solid fa-calendar-times';
        else if (item.category === 'Benefit Education') icon = 'fa-solid fa-laptop-medical';
        else if (item.category === 'Pharmacy Support') icon = 'fa-solid fa-scale-balanced';
        
        itemEl.innerHTML = `
            <div class="intervention-icon">
                <i class="${icon}"></i>
            </div>
            <div class="intervention-details">
                <h5>${item.action} <span class="node-tag ${priorityClass === 'critical' || priorityClass === 'high' ? 'churn' : (priorityClass === 'medium' ? 'medium-text' : 'safe')}" style="margin-top:0; font-size:0.55rem; padding: 1px 4px;">${item.priority.toUpperCase()}</span></h5>
                <p><strong>Reason:</strong> ${item.reason}</p>
                <p style="font-size:0.75rem; color:var(--text-secondary); margin-top:4px;"><strong>Evidence:</strong> ${item.evidence.join('; ')}</p>
                <p style="font-size:0.75rem; color:var(--text-secondary); margin-top:2px;"><strong>Next Step:</strong> ${item.suggested_next_step}</p>
            </div>
        `;
        listEl.appendChild(itemEl);
    });
}

function renderDetailShapChart(drivers) {
    const canvas = document.getElementById('detail-shap-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (detailShapChart) detailShapChart.destroy();

    const labels = drivers.map(d => d.label);
    const values = drivers.map(d => d.shap_value);
    const colors = values.map(val => val > 0 ? 'rgba(244, 63, 94, 0.7)' : 'rgba(16, 185, 129, 0.7)');

    detailShapChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'SHAP Contribution Value',
                data: values,
                backgroundColor: colors,
                borderColor: values.map(v => v > 0 ? 'var(--risk-high)' : 'var(--risk-low)'),
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'Negative Impact (Reduces Risk) ◀  ▶ Positive Impact (Increases Risk)', color: '#9ca3af', font: { size: 10 } },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9ca3af' }
                },
                y: { grid: { display: false }, ticks: { color: '#f3f4f6', font: { size: 11, weight: '500' } } }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

