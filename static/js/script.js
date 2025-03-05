document.addEventListener('DOMContentLoaded', function() {
    const transactionFeatures = [
        { id: 'transaction_amount', label: 'v1', type: 'number', step: '0.01', min: '0' },
        { id: 'transaction_time', label: 'v2', type: 'number', step: '0.01', min: '0', max: '24' },
        { id: 'distance_from_home', label: 'Distance from Home (miles)', type: 'number', step: '0.1', min: '0' },
        { id: 'ratio_to_median_purchase', label: 'Ratio to Median Purchase', type: 'number', step: '0.01', min: '0' },
        { id: 'repeat_retailer', label: 'Repeat Retailer', type: 'checkbox' },
        { id: 'used_chip', label: 'Used Chip', type: 'checkbox' },
        { id: 'used_pin_number', label: 'Used PIN Number', type: 'checkbox' },
        { id: 'online_order', label: 'Online Order', type: 'checkbox' },
        { id: 'fraud_history', label: 'Previous Fraud History', type: 'checkbox' },
        { id: 'transaction_freq_24h', label: 'Transactions in Last 24h', type: 'number', step: '1', min: '0' },
        { id: 'transaction_freq_7d', label: 'Transactions in Last 7 Days', type: 'number', step: '1', min: '0' },
        { id: 'daily_rate_compared_to_avg', label: 'Daily Rate vs Average', type: 'number', step: '0.01', min: '0' },
        { id: 'medium_purchase_price', label: 'Medium Purchase Price ($)', type: 'number', step: '0.01', min: '0' },
        { id: 'purchase_time_variance', label: 'Purchase Time Variance (hours)', type: 'number', step: '0.1', min: '0' },
        { id: 'transaction_type', label: 'Transaction Type', type: 'select', options: [
            { value: '0', label: 'In-store' },
            { value: '1', label: 'Online' },
            { value: '2', label: 'ATM' },
            { value: '3', label: 'Other' }
        ]},
        { id: 'customer_age', label: 'Customer Age', type: 'number', step: '1', min: '18', max: '100' },
        { id: 'merchant_category', label: 'Merchant Category', type: 'select', options: [
            { value: '0', label: 'Retail' },
            { value: '1', label: 'Travel' },
            { value: '2', label: 'Entertainment' },
            { value: '3', label: 'Groceries' },
            { value: '4', label: 'Restaurant' },
            { value: '5', label: 'Services' },
            { value: '6', label: 'Healthcare' },
            { value: '7', label: 'Gas/Automotive' },
            { value: '8', label: 'Electronics' },
            { value: '9', label: 'Other' }
        ]},
        { id: 'merchant_rating', label: 'Merchant Rating', type: 'number', step: '0.1', min: '1', max: '5' },
        { id: 'num_declined_24h', label: 'Declined Transactions (24h)', type: 'number', step: '1', min: '0' },
        { id: 'num_declined_7d', label: 'Declined Transactions (7d)', type: 'number', step: '1', min: '0' },
        { id: 'foreign_transaction', label: 'Foreign Transaction', type: 'checkbox' },
        { id: 'high_risk_country', label: 'High Risk Country', type: 'checkbox' },
        { id: 'high_risk_email', label: 'High Risk Email', type: 'checkbox' },
        { id: 'high_risk_ip', label: 'High Risk IP', type: 'checkbox' },
        { id: 'high_risk_device', label: 'High Risk Device', type: 'checkbox' },
        { id: 'device_change', label: 'Recent Device Change', type: 'checkbox' },
        { id: 'location_change', label: 'Location Change', type: 'checkbox' },
        { id: 'card_present', label: 'Card Present', type: 'checkbox' },
        { id: 'cvv_provided', label: 'CVV Provided', type: 'checkbox' },
        { id: 'billing_shipping_mismatch', label: 'Billing/Shipping Mismatch', type: 'checkbox' }
    ];

    // Generate form fields
    const featuresDiv = document.getElementById('features');
    featuresDiv.innerHTML = ''; // Clear existing fields
    
    transactionFeatures.forEach(feature => {
        const col = document.createElement('div');
        col.className = feature.type === 'checkbox' ? 'col-md-4 mb-3' : 'col-md-6 mb-3';
        
        let input;
        if (feature.type === 'select') {
            input = `<select class="form-select" id="${feature.id}" required>
                ${feature.options.map(opt => 
                    `<option value="${opt.value}">${opt.label}</option>`
                ).join('')}
            </select>`;
        } else if (feature.type === 'checkbox') {
            input = `<div class="form-check">
                <input type="checkbox" class="form-check-input" id="${feature.id}">
            </div>`;
        } else {
            input = `<input type="${feature.type}" class="form-control" id="${feature.id}" 
                     ${feature.step ? `step="${feature.step}"` : ''} 
                     ${feature.min ? `min="${feature.min}"` : ''} 
                     ${feature.max ? `max="${feature.max}"` : ''} required>`;
        }
        
        col.innerHTML = `
            <label for="${feature.id}" class="form-label">${feature.label}</label>
            ${input}
        `;
        featuresDiv.appendChild(col);
    });

    // Handle form submission
    document.getElementById('transactionForm').addEventListener('submit', async function(e) {
        e.preventDefault();

        // Collect form data
        const formData = {};
        transactionFeatures.forEach(feature => {
            const element = document.getElementById(feature.id);
            if (feature.type === 'checkbox') {
                formData[feature.id] = element.checked;
            } else if (feature.type === 'select') {
                // Get the label instead of the value for select elements
                const selectedOption = element.options[element.selectedIndex];
                formData[feature.id] = selectedOption.label;
            } else {
                formData[feature.id] = parseFloat(element.value) || 0;
            }
        });

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();

            if (result.success) {
                displayResults(result);
            } else {
                alert('Error: ' + (result.error || 'Unknown error occurred'));
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Error submitting form: ' + error.message);
        }
    });
});

function displayResults(result) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.style.display = 'block';
    
    const predictions = result.predictions;
    const rf = predictions.random_forest;
    const xgb = predictions.xgboost;
    const overall = predictions.overall_assessment;
    
    resultsDiv.innerHTML = `
        <div class="card">
            <div class="card-body">
                <h3 class="card-title mb-4">Transaction Analysis Results</h3>
                
                <div class="alert ${getAlertClass(overall.risk_level)} mb-4">
                    <h4 class="alert-heading">Risk Assessment: ${overall.risk_level}</h4>
                    <p class="mb-0">Verdict: ${overall.final_verdict}</p>
                    <hr>
                    <p class="mb-0">${overall.recommendation}</p>
                </div>

                <div class="row mb-4">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header bg-primary text-white">
                                <h5 class="mb-0">Transaction Details</h5>
                            </div>
                            <div class="card-body">
                                <p><strong>Amount:</strong> $${result.transaction_details.amount.toFixed(2)}</p>
                                <p><strong>Time:</strong> ${new Date(result.transaction_details.timestamp).toLocaleString()}</p>
                                <p><strong>Type:</strong> ${result.transaction_details.transaction_type}</p>
                                <p><strong>Merchant:</strong> ${result.transaction_details.merchant_category}</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header bg-info text-white">
                                <h5 class="mb-0">Risk Analysis</h5>
                            </div>
                            <div class="card-body">
                                <h6>Model Predictions</h6>
                                <div class="mb-3">
                                    <p class="mb-1">Random Forest</p>
                                    <div class="progress">
                                        <div class="progress-bar ${getProgressBarClass(rf.risk_score/100)}" 
                                             role="progressbar" 
                                             style="width: ${rf.risk_score}%">
                                            ${rf.risk_score.toFixed(1)}%
                                        </div>
                                    </div>
                                </div>
                                <div class="mb-3">
                                    <p class="mb-1">XGBoost</p>
                                    <div class="progress">
                                        <div class="progress-bar ${getProgressBarClass(xgb.risk_score/100)}" 
                                             role="progressbar" 
                                             style="width: ${xgb.risk_score}%">
                                            ${xgb.risk_score.toFixed(1)}%
                                        </div>
                                    </div>
                                </div>
                                <div class="mb-3">
                                    <p class="mb-1">Overall Risk Score</p>
                                    <div class="progress">
                                        <div class="progress-bar ${getProgressBarClass(overall.risk_score/100)}" 
                                             role="progressbar" 
                                             style="width: ${overall.risk_score}%">
                                            ${overall.risk_score.toFixed(1)}%
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                ${overall.risk_factors.length > 0 ? `
                    <div class="alert alert-danger mb-3">
                        <h5 class="alert-heading">High Risk Factors</h5>
                        <ul class="mb-0">
                            ${overall.risk_factors.map(factor => `<li>${factor}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}

                ${overall.suspicious_factors.length > 0 ? `
                    <div class="alert alert-warning mb-3">
                        <h5 class="alert-heading">Suspicious Factors</h5>
                        <ul class="mb-0">
                            ${overall.suspicious_factors.map(factor => `<li>${factor}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
            </div>
        </div>
    `;
    
    // Scroll to results
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

function getAlertClass(riskLevel) {
    switch(riskLevel.toUpperCase()) {
        case 'HIGH':
            return 'alert-danger';
        case 'MEDIUM':
            return 'alert-warning';
        default:
            return 'alert-success';
    }
}

function getProgressBarClass(riskScore) {
    if (riskScore > 0.7) return 'bg-danger';
    if (riskScore > 0.3) return 'bg-warning';
    return 'bg-success';
}