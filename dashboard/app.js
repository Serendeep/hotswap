// Dashboard JavaScript

const API_BASE = '/api';
let ws = null;
let reconnectTimeout = null;
let isIntentionalClose = false;
let isConnecting = false;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    refreshAll();
    setInterval(refreshAll, 5000); // Refresh every 5 seconds
});

// WebSocket connection
function connectWebSocket() {
    // Prevent multiple simultaneous connection attempts
    if (isConnecting) {
        return;
    }

    // Don't reconnect if we already have a good connection
    if (ws && ws.readyState === WebSocket.OPEN) {
        return;
    }

    // Clean up any existing connection
    if (ws) {
        isIntentionalClose = true;
        if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
            ws.close();
        }
        ws = null;
    }

    isConnecting = true;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}${API_BASE}/ws`;

    try {
        ws = new WebSocket(wsUrl);
    } catch (e) {
        console.error('Failed to create WebSocket:', e);
        isConnecting = false;
        scheduleReconnect();
        return;
    }

    ws.onopen = () => {
        console.log('WebSocket connected');
        isConnecting = false;
        isIntentionalClose = false;
        updateConnectionStatus(true);
        cancelReconnect();
    };

    ws.onclose = () => {
        isConnecting = false;
        // Only reconnect if this wasn't an intentional close
        if (!isIntentionalClose) {
            console.log('WebSocket disconnected');
            updateConnectionStatus(false);
            scheduleReconnect();
        }
        isIntentionalClose = false;
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        isConnecting = false;
    };

    ws.onmessage = (event) => {
        try {
            const message = JSON.parse(event.data);
            handleWebSocketMessage(message);
        } catch (e) {
            console.error('Failed to parse WebSocket message:', e);
        }
    };
}

function scheduleReconnect() {
    // Use timeout instead of interval to prevent stacking
    if (!reconnectTimeout) {
        reconnectTimeout = setTimeout(() => {
            reconnectTimeout = null;
            console.log('Attempting to reconnect WebSocket...');
            connectWebSocket();
        }, 5000);
    }
}

function cancelReconnect() {
    if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
        reconnectTimeout = null;
    }
}

function updateConnectionStatus(connected) {
    const status = document.getElementById('connection-status');
    if (connected) {
        status.classList.add('connected');
        status.classList.remove('disconnected');
        status.querySelector('.text').textContent = 'Connected';
    } else {
        status.classList.remove('connected');
        status.classList.add('disconnected');
        status.querySelector('.text').textContent = 'Disconnected';
    }
}

function handleWebSocketMessage(message) {
    const { event, data } = message;

    switch (event) {
        case 'status':
            updateEngineStatus(data.engine);
            updateSwapStatus(data.swap);
            break;
        case 'training_progress':
            updateTrainingStatus(data);
            break;
        case 'training_complete':
            updateTrainingStatus(data);
            refreshModels();
            break;
        case 'model_swap':
            refreshAll();
            break;
    }
}

// API Functions
async function fetchJson(url, options = {}) {
    try {
        const response = await fetch(url, options);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`API error for ${url}:`, error);
        throw error;
    }
}

async function refreshAll() {
    try {
        const status = await fetchJson(`${API_BASE}/status`);
        updateEngineStatus(status.engine);
        updateSwapStatus(status.swap_coordinator);
        if (status.training) {
            updateTrainingStatus(status.training);
        } else {
            clearTrainingStatus();
        }
    } catch (e) {
        console.error('Failed to refresh status:', e);
    }

    refreshModels();
}

async function refreshModels() {
    try {
        const models = await fetchJson(`${API_BASE}/models`);
        updateModelsTable(models);
        updateModelSelect(models);
    } catch (e) {
        console.error('Failed to refresh models:', e);
    }
}

// Update Functions
function updateEngineStatus(engine) {
    if (!engine) return;

    // Active model
    const activeVersion = document.getElementById('active-version');
    const activeId = document.getElementById('active-id');
    const predictionCount = document.getElementById('prediction-count');

    if (engine.has_active_model) {
        activeVersion.textContent = `v${engine.active_model_version || '--'}`;
        activeId.textContent = engine.active_model_id || 'Unknown';
    } else {
        activeVersion.textContent = '--';
        activeId.textContent = 'No active model';
    }
    predictionCount.textContent = engine.prediction_count || 0;

    // Shadow model
    const shadowCard = document.getElementById('shadow-model-card');
    const shadowVersion = document.getElementById('shadow-version');
    const shadowId = document.getElementById('shadow-id');
    const shadowControls = document.getElementById('shadow-controls');
    const shadowMetrics = document.getElementById('shadow-metrics-section');
    const agreementRate = document.getElementById('agreement-rate');

    if (engine.shadow_enabled && engine.has_shadow_model) {
        shadowCard.classList.add('active');
        shadowVersion.textContent = `v${engine.shadow_model_version || '--'}`;
        shadowId.textContent = engine.shadow_model_id || 'Unknown';
        shadowControls.style.display = 'flex';
        shadowMetrics.style.display = 'block';

        if (engine.shadow_metrics) {
            updateShadowMetrics(engine.shadow_metrics);
        }
    } else {
        shadowCard.classList.remove('active');
        shadowVersion.textContent = '--';
        shadowId.textContent = 'No shadow model';
        agreementRate.textContent = '--';
        shadowControls.style.display = 'none';
        shadowMetrics.style.display = 'none';
    }
}

function updateSwapStatus(swap) {
    if (!swap) return;

    // Update shadow metrics if in shadow mode
    if (swap.state === 'shadow_mode' && swap.metrics) {
        updateShadowMetrics(swap.metrics);

        const decisionText = document.getElementById('decision-text');
        decisionText.textContent = swap.decision || '--';
        decisionText.className = `decision ${swap.decision || ''}`;
    }
}

function updateShadowMetrics(metrics) {
    document.getElementById('metric-samples').textContent = metrics.total_samples || 0;
    document.getElementById('metric-agreement').textContent =
        `${((metrics.agreement_rate || 0) * 100).toFixed(1)}%`;
    document.getElementById('metric-latency-active').textContent =
        `${(metrics.avg_active_latency_ms || 0).toFixed(2)}ms`;
    document.getElementById('metric-latency-shadow').textContent =
        `${(metrics.avg_shadow_latency_ms || 0).toFixed(2)}ms`;
    document.getElementById('agreement-rate').textContent =
        `${((metrics.agreement_rate || 0) * 100).toFixed(1)}%`;
}

function updateTrainingStatus(training) {
    const card = document.getElementById('training-card');
    const status = document.getElementById('training-status');
    const progressContainer = document.getElementById('training-progress-container');
    const progressBar = document.getElementById('training-progress');
    const details = document.getElementById('training-details');

    if (training.status === 'running') {
        card.classList.add('active');
        status.textContent = 'Training...';
        progressContainer.style.display = 'block';
        progressBar.style.width = `${(training.progress || 0) * 100}%`;

        let detailText = `Epoch ${training.current_epoch}/${training.total_epochs}`;
        if (training.current_loss !== null) {
            detailText += ` | Loss: ${training.current_loss.toFixed(4)}`;
        }
        details.textContent = detailText;
    } else if (training.status === 'completed') {
        card.classList.remove('active');
        status.textContent = 'Complete';
        progressContainer.style.display = 'none';
        details.textContent = `Model ${training.model_id} trained`;
    } else if (training.status === 'failed') {
        card.classList.remove('active');
        status.textContent = 'Failed';
        progressContainer.style.display = 'none';
        details.textContent = training.error || 'Unknown error';
    }
}

function clearTrainingStatus() {
    const card = document.getElementById('training-card');
    const status = document.getElementById('training-status');
    const progressContainer = document.getElementById('training-progress-container');
    const details = document.getElementById('training-details');

    card.classList.remove('active');
    status.textContent = 'Idle';
    progressContainer.style.display = 'none';
    details.textContent = '';
}

function updateModelsTable(models) {
    const tbody = document.getElementById('models-tbody');

    if (!models || models.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="loading">No models found</td></tr>';
        return;
    }

    tbody.innerHTML = models.map(model => {
        const loss = model.metrics?.final_loss;
        const lossText = loss !== undefined ? loss.toFixed(4) : '--';
        const date = new Date(model.created_at).toLocaleString();

        return `
            <tr>
                <td>v${model.version}</td>
                <td><span class="status-badge ${model.status}">${model.status}</span></td>
                <td>${date}</td>
                <td>${lossText}</td>
                <td>
                    ${model.status === 'ready' || model.status === 'archived' ?
                        `<button class="btn btn-sm btn-secondary" onclick="activateModel('${model.id}')">Activate</button>` :
                        ''
                    }
                </td>
            </tr>
        `;
    }).join('');
}

function updateModelSelect(models) {
    const select = document.getElementById('shadow-model-select');
    const currentValue = select.value;

    // Filter to only ready models
    const readyModels = models.filter(m => m.status === 'ready');

    select.innerHTML = '<option value="">Select a model...</option>' +
        readyModels.map(model =>
            `<option value="${model.id}">v${model.version} - ${model.id.slice(0, 8)}...</option>`
        ).join('');

    // Restore selection if still valid
    if (currentValue && readyModels.some(m => m.id === currentValue)) {
        select.value = currentValue;
    }
}

// Action Functions
async function startShadow() {
    const select = document.getElementById('shadow-model-select');
    const modelId = select.value;

    if (!modelId) {
        alert('Please select a model');
        return;
    }

    try {
        await fetchJson(`${API_BASE}/shadow/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId })
        });
        refreshAll();
    } catch (e) {
        alert('Failed to start shadow mode: ' + e.message);
    }
}

async function promoteShadow() {
    if (!confirm('Promote shadow model to active?')) return;

    try {
        await fetchJson(`${API_BASE}/shadow/promote`, { method: 'POST' });
        refreshAll();
    } catch (e) {
        alert('Failed to promote shadow: ' + e.message);
    }
}

async function cancelShadow() {
    if (!confirm('Cancel shadow mode?')) return;

    try {
        await fetchJson(`${API_BASE}/shadow/cancel`, { method: 'POST' });
        refreshAll();
    } catch (e) {
        alert('Failed to cancel shadow: ' + e.message);
    }
}

async function activateModel(modelId) {
    if (!confirm('Activate this model?')) return;

    try {
        await fetchJson(`${API_BASE}/models/${modelId}/activate`, { method: 'POST' });
        refreshAll();
    } catch (e) {
        alert('Failed to activate model: ' + e.message);
    }
}
