// demo/frontend/static/js/app.js
const API_BASE = '';
let uploadQueue = [];
let isEvaluating = false;
let results = [];

// ============ 初始化 ============
document.addEventListener('DOMContentLoaded', function() {
    checkHealth();
    setupDropZone();
    setupFileInput();
});

// ============ 健康检查 ============
async function checkHealth() {
    const statusEl = document.getElementById('modelStatus');
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        if (data.model_loaded) {
            statusEl.innerHTML = `
                <i class="fas fa-check-circle" style="color: #48bb78;"></i>
                模型已加载: ${data.model_name || '未命名'}
                <span style="font-size: 0.8rem; opacity: 0.7;">(设备: ${data.device})</span>
            `;
            statusEl.className = 'model-status success';
        } else {
            statusEl.innerHTML = `
                <i class="fas fa-exclamation-triangle" style="color: #ed8936;"></i>
                模型未加载，请检查权重文件
            `;
            statusEl.className = 'model-status warning';
        }
    } catch (error) {
        statusEl.innerHTML = `
            <i class="fas fa-times-circle" style="color: #fc8181;"></i>
            服务连接失败
        `;
        statusEl.className = 'model-status error';
    }
}

// ============ 拖拽上传 ============
function setupDropZone() {
    const dropZone = document.getElementById('dropZone');
    
    ['dragenter', 'dragover'].forEach(event => {
        dropZone.addEventListener(event, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
    });
    
    ['dragleave', 'dragend'].forEach(event => {
        dropZone.addEventListener(event, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        handleFiles(files);
    });
}

function setupFileInput() {
    document.getElementById('fileInput').addEventListener('change', (e) => {
        handleFiles(e.target.files);
        e.target.value = ''; // 重置
    });
}

function handleFiles(files) {
    const videoFiles = Array.from(files).filter(f => f.type.startsWith('video/'));
    if (videoFiles.length === 0) {
        alert('请上传视频文件');
        return;
    }
    
    uploadQueue = [...uploadQueue, ...videoFiles];
    updateQueueUI();
    document.getElementById('queueSection').style.display = 'block';
}

// ============ 队列管理 ============
function updateQueueUI() {
    const queueList = document.getElementById('queueList');
    const queueCount = document.getElementById('queueCount');
    queueCount.textContent = uploadQueue.length;
    
    if (uploadQueue.length === 0) {
        queueList.innerHTML = '<div class="empty-queue">队列为空</div>';
        return;
    }
    
    queueList.innerHTML = uploadQueue.map((file, index) => `
        <div class="queue-item">
            <div class="queue-item-info">
                <span class="queue-index">${index + 1}</span>
                <span class="queue-name"><i class="fas fa-video"></i> ${file.name}</span>
                <span class="queue-size">(${(file.size / 1024 / 1024).toFixed(2)} MB)</span>
            </div>
            <button onclick="removeFromQueue(${index})" class="btn-icon" title="移除">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `).join('');
}

function removeFromQueue(index) {
    uploadQueue.splice(index, 1);
    updateQueueUI();
    if (uploadQueue.length === 0) {
        document.getElementById('queueSection').style.display = 'none';
    }
}

function clearQueue() {
    if (confirm('确定要清空队列吗？')) {
        uploadQueue = [];
        updateQueueUI();
        document.getElementById('queueSection').style.display = 'none';
    }
}

// ============ 评估 ============
async function evaluateAll() {
    if (isEvaluating || uploadQueue.length === 0) return;
    
    isEvaluating = true;
    const btn = document.getElementById('batchEvalBtn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 评估中...';
    
    const resultsGrid = document.getElementById('resultsGrid');
    resultsGrid.innerHTML = '';
    document.getElementById('resultsSection').style.display = 'block';
    
    const batchResults = [];
    const total = uploadQueue.length;
    
    for (let i = 0; i < total; i++) {
        const file = uploadQueue[i];
        
        // 创建结果卡片
        const card = createResultCard(file, i, total);
        resultsGrid.appendChild(card);
        
        // 上传并评估
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            // 更新卡片
            updateResultCard(card, data);
            batchResults.push(data);
            
        } catch (error) {
            updateResultCard(card, null, error.message);
        }
    }
    
    // 更新统计
    updateStatistics(batchResults);
    
    isEvaluating = false;
    btn.disabled = false;
    btn.innerHTML = '<i class="fas fa-play"></i> 开始评估';
    
    // 清空队列
    uploadQueue = [];
    updateQueueUI();
    document.getElementById('queueSection').style.display = 'none';
}

function createResultCard(file, index, total) {
    const card = document.createElement('div');
    card.className = 'result-card processing';
    card.innerHTML = `
        <div class="result-header">
            <div class="result-title">
                <span class="result-number">#${index + 1}/${total}</span>
                <span class="result-name">${file.name}</span>
            </div>
            <span class="result-status"><i class="fas fa-spinner fa-spin"></i> 处理中...</span>
        </div>
        <div class="result-body">
            <div class="progress-bar">
                <div class="progress-fill" style="width: 0%"></div>
            </div>
        </div>
    `;
    return card;
}

function updateResultCard(card, data, error) {
    card.className = 'result-card';
    
    if (error) {
        card.classList.add('error');
        card.querySelector('.result-status').innerHTML = `
            <span class="badge-error"><i class="fas fa-times"></i> 失败</span>
        `;
        card.querySelector('.result-body').innerHTML = `
            <div class="error-msg"><i class="fas fa-exclamation-circle"></i> ${error}</div>
        `;
        return;
    }
    
    if (data.status === 'success') {
        card.classList.add('completed');
        const score = data.mos_score;
        
        card.querySelector('.result-status').innerHTML = `
            <span class="mos-score">${score.toFixed(2)}</span>
        `;
        
        card.querySelector('.result-body').innerHTML = `
            <div class="score-detail">
                <div class="score-bar-container">
                    <div class="score-bar">
                        <div class="score-fill" style="width: ${(score / 5) * 100}%"></div>
                    </div>
                    <span class="score-label">${score.toFixed(2)} / 5.0</span>
                </div>
                <div class="score-meta">
                    <span><i class="fas fa-layer-group"></i> ${data.num_frames || 8} 帧</span>
                    <span><i class="fas fa-microchip"></i> ${data.model_name || 'model'}</span>
                </div>
            </div>
        `;
    } else {
        card.classList.add('error');
        card.querySelector('.result-status').innerHTML = `
            <span class="badge-error"><i class="fas fa-times"></i> 失败</span>
        `;
        card.querySelector('.result-body').innerHTML = `
            <div class="error-msg"><i class="fas fa-exclamation-circle"></i> ${data.message || '评估失败'}</div>
        `;
    }
}

// ============ 统计 ============
function updateStatistics(batchResults) {
    const validResults = batchResults.filter(r => r && r.status === 'success' && r.mos_score !== undefined);
    
    if (validResults.length === 0) {
        document.getElementById('statCount').textContent = '0';
        document.getElementById('statAvg').textContent = '-';
        document.getElementById('statMax').textContent = '-';
        document.getElementById('statMin').textContent = '-';
        return;
    }
    
    const scores = validResults.map(r => r.mos_score);
    const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
    const max = Math.max(...scores);
    const min = Math.min(...scores);
    
    document.getElementById('statCount').textContent = validResults.length;
    document.getElementById('statAvg').textContent = avg.toFixed(2);
    document.getElementById('statMax').textContent = max.toFixed(2);
    document.getElementById('statMin').textContent = min.toFixed(2);
}

function clearResults() {
    if (confirm('确定要清空所有结果吗？')) {
        document.getElementById('resultsGrid').innerHTML = '';
        document.getElementById('resultsSection').style.display = 'none';
        results = [];
    }
}

// ============ 工具函数 ============
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1024 / 1024).toFixed(1) + ' MB';
}