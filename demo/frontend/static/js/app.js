// demo/frontend/static/js/app.js —— 多模型版
const API_BASE = '';
let uploadQueue = [];
let isEvaluating = false;
let models = [];
let selectedModel = null;

// ============ 初始化 ============
document.addEventListener('DOMContentLoaded', function () {
    loadModels();
    setupDropZone();
    setupFileInput();
});

async function loadModels() {
    try {
        const r = await fetch('/api/models');
        const data = await r.json();
        models = data.models;
        renderModelCards();
        renderScoreTxtSelect();
        renderModelInfo();
        if (models.length) selectModel(models.find(m => m.available) || models[0]);
    } catch (e) {
        document.getElementById('modelCards').innerHTML =
            '<div class="empty-hint">模型列表加载失败：' + e.message + '</div>';
        document.getElementById('modelStatus').innerHTML =
            '<i class="fas fa-times-circle" style="color:#fc8181;"></i> 服务连接失败';
    }
}

function renderModelCards() {
    const wrap = document.getElementById('modelCards');
    wrap.innerHTML = models.map(m => `
        <div class="model-card ${m.id === (selectedModel && selectedModel.id) ? 'selected' : ''}"
             onclick="selectModelById('${m.id}')">
            <div class="model-card-head">
                <i class="fas ${m.available ? 'fa-check-circle' : 'fa-hourglass-half'}"
                   style="color:${m.available ? '#48bb78' : '#ed8936'};"></i>
                <span class="model-card-name">${m.name}</span>
            </div>
            <div class="model-card-desc">${m.desc}</div>
            <div class="model-card-meta">
                <span class="model-card-scale">量纲：${m.scale}</span>
                ${m.loaded ? '<span class="model-card-loaded">已加载</span>' : ''}
                ${!m.available ? '<span class="model-card-wait">权重待训练完成</span>' : ''}
            </div>
        </div>
    `).join('');
}

function selectModelById(id) {
    const m = models.find(x => x.id === id);
    if (m && m.available) { selectModel(m); renderModelCards(); }
}

function selectModel(m) {
    selectedModel = m;
    document.getElementById('modelStatus').innerHTML = `
        <i class="fas fa-check-circle" style="color:#48bb78;"></i>
        当前模型：${m.name}（量纲 ${m.scale}）
    `;
    document.getElementById('modelStatus').className = 'model-status success';
}

function renderScoreTxtSelect() {
    const sel = document.getElementById('scoreTxtModel');
    sel.innerHTML = models.map(m =>
        `<option value="${m.id}">${m.name}</option>`).join('');
}

function renderModelInfo() {
    document.getElementById('modelInfoDetail').innerHTML = models.map(m => `
        <div class="model-info-item">
            <h4>${m.name}（量纲 ${m.scale}${m.available ? '' : ' · 训练中'}）</h4>
            <p>${m.desc}</p>
        </div>
    `).join('');
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
        handleFiles(e.dataTransfer.files);
    });
}

function setupFileInput() {
    document.getElementById('fileInput').addEventListener('change', (e) => {
        handleFiles(e.target.files);
        e.target.value = '';
    });
}

function handleFiles(files) {
    const videoFiles = Array.from(files).filter(f => f.type.startsWith('video/'));
    if (!videoFiles.length) { alert('请上传视频文件'); return; }
    uploadQueue = [...uploadQueue, ...videoFiles];
    updateQueueUI();
    document.getElementById('queueSection').style.display = 'block';
}

// ============ 队列管理 ============
function updateQueueUI() {
    const list = document.getElementById('queueList');
    document.getElementById('queueCount').textContent = uploadQueue.length;
    if (!uploadQueue.length) {
        list.innerHTML = '<div class="empty-queue">队列为空</div>';
        return;
    }
    list.innerHTML = uploadQueue.map((file, i) => `
        <div class="queue-item">
            <div class="queue-item-info">
                <span class="queue-index">${i + 1}</span>
                <span class="queue-name"><i class="fas fa-video"></i> ${file.name}</span>
                <span class="queue-size">(${(file.size / 1048576).toFixed(2)} MB)</span>
            </div>
            <button onclick="removeFromQueue(${i})" class="btn-icon" title="移除">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `).join('');
}

function removeFromQueue(index) {
    uploadQueue.splice(index, 1);
    updateQueueUI();
    if (!uploadQueue.length) document.getElementById('queueSection').style.display = 'none';
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
    if (isEvaluating || !uploadQueue.length) return;
    if (!selectedModel || !selectedModel.available) { alert('请先选择可用的模型'); return; }

    isEvaluating = true;
    const btn = document.getElementById('batchEvalBtn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 评估中...';

    const grid = document.getElementById('resultsGrid');
    grid.innerHTML = '';
    document.getElementById('resultsSection').style.display = 'block';

    for (let i = 0; i < uploadQueue.length; i++) {
        const file = uploadQueue[i];
        const card = createResultCard(file, i);
        grid.appendChild(card);

        try {
            const formData = new FormData();
            formData.append('file', file);
            const r = await fetch(`/api/predict?model_id=${selectedModel.id}`,
                { method: 'POST', body: formData });
            const data = await r.json();
            updateResultCard(card, data);
        } catch (error) {
            updateResultCard(card, null, error.message);
        }
    }

    isEvaluating = false;
    btn.disabled = false;
    btn.innerHTML = '<i class="fas fa-play"></i> 开始评估';
    uploadQueue = [];
    updateQueueUI();
    document.getElementById('queueSection').style.display = 'none';
    loadScoreTxt();
}

function createResultCard(file, index) {
    const card = document.createElement('div');
    card.className = 'result-card processing';
    card.innerHTML = `
        <div class="result-header">
            <div class="result-title">
                <span class="result-number">#${index + 1}</span>
                <span class="result-name">${file.name}</span>
            </div>
            <span class="result-status"><i class="fas fa-spinner fa-spin"></i> 处理中...</span>
        </div>
        <div class="result-body"><div class="progress-bar"><div class="progress-fill"></div></div></div>
    `;
    return card;
}

function issuesHtml(issues) {
    if (!issues || !Object.keys(issues).length) return '';
    const names = ['闪烁', '噪点', '模糊'];
    const levelColor = { '无': '#48bb78', '轻': '#ecc94b', '中': '#ed8936', '重': '#fc8181', '未知': '#a0aec0' };
    return `<div class="issues-box">
        <div class="issues-title"><i class="fas fa-bug"></i> 失真问题反馈</div>
        <div class="issues-grid">
            ${Object.entries(issues).map(([name, it]) => `
                <div class="issue-chip" style="border-color:${levelColor[it.level] || '#a0aec0'};">
                    <span class="issue-name">${name}</span>
                    <span class="issue-level" style="color:${levelColor[it.level] || '#a0aec0'};">${it.level}</span>
                    <span class="issue-score">强度 ${it.score.toFixed(2)}</span>
                </div>
            `).join('')}
        </div>
    </div>`;
}

function updateResultCard(card, data, error) {
    card.className = 'result-card';
    if (error) {
        card.classList.add('error');
        card.querySelector('.result-status').innerHTML =
            '<span class="badge-error"><i class="fas fa-times"></i> 失败</span>';
        card.querySelector('.result-body').innerHTML =
            `<div class="error-msg">${error}</div>`;
        return;
    }
    if (data.status === 'success') {
        card.classList.add('completed');
        const score = data.score;
        card.querySelector('.result-status').innerHTML =
            `<span class="mos-score">${score.toFixed(2)}</span>`;
        card.querySelector('.result-body').innerHTML = `
            <div class="score-detail">
                <div class="score-meta">
                    <span><i class="fas fa-microchip"></i> ${data.model_name}</span>
                    <span><i class="fas fa-ruler"></i> 量纲 ${data.scale}</span>
                    <span><i class="fas fa-file-alt"></i> score.txt 行：${data.score_txt_line}</span>
                </div>
                ${issuesHtml(data.issues)}
            </div>`;
    } else {
        card.classList.add('error');
        card.querySelector('.result-status').innerHTML =
            '<span class="badge-error"><i class="fas fa-times"></i> 失败</span>';
        card.querySelector('.result-body').innerHTML =
            `<div class="error-msg">${data.detail || data.message || '评估失败'}</div>`;
    }
}

function clearResults() {
    if (confirm('确定要清空所有结果吗？')) {
        document.getElementById('resultsGrid').innerHTML = '';
        document.getElementById('resultsSection').style.display = 'none';
    }
}

// ============ score.txt ============
async function loadScoreTxt() {
    const mid = document.getElementById('scoreTxtModel').value;
    const r = await fetch(`/api/score/txt?model_id=${mid}`);
    const txt = await r.text();
    document.getElementById('scoreTxtContent').textContent =
        txt || '（该模型暂无打分记录）';
}

function downloadScoreTxt() {
    const mid = document.getElementById('scoreTxtModel').value;
    const content = document.getElementById('scoreTxtContent').textContent;
    const blob = new Blob([content], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `score_${mid}.txt`;
    a.click();
    URL.revokeObjectURL(a.href);
}

async function resetScore() {
    if (!confirm('确定清空所有打分记录？')) return;
    await fetch('/api/score/reset', { method: 'POST' });
    loadScoreTxt();
}
