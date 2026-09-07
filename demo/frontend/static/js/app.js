// demo/frontend/static/js/app.js —— 多模型版
const API_BASE = '';
let uploadQueue = [];
let isEvaluating = false;
let models = [];
let selectedModel = null;

// ---- 重新打分：优先复用服务端暂存 video_id（免重传）；暂存失效时回退重传内存中的原文件 ----
let _cardSeq = 0;
let _cardFiles = {};                 // cardId -> File（回退通道用）
let _rescoreChain = Promise.resolve();  // 串行执行，避免多个推理并发抢占 GPU
let _rescorePending = 0;             // 排队/执行中的重新打分数量

function _keepFile(file) {
    const id = String(++_cardSeq);
    _cardFiles[id] = file;
    return id;
}

function _rescoreActionsHtml(cardId) {
    if (!cardId) return '';
    return `
        <div class="card-actions">
            <button type="button" class="btn-rescore" onclick="rescoreCard('${cardId}')"
                    title="使用当前选中的模型重新打分，无需重新上传">
                <i class="fas fa-redo-alt"></i> 重新打分
            </button>
        </div>`;
}

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
    if (_rescorePending) { alert('有结果卡片正在重新打分，请稍候再开始批量评估'); return; }

    isEvaluating = true;
    const btn = document.getElementById('batchEvalBtn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 评估中...';

    const grid = document.getElementById('resultsGrid');
    grid.innerHTML = '';
    _cardFiles = {};   // 旧卡片即将被清空，释放其视频引用
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
    card.dataset.cardId = _keepFile(file);  // 保留视频文件供“重新打分”复用
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
    const cardId = card.dataset.cardId || '';
    const actionsHtml = _rescoreActionsHtml(cardId);
    if (error) {
        card.classList.add('error');
        card.querySelector('.result-status').innerHTML =
            '<span class="badge-error"><i class="fas fa-times"></i> 失败</span>';
        card.querySelector('.result-body').innerHTML =
            `<div class="error-msg">${error}</div>${actionsHtml}`;
        return;
    }
    if (data.status === 'success') {
        card.classList.add('completed');
        if (data.video_id) card._videoId = data.video_id;   // 服务端暂存 id，供免重传重新打分
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
            </div>
            ${actionsHtml}`;
    } else {
        card.classList.add('error');
        card.querySelector('.result-status').innerHTML =
            '<span class="badge-error"><i class="fas fa-times"></i> 失败</span>';
        card.querySelector('.result-body').innerHTML =
            `<div class="error-msg">${data.detail || data.message || '评估失败'}</div>${actionsHtml}`;
    }
}

// ============ 重新打分（切换模型后无需重新上传） ============
function rescoreCard(cardId) {
    const card = document.querySelector(`[data-card-id="${cardId}"]`);
    const file = _cardFiles[cardId];
    if (!card) return;
    if (card._busy) return;                        // 防止同一卡片重复点击
    if (isEvaluating) { alert('正在批量评估，请稍候再重新打分'); return; }
    if (!file) { alert('该视频数据已释放，请重新上传后再打分'); return; }
    if (!selectedModel || !selectedModel.available) { alert('请先选择可用的模型'); return; }

    const model = selectedModel;
    card._busy = true;
    card.classList.remove('completed', 'error');
    card.classList.add('processing');
    const statusEl = card.querySelector('.result-status');
    if (statusEl) {
        statusEl.innerHTML = '<span><i class="fas fa-spinner fa-spin"></i> 重新打分中…</span>';
    }
    const btn = card.querySelector('.btn-rescore');
    if (btn) btn.disabled = true;

    // 追加到串行链：同一时刻只跑一个重新打分
    _rescorePending++;
    _rescoreChain = _rescoreChain.then(async () => {
        try {
            const videoId = card._videoId || '';   // 服务端暂存 id（重启/过期后可能失效）
            let data = null;
            if (videoId) {
                // 免重传通道：后端按暂存 video_id 重新评估
                const r = await fetch(
                    `/api/predict/${encodeURIComponent(videoId)}?model_id=${encodeURIComponent(model.id)}`,
                    { method: 'POST' });
                if (r.ok) {
                    data = await r.json();
                } else if (r.status === 404) {
                    data = null;   // 暂存已过期/服务重启 → 回退重传
                } else {
                    data = await r.json();   // 其它后端错误（如推理失败 detail）直接展示
                }
            }
            if (!data) {
                // 回退通道：重传内存中保留的原视频文件（与批量评估同接口）
                const formData = new FormData();
                formData.append('file', file);
                const r2 = await fetch(`/api/predict?model_id=${encodeURIComponent(model.id)}`,
                    { method: 'POST', body: formData });
                data = await r2.json();
            }
            updateResultCard(card, data);
        } catch (err) {
            try { updateResultCard(card, null, err.message || '重新打分失败'); }
            catch (e) { console.error('更新重新打分结果失败:', e); }
        } finally {
            card._busy = false;
            _rescorePending--;
            loadScoreTxt();
        }
    }).catch(err => console.error('重新打分任务异常:', err));
}

function clearResults() {
    if (confirm('确定要清空所有结果吗？')) {
        document.getElementById('resultsGrid').innerHTML = '';
        _cardFiles = {};   // 释放卡片保留的视频引用
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
