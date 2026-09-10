// demo/frontend/static/js/app.js —— 多模型版
const API_BASE = '';
let uploadQueue = [];
let isEvaluating = false;
let models = [];
let multiAxes = [];               // /api/diagnose/axes 的维度与权重变体
let axisVariants = {};            // {轴 id: 权重文件名}，多维诊断的逐轴选择
let extraDetectors = [];          // 可选的组员检测器名称（噪点 / 模糊）
let extraSelected = {};           // {名称: 是否勾选}
let leaderboard = [];             // 排行榜：{video, model, modelId, score, scale}
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

function _rescoreActionsHtml(cardId, currentModelId) {
    if (!cardId) return '';
    // 其他可用模型：一键换模型快打（免重传，服务端暂存失效自动回退重传）
    const others = models.filter(m => m.available && m.id !== currentModelId);
    const quickBtns = others.map(m => `
        <button type="button" class="btn-quick" onclick="rescoreWithModel('${cardId}', '${m.id}')"
                title="用「${m.name}」重新评估，无需重新上传">
            <i class="fas fa-exchange-alt"></i> ${m.name}
        </button>`).join('');
    return `
        <div class="card-actions">
            <div class="quick-row">
                <span class="quick-label"><i class="fas fa-sync-alt"></i> 换模型再打分（免重传）</span>
                ${quickBtns}
            </div>
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
        loadAxes();
        renderModelInfo();
        if (models.length) selectModel(models.find(m => m.available) || models[0]);
    } catch (e) {
        document.getElementById('modelCards').innerHTML =
            '<div class="empty-hint">模型列表加载失败：' + e.message + '</div>';
        document.getElementById('modelStatus').innerHTML =
            '<i class="fas fa-times-circle" style="color:#fc8181;"></i> 服务连接失败';
    }
}

async function loadAxes() {
    try {
        const r = await fetch('/api/diagnose/axes');
        const d = await r.json();
        multiAxes = d.axes || [];
        extraDetectors = d.extras || [];
        extraDetectors.forEach(n => {
            if (!(n in extraSelected)) extraSelected[n] = true;
        });
    } catch (e) { multiAxes = []; }
    renderModelCards();
}

function variantPanelHtml() {
    // 仅在选中多维诊断时出现。左侧是四个测量轴，右侧是该轴已训练的权重变体，
    // 用单选圆点切换——即报告里那组分支 / 时间聚合消融。
    if (!selectedModel || !selectedModel.multiaxis) return '';
    const axes = multiAxes.filter(a => a.available);
    if (!axes.length) return '';
    axes.forEach(a => { if (!axisVariants[a.id]) axisVariants[a.id] = a.default; });
    return `
        <div class="variant-panel">
            <div class="variant-panel-head">
                <i class="fas fa-sliders-h"></i>
                <span>各维度权重（分支 / 时间聚合消融）</span>
            </div>
            ${axes.map(a => `
                <div class="axis-row">
                    <div class="axis-info">
                        <div class="axis-name">${a.name}</div>
                        <div class="axis-dir">runs/${a.dir}/ · 量纲 ${a.scale}</div>
                    </div>
                    <div class="axis-opts n${a.variants.length}">
                        ${a.variants.map(v => `
                            <label class="variant-item ${v.file === axisVariants[a.id] ? 'selected' : ''}">
                                <input type="radio" name="ax_${a.id}" value="${v.file}"
                                       ${v.file === axisVariants[a.id] ? 'checked' : ''}
                                       onchange="selectAxisVariant('${a.id}','${v.file}')">
                                <span class="variant-label">${v.label}</span>
                            </label>
                        `).join('')}
                    </div>
                </div>
            `).join('')}
            ${extrasHtml()}
            <div class="variant-hint">闪烁维度用启发式检测，无权重可选</div>
        </div>`;
}

function extrasHtml() {
    // 组员实现的检测器（噪点 / 模糊）：勾选后与四项测量一并输出
    if (!extraDetectors.length) return '';
    return `<div class="axis-row extras-row">
        <div class="axis-info">
            <div class="axis-name">附加检测</div>
            <div class="axis-dir">vqa.diagnosis</div>
        </div>
        <div class="axis-opts n2">
            ${extraDetectors.map(n => `
                <label class="variant-item ${extraSelected[n] ? 'selected' : ''}">
                    <input type="checkbox" ${extraSelected[n] ? 'checked' : ''}
                           onchange="toggleExtra('${n}', this.checked)">
                    <span class="variant-label">${n}</span>
                </label>
            `).join('')}
        </div>
    </div>`;
}

function toggleExtra(name, on) {
    extraSelected[name] = on;
    renderModelCards();
}

function selectAxisVariant(axisId, file) {
    axisVariants[axisId] = file;
    renderModelCards();
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
    `).join('') + variantPanelHtml();
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

// ============ 视频预览 ============
let _previewUrl = null;
let _previewFile = null;   // 当前预览的文件，用于避免重复重建

function showPreview(file) {
    if (!file) return;
    const pane = document.getElementById('previewPane');
    const video = document.getElementById('previewVideo');
    const layout = document.getElementById('uploadLayout');
    if (!pane || !video || !layout) return;

    if (_previewFile === file) return;   // 同一个文件不重建，避免播放被打断
    _previewFile = file;

    const oldUrl = _previewUrl;
    _previewUrl = URL.createObjectURL(file);
    video.src = _previewUrl;
    video.load();                        // 不调 load() 时换源后偶尔不响应播放
    // 旧 blob 要等新视频真正可播放后再释放，提前回收会让播放卡住
    if (oldUrl) {
        const release = () => {
            URL.revokeObjectURL(oldUrl);
            video.removeEventListener('loadeddata', release);
        };
        video.addEventListener('loadeddata', release);
        setTimeout(release, 5000);       // 兜底，避免事件不触发时泄漏
    }

    document.getElementById('previewName').textContent = file.name;
    const meta = document.getElementById('previewMeta');
    const mb = (file.size / 1024 / 1024).toFixed(1);
    meta.textContent = `${mb} MB`;
    video.onloadedmetadata = () => {
        meta.textContent = `${video.videoWidth}x${video.videoHeight} · `
            + `${video.duration.toFixed(1)} s · ${mb} MB`;
    };
    layout.classList.add('has-preview');
}

function clearPreview() {
    const layout = document.getElementById('uploadLayout');
    const video = document.getElementById('previewVideo');
    if (_previewUrl) { URL.revokeObjectURL(_previewUrl); _previewUrl = null; }
    _previewFile = null;
    if (video) { video.pause(); video.removeAttribute('src'); video.load(); }
    if (layout) layout.classList.remove('has-preview');
}

function handleFiles(files) {
    const videoFiles = Array.from(files).filter(f => f.type.startsWith('video/'));
    if (!videoFiles.length) { alert('请上传视频文件'); return; }
    uploadQueue = [...uploadQueue, ...videoFiles];
    updateQueueUI();
    syncPreview();          // 队列变化后刷新左侧预览
    document.getElementById('queueSection').style.display = 'block';
}

// ============ 队列管理 ============
function syncPreview(hideIfEmpty) {
    // 预览跟随队列第一个视频。评估结束后队列会被清空，但预览要留着，
    // 方便对照结果反复播放——只有手动清空或关闭时才收起。
    if (uploadQueue.length) showPreview(uploadQueue[0]);
    else if (hideIfEmpty) clearPreview();
}

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
    syncPreview(true);      // 手动移除：队列空了就收起
    if (!uploadQueue.length) document.getElementById('queueSection').style.display = 'none';
}

function clearQueue() {
    if (confirm('确定要清空队列吗？')) {
        uploadQueue = [];
        updateQueueUI();
        document.getElementById('queueSection').style.display = 'none';
        syncPreview(true);   // 手动清空：连同预览一起收起
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

    // 结果不再清空：同一个视频换模型评估会追加一张新卡片，方便横向对比
    const grid = document.getElementById('resultsGrid');
    grid.classList.add('results-strip');   // 横向滚动条，卡片左右排列
    const base = grid.querySelectorAll('.result-card').length;
    document.getElementById('resultsSection').style.display = 'block';

    for (let i = 0; i < uploadQueue.length; i++) {
        const file = uploadQueue[i];
        const card = createResultCard(file, base + i);
        grid.appendChild(card);

        try {
            const formData = new FormData();
            formData.append('file', file);
            // 多维诊断走 /api/diagnose，一次特征提取返回四项测量 + 闪烁 + 可视化
            let url;
            if (selectedModel.multiaxis) {
                const ex = extraDetectors.filter(n => extraSelected[n]);
                url = '/api/diagnose?with_visuals=true&variants='
                    + encodeURIComponent(JSON.stringify(axisVariants));
                if (ex.length) url += '&extras=' + encodeURIComponent(ex.join(','));
            } else {
                url = `/api/predict?model_id=${selectedModel.id}`;
            }
            const r = await fetch(url, { method: 'POST', body: formData });
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
    syncPreview();
}

function createResultCard(file, index) {
    const card = document.createElement('div');
    card.className = 'result-card processing';
    // Gradio 风格的加载占位：骨架块 + 轮换提示语，而不是单一进度条
    const hints = selectedModel && selectedModel.multiaxis
        ? ['正在解码抽帧...', '提取骨干特征...', '四个回归头打分...', '检测亮度闪烁...']
        : ['正在解码抽帧...', '模型推理中...'];
    card.innerHTML = `
        <div class="result-header">
            <div class="result-title">
                <span class="result-number">#${index + 1}</span>
                <span class="result-name">${file.name}</span>
            </div>
            <span class="result-status loading-hint"><i class="fas fa-spinner fa-spin"></i> <span class="hint-text">${hints[0]}</span></span>
        </div>
        <div class="result-body">
            <div class="skeleton-wrap">
                <div class="skeleton skeleton-line w70"></div>
                <div class="skeleton skeleton-line w45"></div>
                <div class="skeleton skeleton-block"></div>
            </div>
        </div>
    `;
    // 轮换提示语，让长耗时的多维诊断看起来有进展
    let hi = 0;
    card._hintTimer = setInterval(() => {
        hi = (hi + 1) % hints.length;
        const el = card.querySelector('.hint-text');
        if (el) el.textContent = hints[hi]; else clearInterval(card._hintTimer);
    }, 1200);
    card.dataset.cardId = _keepFile(file);  // 保留视频文件供“重新打分”复用
    return card;
}

// 失真等级配色，失真反馈与多维测量共用
const LEVEL_COLOR = { '无': '#48bb78', '轻': '#ecc94b', '中': '#ed8936',
                      '重': '#fc8181', '未知': '#a0aec0' };

function issuesHtml(issues) {
    if (!issues || !Object.keys(issues).length) return '';
    // 与多维测量同一张表的样式：一行一项，右侧一条强度条，
    // 这样不同模型的失真反馈可以横向直接对比
    const rows = Object.entries(issues).map(([name, it]) => {
        const c = LEVEL_COLOR[it.level] || '#a0aec0';
        return `<tr>
            <td class="mx-name">${name}</td>
            <td class="mx-level" style="color:${c};">${it.level}
                <span class="mx-sev">(${it.score.toFixed(2)})</span></td>
            <td class="mx-bar"><span style="width:${(it.score*100).toFixed(0)}%;background:${c};"></span></td>
        </tr>`;
    }).join('');
    return `<div class="mx-box">
        <div class="mx-title"><i class="fas fa-bug"></i> 失真问题反馈</div>
        <table class="mx-table">
            <thead><tr><th>失真类型</th><th>程度</th><th></th></tr></thead>
            <tbody>${rows}</tbody>
        </table>
    </div>`;
}

// ============ 排行榜 ============
function pushLeader(video, modelId, modelName, score, scale) {
    leaderboard.push({ video, modelId, model: modelName, score, scale });
    renderLeaderboard();
}

function renderLeaderboard() {
    const box = document.getElementById('leaderboard');
    if (!box) return;
    const sel = document.getElementById('leaderModel');
    const filter = sel ? sel.value : 'all';

    // 模型下拉：只列出已经出过结果的模型
    if (sel) {
        const ids = [...new Set(leaderboard.map(r => r.modelId))];
        const want = '<option value="all">全部模型</option>' + ids.map(id => {
            const n = (leaderboard.find(r => r.modelId === id) || {}).model || id;
            return `<option value="${id}">${n}</option>`;
        }).join('');
        if (sel.innerHTML !== want) {
            sel.innerHTML = want;
            sel.value = ids.includes(filter) || filter === 'all' ? filter : 'all';
        }
    }

    const rows = leaderboard
        .filter(r => filter === 'all' || r.modelId === filter)
        .slice()
        .sort((a, b) => b.score - a.score);

    if (!rows.length) {
        box.innerHTML = '<div class="empty-hint">评估后此处按分数排名</div>';
        return;
    }
    const mixed = filter === 'all'
        && new Set(rows.map(r => r.scale)).size > 1;

    box.innerHTML = `
        ${mixed ? '<div class="leader-warn"><i class="fas fa-exclamation-triangle"></i> 当前混合了不同量纲的模型，跨模型排名仅供参考</div>' : ''}
        <table class="mx-table leader-table">
            <thead><tr><th>名次</th><th>视频</th><th>分数</th><th>模型</th><th>量纲</th></tr></thead>
            <tbody>
                ${rows.map((r, i) => `
                    <tr class="${i === 0 ? 'leader-top' : ''}">
                        <td class="leader-rank">${i + 1}</td>
                        <td class="mx-name">${r.video}</td>
                        <td class="mx-pred">${r.score.toFixed(2)}</td>
                        <td class="leader-model">${r.model}</td>
                        <td class="leader-scale">${r.scale}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>`;
}

function clearLeaderboard() {
    leaderboard = [];
    renderLeaderboard();
}

// ============ 多维诊断渲染 ============
function measurementsHtml(ms) {
    if (!ms) return '';
    // 已知四轴 + 闪烁在前，组员的附加检测器（噪点 / 模糊等）接在后面
    const known = ['overall', 'shake', 'stutter', 'temporal', 'flicker'];
    const order = known.filter(k => ms[k])
        .concat(Object.keys(ms).filter(k => !known.includes(k)));
    const rows = order.map(k => {
        const m = ms[k];
        const c = LEVEL_COLOR[m.level] || '#a0aec0';
        const pred = (m.prediction === null || m.prediction === undefined)
            ? '-' : m.prediction.toFixed(2);
        return `<tr>
            <td class="mx-name">${m.name}</td>
            <td class="mx-pred">${pred}</td>
            <td class="mx-level" style="color:${c};">${m.level}
                <span class="mx-sev">(${m.severity.toFixed(2)})</span></td>
            <td class="mx-bar"><span style="width:${(m.severity*100).toFixed(0)}%;background:${c};"></span></td>
            <td class="mx-desc">${m.desc}</td>
        </tr>`;
    }).join('');
    const fl = ms.flicker;
    const detail = (fl && fl.detail) ? `<div class="mx-detail">
        闪烁细节：亮度波动 ${fl.detail.luma_pump} ｜ 周期性 ${fl.detail.periodicity}
        ｜ 帧差 ${fl.detail.frame_diff_mean} ｜ ${fl.sampling || ''}</div>` : '';
    return `<div class="mx-box">
        <div class="mx-title"><i class="fas fa-sliders-h"></i> 多维测量</div>
        <table class="mx-table">
            <thead><tr><th>维度</th><th>预测值</th><th>程度</th><th></th><th>说明</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>${detail}
        <div class="mx-note">预测值为该轴原始输出（越高越好）；程度为 0~1 失真严重度（越高越差）</div>
    </div>`;
}

// 折叠块：默认收起，点击标题展开——抽取的帧与图占位较大
function collapsibleHtml(icon, title, bodyHtml, openByDefault) {
    if (!bodyHtml) return '';
    const open = openByDefault ? ' open' : '';
    return `<div class="mx-box mx-collapse${open}">
        <div class="mx-title mx-toggle" onclick="toggleBox(this)">
            <i class="fas ${icon}"></i> ${title}
            <i class="fas fa-chevron-down mx-chevron"></i>
        </div>
        <div class="mx-collapse-body">${bodyHtml}</div>
    </div>`;
}

function toggleBox(el) {
    el.parentElement.classList.toggle('open');
}

function framesHtml(frames) {
    if (!frames || !frames.length) return '';
    const body = `<div class="mx-frames">
        ${frames.map((f, i) => `<img src="data:image/jpeg;base64,${f}" title="frame ${i+1}">`).join('')}
    </div>`;
    return collapsibleHtml('fa-images', `抽取的帧（${frames.length}）`, body, false);
}

function plotsHtml(plots) {
    if (!plots) return '';
    const body = `<div class="mx-plots">
        ${plots.instability ? `<img src="data:image/png;base64,${plots.instability}">` : ''}
        ${plots.luma ? `<img src="data:image/png;base64,${plots.luma}">` : ''}
    </div>`;
    return collapsibleHtml('fa-chart-line', '时域信号', body, false);
}

// 换抖动权重重新诊断：复用服务端暂存视频
async function rediagnose(cardId, variant) {
    const card = document.querySelector(`[data-card-id="${cardId}"]`);
    if (!card || !card._videoId) { alert('视频已过期，请重新上传'); return; }
    if (card._busy) return;
    card._busy = true;
    const status = card.querySelector('.result-status');
    const old = status.innerHTML;
    status.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 重新诊断...';
    try {
        const ex = extraDetectors.filter(n => extraSelected[n]);
        let u = `/api/diagnose/${card._videoId}?with_visuals=true&variants=`
            + encodeURIComponent(JSON.stringify(axisVariants));
        if (ex.length) u += '&extras=' + encodeURIComponent(ex.join(','));
        const r = await fetch(u, { method: 'POST' });
        const data = await r.json();
        card._busy = false;
        updateResultCard(card, data);
    } catch (e) {
        card._busy = false;
        status.innerHTML = old;
        alert('重新诊断失败：' + e.message);
    }
}

function updateResultCard(card, data, error) {
    if (card._hintTimer) { clearInterval(card._hintTimer); card._hintTimer = null; }
    card.className = 'result-card';
    const cardId = card.dataset.cardId || '';
    if (data && data.model_id) card._modelId = data.model_id;
    const curModel = (data && data.model_id) || card._modelId || '';
    const actionsHtml = _rescoreActionsHtml(cardId, curModel);
    if (error) {
        card.classList.add('error');
        card.querySelector('.result-status').innerHTML =
            '<span class="badge-error"><i class="fas fa-times"></i> 失败</span>';
        card.querySelector('.result-body').innerHTML =
            `<div class="error-msg">${error}</div>${actionsHtml}`;
        return;
    }
    if (data.status === 'success' && data.measurements) {
        // 多维诊断结果
        card.classList.add('completed');
        if (data.video_id) card._videoId = data.video_id;
        const ov = data.measurements.overall;
        if (ov) pushLeader(data.video || card.querySelector('.result-name').textContent,
                           'multiaxis', '多维诊断', ov.prediction, '1~5');
        card.querySelector('.result-status').innerHTML = ov
            ? `<span class="mos-score">${ov.prediction.toFixed(2)}</span>`
            : '<span class="mos-score">-</span>';
        card.querySelector('.result-body').innerHTML = `
            <div class="score-detail">
                <div class="score-meta">
                    <span><i class="fas fa-film"></i> ${data.num_frames} 帧</span>
                    <span><i class="fas fa-crop"></i> 采样 ${data.sampling}</span>
                    ${data.score_txt_line ? `<span><i class="fas fa-file-alt"></i> score.txt 行：${data.score_txt_line}</span>` : ''}
                </div>
                ${measurementsHtml(data.measurements)}
                ${framesHtml(data.frames)}
                ${plotsHtml(data.plots)}
            </div>
            ${actionsHtml}`;
        return;
    }
    if (data.status === 'success') {
        card.classList.add('completed');
        if (data.video_id) card._videoId = data.video_id;   // 服务端暂存 id，供免重传重新打分
        const score = data.score;
        pushLeader(data.video_name || card.querySelector('.result-name').textContent,
                   data.model_id, data.model_name, score, data.scale);
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
    if (!selectedModel || !selectedModel.available) { alert('请先选择可用的模型'); return; }
    rescoreWithModel(cardId, selectedModel.id);
}

// 指定模型重新打分：优先免重传（服务端暂存 video_id），失效自动回退重传
function rescoreWithModel(cardId, modelId) {
    const card = document.querySelector(`[data-card-id="${cardId}"]`);
    const file = _cardFiles[cardId];
    if (!card) return;
    if (card._busy) return;                        // 防止同一卡片重复点击
    if (isEvaluating) { alert('正在批量评估，请稍候再重新打分'); return; }
    if (!file) { alert('该视频数据已释放，请重新上传后再打分'); return; }
    const model = models.find(m => m.id === modelId && m.available);
    if (!model) { alert('目标模型不可用'); return; }

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
