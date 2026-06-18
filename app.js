/**
 * ccatai frontend — app.js
 * ========================
 * Handles:
 *   • Drag-and-drop + click file upload
 *   • Form validation and settings persistence (localStorage)
 *   • XHR multipart upload with progress
 *   • SSE progress stream → live pipeline stage updates
 *   • Results rendering (clips grid, dashboard strip, analysis table)
 *   • Job history panel
 *   • Download all as zip (sequential anchor clicks)
 *   • Toast notifications
 *   • Cancel job
 */

(() => {
'use strict';

const RENDER_BACKEND_URL = "https://my-backend-vzj9.onrender.com";

// ─── State ────────────────────────────────────────────────────────────────
let selectedFile   = null;
let currentJobId   = null;
let evtSource      = null;
let uploadXHR      = null;

// ─── DOM refs ─────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const dropZone       = $('dropZone');
const fileInput      = $('fileInput');
const browseBtn      = $('browseBtn');
const removeFile     = $('removeFile');
const filePreview    = $('filePreview');
const fileName       = $('fileName');
const fileSize       = $('fileSize');
const generateBtn    = $('generateBtn');
const cancelBtn      = $('cancelBtn');
const newJobBtn      = $('newJobBtn');
const retryBtn       = $('retryBtn');

// Panels
const emptyState     = $('emptyState');
const progressPanel  = $('progressPanel');
const resultsPanel   = $('resultsPanel');
const historyPanel   = $('historyPanel');
const errorPanel     = $('errorPanel');

// Progress
const progressFill   = $('progressFill');
const progressPct    = $('progressPct');
const progressStage  = $('progressStage');
const logBox         = $('logBox');
const stageItems     = document.querySelectorAll('.stage-item');

// Results
const dashScore      = $('dashScore');
const dashLang       = $('dashLang');
const dashClips      = $('dashClips');
const dashDuration   = $('dashDuration');
const hookBanner     = $('hookBanner');
const hookText       = $('hookText');
const hookReason     = $('hookReason');
const summaryRow     = $('summaryRow');
const summaryText    = $('summaryText');
const clipsGrid      = $('clipsGrid');
const analysisBody   = $('analysisBody');
const analysisSection= $('analysisSection');
const downloadAllBtn = $('downloadAllBtn');
const historyList    = $('historyList');
const navBtns        = document.querySelectorAll('.nav-btn');
const apiStatus      = $('apiStatus');

// ─── Settings persistence ─────────────────────────────────────────────────
const SETTINGS_KEY = 'ccatai_settings';

function saveSettings() {
  const s = {
    groqKey:        $('groqKey').value,
    numHighlights:  $('numHighlights').value,
    language:       $('language').value,
    whisperSize:    $('whisperSize').value,
    multipleShorts: $('multipleShorts').checked,
    addCaptions:    $('addCaptions').checked,
    addZoom:        $('addZoom').checked,
    addHookCard:    $('addHookCard').checked,
    musicDir:       $('musicDir').value,
  };
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
}

function loadSettings() {
  try {
    const s = JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}');
    if (s.groqKey)       $('groqKey').value        = s.groqKey;
    if (s.numHighlights) $('numHighlights').value   = s.numHighlights;
    if (s.language)      $('language').value        = s.language;
    if (s.whisperSize)   $('whisperSize').value      = s.whisperSize;
    if (s.multipleShorts !== undefined) $('multipleShorts').checked = s.multipleShorts;
    if (s.addCaptions   !== undefined) $('addCaptions').checked    = s.addCaptions;
    if (s.addZoom       !== undefined) $('addZoom').checked        = s.addZoom;
    if (s.addHookCard   !== undefined) $('addHookCard').checked    = s.addHookCard;
    if (s.musicDir)      $('musicDir').value        = s.musicDir;
  } catch (_) {}
}

document.querySelectorAll('.field-input, .toggle-item input').forEach(el => {
  el.addEventListener('change', saveSettings);
});

// ─── File handling ────────────────────────────────────────────────────────
function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 ** 2) return (bytes / 1024).toFixed(1) + ' KB';
  if (bytes < 1024 ** 3) return (bytes / 1024 ** 2).toFixed(1) + ' MB';
  return (bytes / 1024 ** 3).toFixed(1) + ' GB';
}

function setFile(file) {
  if (!file) return;
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  const allowed = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'];
  if (!allowed.includes(ext)) {
    toast('Unsupported file type: ' + ext, 'error');
    return;
  }
  selectedFile = file;
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  filePreview.classList.remove('hidden');
  dropZone.classList.add('hidden');
  validateForm();
}

function clearFile() {
  selectedFile = null;
  fileInput.value = '';
  filePreview.classList.add('hidden');
  dropZone.classList.remove('hidden');
  validateForm();
}

function validateForm() {
  const hasFile = !!selectedFile;
  const hasKey  = $('groqKey').value.trim().length > 10;
  generateBtn.disabled = !(hasFile && hasKey);
}

browseBtn.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });
removeFile.addEventListener('click', clearFile);
$('groqKey').addEventListener('input', validateForm);

// Drag & drop
['dragenter', 'dragover'].forEach(ev =>
  dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.add('over'); })
);
['dragleave', 'drop'].forEach(ev =>
  dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.remove('over'); })
);
dropZone.addEventListener('drop', e => {
  const f = e.dataTransfer?.files?.[0];
  if (f) setFile(f);
});

// ─── Upload & start job ────────────────────────────────────────────────────
generateBtn.addEventListener('click', startJob);

async function startJob() {
  if (!selectedFile) return;

  saveSettings();
  showPanel('progress');
  setStatus('busy', 'Processing…');
  resetProgress();
  logLine('Uploading video…', 'dim');

  const fd = new FormData();
  fd.append('video',           selectedFile);
  fd.append('groq_api_key',    $('groqKey').value.trim());
  fd.append('num_highlights',  $('numHighlights').value);
  fd.append('language',        $('language').value);
  fd.append('whisper_size',    $('whisperSize').value);
  fd.append('multiple_shorts', $('multipleShorts').checked ? 'true' : 'false');
  fd.append('add_captions',    $('addCaptions').checked    ? 'true' : 'false');
  fd.append('add_zoom',        $('addZoom').checked        ? 'true' : 'false');
  fd.append('add_hook_card',   $('addHookCard').checked    ? 'true' : 'false');
  fd.append('music_dir',       $('musicDir').value.trim());

  uploadXHR = new XMLHttpRequest();
  uploadXHR.open('POST', `${RENDER_BACKEND_URL}/api/upload`);

  uploadXHR.upload.addEventListener('progress', e => {
    if (e.lengthComputable) {
      const pct = Math.round(e.loaded / e.total * 4); // upload = 0–4%
      setProgress(pct, 'Uploading', formatBytes(e.loaded) + ' / ' + formatBytes(e.total));
    }
  });

  uploadXHR.addEventListener('load', () => {
    if (uploadXHR.status === 202) {
      const res = JSON.parse(uploadXHR.responseText);
      currentJobId = res.job_id;
      logLine('Job started: ' + currentJobId.slice(0, 8) + '…', 'accent');
      openProgressStream(currentJobId);
    } else {
      let msg = 'Upload failed';
      try { msg = JSON.parse(uploadXHR.responseText).error || msg; } catch (_) {}
      showError(msg);
    }
  });

  uploadXHR.addEventListener('error', () => showError('Network error during upload.'));
  uploadXHR.send(fd);
}

// ─── SSE progress stream ───────────────────────────────────────────────────
function openProgressStream(jobId) {
  if (evtSource) evtSource.close();
  evtSource = new EventSource(`${RENDER_BACKEND_URL}/api/progress/${jobId}`);

  evtSource.onmessage = e => {
    if (!e.data) return;
    let msg;
    try { msg = JSON.parse(e.data); } catch (_) { return; }

    if (msg.heartbeat) return;
    if (msg.done) { evtSource.close(); fetchJobResult(jobId); return; }

    const { percent, stage, detail, status } = msg;
    setProgress(percent, stage, detail);
    logLine(stage + (detail ? ': ' + detail : ''), status === 'error' ? 'error' : 'dim');
    updateStageIndicators(percent, stage);

    if (status === 'error') {
      evtSource.close();
      showError(detail || 'An error occurred during processing.');
    }
    if (status === 'cancelled') {
      evtSource.close();
      showPanel('empty');
      setStatus('ready', 'Ready');
      toast('Job cancelled.', 'info');
    }
  };

  evtSource.onerror = () => {
    evtSource.close();
    // Retry via polling fallback
    pollJobStatus(jobId);
  };
}

function pollJobStatus(jobId, attempts = 0) {
  if (attempts > 60) { showError('Connection lost. Check server logs.'); return; }
  setTimeout(async () => {
    try {
      const r = await fetch(`${RENDER_BACKEND_URL}/api/job/${jobId}`);
      const job = await r.json();
      setProgress(job.progress, job.stage, '');
      if (job.status === 'done')      { fetchJobResult(jobId); return; }
      if (job.status === 'error')     { showError(job.error || 'Processing error'); return; }
      if (job.status === 'cancelled') { showPanel('empty'); return; }
      pollJobStatus(jobId, attempts + 1);
    } catch (_) {
      pollJobStatus(jobId, attempts + 1);
    }
  }, 2000);
}

// ─── Fetch final result ────────────────────────────────────────────────────
async function fetchJobResult(jobId) {
  try {
    const r = await fetch(`${RENDER_BACKEND_URL}/api/job/${jobId}`);
    const job = await r.json();
    if (job.status === 'done') {
      renderResults(jobId, job.dashboard, job.outputs);
      setStatus('ready', 'Ready');
      toast('Shorts generated! 🎉', 'success');
    } else if (job.status === 'error') {
      showError(job.error || 'Processing failed');
    }
  } catch (e) {
    showError('Failed to fetch results: ' + e.message);
  }
}

// ─── Render results ────────────────────────────────────────────────────────
function renderResults(jobId, dashboard, outputs) {
  showPanel('results');

  // Dashboard strip
  const score = dashboard?.overall_virality_score ?? '—';
  dashScore.textContent    = typeof score === 'number' ? score + '/10' : score;
  dashLang.textContent     = dashboard?.language ?? '—';
  dashClips.textContent    = (dashboard?.num_highlights ?? 0).toString();
  const dur = dashboard?.video_duration_s;
  dashDuration.textContent = dur ? fmtDuration(dur) : '—';

  // Hook
  const hook = dashboard?.hook;
  if (hook?.text) {
    hookText.textContent   = hook.text;
    hookReason.textContent = hook.reason || '';
    hookBanner.classList.remove('hidden');
  }

  // Summary
  const summary = dashboard?.content_summary;
  if (summary) {
    summaryText.textContent = summary;
    summaryRow.classList.remove('hidden');
  }

  // Clips grid
  clipsGrid.innerHTML = '';
  const videoOutputs = outputs.filter(f => f.endsWith('.mp4'));
  const highlights   = dashboard?.highlights ?? [];

  videoOutputs.forEach((fname, i) => {
    const hl     = highlights[i] || {};
    const thumb  = fname.replace('.mp4', '_thumb.jpg');
    const card   = document.createElement('div');
    card.className = 'clip-card';
    card.innerHTML = `
      <div class="clip-thumb">
        <img src="${RENDER_BACKEND_URL}/api/thumbnail/${jobId}/${thumb}" alt="Clip ${i+1} thumbnail"
             onerror="this.parentElement.innerHTML='<div class=\\'clip-thumb-placeholder\\'><svg width=\\'32\\' height=\\'32\\' viewBox=\\'0 0 24 24\\' fill=\\'none\\' stroke=\\'currentColor\\' stroke-width=\\'1.5\\'><polygon points=\\'5 3 19 12 5 21 5 3\\'/></svg><span style=\\'font-size:11px;color:var(--text-dim)\\'>Preview</span></div>'">
        ${hl.emoji ? `<span class="clip-emoji">${hl.emoji}</span>` : ''}
        ${hl.zoom  ? `<span class="clip-zoom-tag">ZOOM</span>` : ''}
      </div>
      <div class="clip-meta">
        <div class="clip-label">Short ${i + 1}</div>
        <div class="clip-reason">${hl.reason || 'Highlight clip'}</div>
        <div class="clip-ts">${hl.start != null ? fmtTs(hl.start) + '–' + fmtTs(hl.end) : ''}</div>
        <a class="clip-dl-btn" href="${RENDER_BACKEND_URL}/api/download/${jobId}/${fname}" download="${fname}">↓ Download</a>
      </div>`;
    clipsGrid.appendChild(card);
  });

  // Download all button
  if (videoOutputs.length > 1) {
    downloadAllBtn.classList.remove('hidden');
    downloadAllBtn.onclick = () => downloadAll(jobId, videoOutputs);
  }

  // Analysis table
  if (highlights.length > 0) {
    analysisBody.innerHTML = '';
    highlights.forEach((hl, i) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${i + 1}</td>
        <td><span class="ts-badge">${fmtTs(hl.start)}–${fmtTs(hl.end)}</span></td>
        <td style="font-size:18px">${hl.emoji || '—'}</td>
        <td class="${hl.zoom ? 'zoom-yes' : 'zoom-no'}">${hl.zoom ? '🔍 Yes' : 'No'}</td>
        <td>${hl.reason || '—'}</td>`;
      analysisBody.appendChild(tr);
    });
    analysisSection.classList.remove('hidden');
  }
}

function downloadAll(jobId, files) {
  files.forEach((fname, i) => {
    setTimeout(() => {
      const a = document.createElement('a');
      a.href     = `${RENDER_BACKEND_URL}/api/download/${jobId}/${fname}`;
      a.download = fname;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }, i * 600);
  });
}

// ─── Progress helpers ──────────────────────────────────────────────────────
function setProgress(pct, stage, detail) {
  progressFill.style.width = pct + '%';
  progressPct.textContent  = pct + '%';
  progressStage.textContent = stage + (detail ? ' — ' + detail : '');
}

function resetProgress() {
  setProgress(0, 'Starting', '');
  logBox.innerHTML = '';
  stageItems.forEach(el => el.classList.remove('active', 'done', 'error'));
}

const STAGE_MAP = {
  'Transcrib': 'transcribe',
  'AI Analys': 'analysis',
  'Analysis':  'analysis',
  'Rendering': 'render',
  'Merging':   'render',
  'Encoding':  'export',
  'Export':    'export',
  'Finalising':'export',
  'Complete':  'export',
};

function updateStageIndicators(pct, stage) {
  const thresholds = { transcribe: 25, analysis: 45, render: 90, export: 99 };
  const order = ['transcribe', 'analysis', 'render', 'export'];

  let activeStage = null;
  for (const [key, val] of Object.entries(STAGE_MAP)) {
    if (stage.startsWith(key)) { activeStage = val; break; }
  }

  order.forEach(s => {
    const el = document.querySelector(`.stage-item[data-stage="${s}"]`);
    if (!el) return;
    el.classList.remove('active', 'done', 'error');
    if (pct >= thresholds[s]) {
      el.classList.add('done');
      const ss = el.querySelector('.stage-status');
      if (ss) ss.textContent = '✓';
    } else if (s === activeStage) {
      el.classList.add('active');
      const ss = el.querySelector('.stage-status');
      if (ss) ss.textContent = pct + '%';
    }
  });
}

function logLine(text, cls = '') {
  const line = document.createElement('span');
  line.className = 'log-line' + (cls ? ' ' + cls : '');
  line.textContent = '› ' + text;
  logBox.appendChild(line);
  logBox.scrollTop = logBox.scrollHeight;
}

// ─── Cancel ────────────────────────────────────────────────────────────────
cancelBtn.addEventListener('click', async () => {
  if (!currentJobId) return;
  if (evtSource) evtSource.close();
  if (uploadXHR)  uploadXHR.abort();
  try {
    await fetch(`${RENDER_BACKEND_URL}/api/cancel/${currentJobId}`, { method: 'POST' });
  } catch (_) {}
  showPanel('empty');
  setStatus('ready', 'Ready');
  toast('Job cancelled.', 'info');
});

// ─── Nav ────────────────────────────────────────────────────────────────────
navBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    navBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const panel = btn.dataset.panel;
    if (panel === 'history') {
      loadHistory();
      showPanel('history');
    } else {
      showPanel(currentJobId ? 'results' : 'empty');
    }
  });
});

newJobBtn.addEventListener('click', () => {
  currentJobId = null;
  clearFile();
  showPanel('empty');
});

retryBtn.addEventListener('click', () => {
  showPanel('empty');
});

// ─── History ────────────────────────────────────────────────────────────────
async function loadHistory() {
  try {
    const r    = await fetch(`${RENDER_BACKEND_URL}/api/jobs`);
    const jobs = await r.json();
    historyList.innerHTML = '';
    if (!jobs.length) {
      historyList.innerHTML = '<p style="color:var(--text-dim);font-size:14px">No jobs yet.</p>';
      return;
    }
    jobs.forEach(j => {
      const el = document.createElement('div');
      el.className = 'history-item';
      el.innerHTML = `
        <div class="history-status ${j.status}"></div>
        <div class="history-info">
          <span class="history-id">${j.id.slice(0, 8)}…</span>
          <div class="history-stage">${j.stage} — ${j.status}</div>
        </div>
        <span class="history-time">${relTime(j.created_at)}</span>`;
      el.addEventListener('click', () => {
        if (j.status === 'done') {
          currentJobId = j.id;
          fetchJobResult(j.id);
          navBtns.forEach(b => b.classList.remove('active'));
          document.querySelector('[data-panel="upload"]').classList.add('active');
        }
      });
      historyList.appendChild(el);
    });
  } catch (e) {
    historyList.innerHTML = '<p style="color:var(--text-dim)">Could not load history.</p>';
  }
}

// ─── Panel visibility ──────────────────────────────────────────────────────
function showPanel(name) {
  emptyState.classList.add('hidden');
  progressPanel.classList.add('hidden');
  resultsPanel.classList.add('hidden');
  historyPanel.classList.add('hidden');
  errorPanel.classList.add('hidden');
  if (name === 'empty')    emptyState.classList.remove('hidden');
  if (name === 'progress') progressPanel.classList.remove('hidden');
  if (name === 'results')  resultsPanel.classList.remove('hidden');
  if (name === 'history')  historyPanel.classList.remove('hidden');
  if (name === 'error')    errorPanel.classList.remove('hidden');
}

function showError(msg) {
  $('errorMsg').textContent = msg;
  showPanel('error');
  setStatus('error', 'Error');
  toast(msg, 'error');
}

// ─── Status indicator ──────────────────────────────────────────────────────
function setStatus(state, label) {
  const dot  = apiStatus.querySelector('.status-dot');
  const lbl  = apiStatus.querySelector('.status-label');
  dot.className  = 'status-dot' + (state !== 'ready' ? ' ' + state : '');
  lbl.textContent = label;
}

// ─── Toast ─────────────────────────────────────────────────────────────────
function toast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  $('toastContainer').appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

// ─── Formatters ────────────────────────────────────────────────────────────
function fmtDuration(s) {
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}
function fmtTs(s) {
  if (s == null) return '';
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1);
  return m > 0 ? `${m}:${String(Math.floor(s % 60)).padStart(2,'0')}` : `${parseFloat(sec).toFixed(1)}s`;
}
function relTime(iso) {
  const diff = Date.now() - new Date(iso + 'Z').getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1)   return 'just now';
  if (m < 60)  return `${m}m ago`;
  return `${Math.floor(m / 60)}h ago`;
}

// ─── Init ──────────────────────────────────────────────────────────────────
loadSettings();
validateForm();
showPanel('empt  if (evtSource) evtSource.close();
  evtSource = new EventSource(`/api/progress/${jobId}`);

  evtSource.onmessage = e => {
    if (!e.data) return;
    let msg;
    try { msg = JSON.parse(e.data); } catch (_) { return; }

    if (msg.heartbeat) return;
    if (msg.done) { evtSource.close(); fetchJobResult(jobId); return; }

    const { percent, stage, detail, status } = msg;
    setProgress(percent, stage, detail);
    logLine(stage + (detail ? ': ' + detail : ''), status === 'error' ? 'error' : 'dim');
    updateStageIndicators(percent, stage);

    if (status === 'error') {
      evtSource.close();
      showError(detail || 'An error occurred during processing.');
    }
    if (status === 'cancelled') {
      evtSource.close();
      showPanel('empty');
      setStatus('ready', 'Ready');
      toast('Job cancelled.', 'info');
    }
  };

  evtSource.onerror = () => {
    evtSource.close();
    // Retry via polling fallback
    pollJobStatus(jobId);
  };
}

function pollJobStatus(jobId, attempts = 0) {
  if (attempts > 60) { showError('Connection lost. Check server logs.'); return; }
  setTimeout(async () => {
    try {
      const r = await fetch(`/api/job/${jobId}`);
      const job = await r.json();
      setProgress(job.progress, job.stage, '');
      if (job.status === 'done')      { fetchJobResult(jobId); return; }
      if (job.status === 'error')     { showError(job.error || 'Processing error'); return; }
      if (job.status === 'cancelled') { showPanel('empty'); return; }
      pollJobStatus(jobId, attempts + 1);
    } catch (_) {
      pollJobStatus(jobId, attempts + 1);
    }
  }, 2000);
}

// ─── Fetch final result ────────────────────────────────────────────────────
async function fetchJobResult(jobId) {
  try {
    const r = await fetch(`/api/job/${jobId}`);
    const job = await r.json();
    if (job.status === 'done') {
      renderResults(jobId, job.dashboard, job.outputs);
      setStatus('ready', 'Ready');
      toast('Shorts generated! 🎉', 'success');
    } else if (job.status === 'error') {
      showError(job.error || 'Processing failed');
    }
  } catch (e) {
    showError('Failed to fetch results: ' + e.message);
  }
}

// ─── Render results ────────────────────────────────────────────────────────
function renderResults(jobId, dashboard, outputs) {
  showPanel('results');

  // Dashboard strip
  const score = dashboard?.overall_virality_score ?? '—';
  dashScore.textContent    = typeof score === 'number' ? score + '/10' : score;
  dashLang.textContent     = dashboard?.language ?? '—';
  dashClips.textContent    = (dashboard?.num_highlights ?? 0).toString();
  const dur = dashboard?.video_duration_s;
  dashDuration.textContent = dur ? fmtDuration(dur) : '—';

  // Hook
  const hook = dashboard?.hook;
  if (hook?.text) {
    hookText.textContent   = hook.text;
    hookReason.textContent = hook.reason || '';
    hookBanner.classList.remove('hidden');
  }

  // Summary
  const summary = dashboard?.content_summary;
  if (summary) {
    summaryText.textContent = summary;
    summaryRow.classList.remove('hidden');
  }

  // Clips grid
  clipsGrid.innerHTML = '';
  const videoOutputs = outputs.filter(f => f.endsWith('.mp4'));
  const highlights   = dashboard?.highlights ?? [];

  videoOutputs.forEach((fname, i) => {
    const hl     = highlights[i] || {};
    const thumb  = fname.replace('.mp4', '_thumb.jpg');
    const card   = document.createElement('div');
    card.className = 'clip-card';
    card.innerHTML = `
      <div class="clip-thumb">
        <img src="/api/thumbnail/${jobId}/${thumb}" alt="Clip ${i+1} thumbnail"
             onerror="this.parentElement.innerHTML='<div class=\\'clip-thumb-placeholder\\'><svg width=\\'32\\' height=\\'32\\' viewBox=\\'0 0 24 24\\' fill=\\'none\\' stroke=\\'currentColor\\' stroke-width=\\'1.5\\'><polygon points=\\'5 3 19 12 5 21 5 3\\'/></svg><span style=\\'font-size:11px;color:var(--text-dim)\\'>Preview</span></div>'">
        ${hl.emoji ? `<span class="clip-emoji">${hl.emoji}</span>` : ''}
        ${hl.zoom  ? `<span class="clip-zoom-tag">ZOOM</span>` : ''}
      </div>
      <div class="clip-meta">
        <div class="clip-label">Short ${i + 1}</div>
        <div class="clip-reason">${hl.reason || 'Highlight clip'}</div>
        <div class="clip-ts">${hl.start != null ? fmtTs(hl.start) + '–' + fmtTs(hl.end) : ''}</div>
        <a class="clip-dl-btn" href="/api/download/${jobId}/${fname}" download="${fname}">↓ Download</a>
      </div>`;
    clipsGrid.appendChild(card);
  });

  // Download all button
  if (videoOutputs.length > 1) {
    downloadAllBtn.classList.remove('hidden');
    downloadAllBtn.onclick = () => downloadAll(jobId, videoOutputs);
  }

  // Analysis table
  if (highlights.length > 0) {
    analysisBody.innerHTML = '';
    highlights.forEach((hl, i) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${i + 1}</td>
        <td><span class="ts-badge">${fmtTs(hl.start)}–${fmtTs(hl.end)}</span></td>
        <td style="font-size:18px">${hl.emoji || '—'}</td>
        <td class="${hl.zoom ? 'zoom-yes' : 'zoom-no'}">${hl.zoom ? '🔍 Yes' : 'No'}</td>
        <td>${hl.reason || '—'}</td>`;
      analysisBody.appendChild(tr);
    });
    analysisSection.classList.remove('hidden');
  }
}

function downloadAll(jobId, files) {
  files.forEach((fname, i) => {
    setTimeout(() => {
      const a = document.createElement('a');
      a.href     = `/api/download/${jobId}/${fname}`;
      a.download = fname;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }, i * 600);
  });
}

// ─── Progress helpers ──────────────────────────────────────────────────────
function setProgress(pct, stage, detail) {
  progressFill.style.width = pct + '%';
  progressPct.textContent  = pct + '%';
  progressStage.textContent = stage + (detail ? ' — ' + detail : '');
}

function resetProgress() {
  setProgress(0, 'Starting', '');
  logBox.innerHTML = '';
  stageItems.forEach(el => el.classList.remove('active', 'done', 'error'));
}

const STAGE_MAP = {
  'Transcrib': 'transcribe',
  'AI Analys': 'analysis',
  'Analysis':  'analysis',
  'Rendering': 'render',
  'Merging':   'render',
  'Encoding':  'export',
  'Export':    'export',
  'Finalising':'export',
  'Complete':  'export',
};

function updateStageIndicators(pct, stage) {
  // Mark stages done/active based on % thresholds
  const thresholds = { transcribe: 25, analysis: 45, render: 90, export: 99 };
  const order = ['transcribe', 'analysis', 'render', 'export'];

  let activeStage = null;
  for (const [key, val] of Object.entries(STAGE_MAP)) {
    if (stage.startsWith(key)) { activeStage = val; break; }
  }

  order.forEach(s => {
    const el = document.querySelector(`.stage-item[data-stage="${s}"]`);
    if (!el) return;
    el.classList.remove('active', 'done', 'error');
    if (pct >= thresholds[s]) {
      el.classList.add('done');
      const ss = el.querySelector('.stage-status');
      if (ss) ss.textContent = '✓';
    } else if (s === activeStage) {
      el.classList.add('active');
      const ss = el.querySelector('.stage-status');
      if (ss) ss.textContent = pct + '%';
    }
  });
}

function logLine(text, cls = '') {
  const line = document.createElement('span');
  line.className = 'log-line' + (cls ? ' ' + cls : '');
  line.textContent = '› ' + text;
  logBox.appendChild(line);
  logBox.scrollTop = logBox.scrollHeight;
}

// ─── Cancel ────────────────────────────────────────────────────────────────
cancelBtn.addEventListener('click', async () => {
  if (!currentJobId) return;
  if (evtSource) evtSource.close();
  if (uploadXHR)  uploadXHR.abort();
  try {
    await fetch(`/api/cancel/${currentJobId}`, { method: 'POST' });
  } catch (_) {}
  showPanel('empty');
  setStatus('ready', 'Ready');
  toast('Job cancelled.', 'info');
});

// ─── Nav ────────────────────────────────────────────────────────────────────
navBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    navBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const panel = btn.dataset.panel;
    if (panel === 'history') {
      loadHistory();
      showPanel('history');
    } else {
      showPanel(currentJobId ? 'results' : 'empty');
    }
  });
});

newJobBtn.addEventListener('click', () => {
  currentJobId = null;
  clearFile();
  showPanel('empty');
});

retryBtn.addEventListener('click', () => {
  showPanel('empty');
});

// ─── History ────────────────────────────────────────────────────────────────
async function loadHistory() {
  try {
    const r    = await fetch('/api/jobs');
    const jobs = await r.json();
    historyList.innerHTML = '';
    if (!jobs.length) {
      historyList.innerHTML = '<p style="color:var(--text-dim);font-size:14px">No jobs yet.</p>';
      return;
    }
    jobs.forEach(j => {
      const el = document.createElement('div');
      el.className = 'history-item';
      el.innerHTML = `
        <div class="history-status ${j.status}"></div>
        <div class="history-info">
          <span class="history-id">${j.id.slice(0, 8)}…</span>
          <div class="history-stage">${j.stage} — ${j.status}</div>
        </div>
        <span class="history-time">${relTime(j.created_at)}</span>`;
      el.addEventListener('click', () => {
        if (j.status === 'done') {
          currentJobId = j.id;
          fetchJobResult(j.id);
          navBtns.forEach(b => b.classList.remove('active'));
          document.querySelector('[data-panel="upload"]').classList.add('active');
        }
      });
      historyList.appendChild(el);
    });
  } catch (e) {
    historyList.innerHTML = '<p style="color:var(--text-dim)">Could not load history.</p>';
  }
}

// ─── Panel visibility ──────────────────────────────────────────────────────
function showPanel(name) {
  emptyState.classList.add('hidden');
  progressPanel.classList.add('hidden');
  resultsPanel.classList.add('hidden');
  historyPanel.classList.add('hidden');
  errorPanel.classList.add('hidden');
  if (name === 'empty')    emptyState.classList.remove('hidden');
  if (name === 'progress') progressPanel.classList.remove('hidden');
  if (name === 'results')  resultsPanel.classList.remove('hidden');
  if (name === 'history')  historyPanel.classList.remove('hidden');
  if (name === 'error')    errorPanel.classList.remove('hidden');
}

function showError(msg) {
  $('errorMsg').textContent = msg;
  showPanel('error');
  setStatus('error', 'Error');
  toast(msg, 'error');
}

// ─── Status indicator ──────────────────────────────────────────────────────
function setStatus(state, label) {
  const dot  = apiStatus.querySelector('.status-dot');
  const lbl  = apiStatus.querySelector('.status-label');
  dot.className  = 'status-dot' + (state !== 'ready' ? ' ' + state : '');
  lbl.textContent = label;
}

// ─── Toast ─────────────────────────────────────────────────────────────────
function toast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  $('toastContainer').appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

// ─── Formatters ────────────────────────────────────────────────────────────
function fmtDuration(s) {
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}
function fmtTs(s) {
  if (s == null) return '';
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1);
  return m > 0 ? `${m}:${String(Math.floor(s % 60)).padStart(2,'0')}` : `${parseFloat(sec).toFixed(1)}s`;
}
function relTime(iso) {
  const diff = Date.now() - new Date(iso + 'Z').getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1)   return 'just now';
  if (m < 60)  return `${m}m ago`;
  return `${Math.floor(m / 60)}h ago`;
}

// ─── Init ──────────────────────────────────────────────────────────────────
loadSettings();
validateForm();
showPanel('empty');

})();
