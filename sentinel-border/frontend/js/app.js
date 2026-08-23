/**
 * SentinelBorder — app.js
 * Main controller: drag-and-drop, webcam stream, canvas snapshot, API dispatch.
 */

'use strict';

// ─── Element references ───────────────────────────────────────────────────────
const dropZone       = document.getElementById('drop-zone');
const fileInput      = document.getElementById('file-input');
const docPreview     = document.getElementById('doc-preview');
const docFilename    = document.getElementById('doc-filename');
const webcamFeed     = document.getElementById('webcam-feed');
const webcamCanvas   = document.getElementById('webcam-canvas');
const snapshotPreview = document.getElementById('snapshot-preview');
const btnStartCam    = document.getElementById('btn-start-cam');
const btnSnapshot    = document.getElementById('btn-snapshot');
const btnRun         = document.getElementById('btn-run');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingStep    = document.getElementById('loading-step');
const clockEl        = document.getElementById('clock');

// ─── State ────────────────────────────────────────────────────────────────────
let selectedFile  = null;   // The identity document File object
let snapshotBlob  = null;   // Blob from webcam canvas
let cameraStream  = null;   // MediaStream

// ─── Clock ────────────────────────────────────────────────────────────────────
function updateClock() {
  if (!clockEl) return;
  const now = new Date();
  const pad = n => String(n).padStart(2, '0');
  clockEl.textContent =
    `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())} IST`;
}
setInterval(updateClock, 1000);
updateClock();

// ─── Drag and Drop ────────────────────────────────────────────────────────────
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') fileInput.click();
});

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const files = e.dataTransfer?.files;
  if (files && files.length > 0) handleFileSelected(files[0]);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) handleFileSelected(fileInput.files[0]);
});

function handleFileSelected(file) {
  const allowed = ['image/jpeg', 'image/png', 'application/pdf'];
  if (!allowed.includes(file.type)) {
    showToast('Unsupported file type. Use JPG, PNG, or PDF.', 'error');
    return;
  }
  selectedFile = file;

  // Show filename
  docFilename.textContent = `📎 ${file.name}  (${(file.size / 1024).toFixed(1)} KB)`;
  docFilename.style.display = 'block';

  // Preview image
  if (file.type !== 'application/pdf') {
    const reader = new FileReader();
    reader.onload = ev => {
      docPreview.src = ev.target.result;
      docPreview.style.display = 'block';
    };
    reader.readAsDataURL(file);
  } else {
    docPreview.style.display = 'none';
    docPreview.src = '';
  }

  updateRunButton();
}

// ─── Webcam ───────────────────────────────────────────────────────────────────
btnStartCam.addEventListener('click', toggleCamera);

async function toggleCamera() {
  if (cameraStream) {
    stopCamera();
    return;
  }

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }
    });
    webcamFeed.srcObject = cameraStream;
    btnStartCam.textContent = '⛔ STOP CAM';
    btnSnapshot.disabled = false;
  } catch (err) {
    showToast(`Camera error: ${err.message}`, 'error');
  }
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  webcamFeed.srcObject = null;
  btnStartCam.textContent = '⚡ START CAM';
  btnSnapshot.disabled = true;
}

btnSnapshot.addEventListener('click', () => {
  if (!cameraStream) return;

  const w = webcamFeed.videoWidth  || 640;
  const h = webcamFeed.videoHeight || 480;

  webcamCanvas.width  = w;
  webcamCanvas.height = h;

  const ctx = webcamCanvas.getContext('2d');
  ctx.drawImage(webcamFeed, 0, 0, w, h);

  webcamCanvas.toBlob(blob => {
    snapshotBlob = blob;
    snapshotPreview.src = URL.createObjectURL(blob);
    snapshotPreview.style.display = 'block';
    showToast('Snapshot captured ✓', 'info');
    updateRunButton();
  }, 'image/jpeg', 0.92);
});

// ─── Run Button State ─────────────────────────────────────────────────────────
function updateRunButton() {
  btnRun.disabled = !selectedFile;
}

// ─── Run Triage ───────────────────────────────────────────────────────────────
btnRun.addEventListener('click', async () => {
  if (!selectedFile) return;

  setLoading(true, 'INITIALISING PIPELINE...');
  btnRun.disabled = true;

  const steps = [
    [400,  'MODULE 1: OCR & MRZ EXTRACTION...'],
    [800,  'MODULE 2: ICAO CHECKSUM VALIDATION...'],
    [1200, 'MODULE 3: FORENSIC ELA ANALYSIS...'],
    [1600, 'MODULE 4: BIOMETRIC FACE MATCHING...'],
    [2000, 'COMPUTING COMPOSITE THREAT SCORE...'],
  ];

  // Animate loading steps
  for (const [delay, msg] of steps) {
    setTimeout(() => {
      if (loadingStep) loadingStep.textContent = msg;
    }, delay);
  }

  try {
    const data = await submitScreening(selectedFile, snapshotBlob);
    renderResult(data);
    showToast(`Triage complete — ${data.threat_level} (Score: ${data.composite_risk_score})`,
              data.threat_level === 'RED' ? 'error' : data.threat_level === 'YELLOW' ? 'warn' : 'ok');
  } catch (err) {
    showToast(`API Error: ${err.message}`, 'error');
    console.error('Screening error:', err);
  } finally {
    setLoading(false);
    btnRun.disabled = false;
  }
});

// ─── Loading overlay ──────────────────────────────────────────────────────────
function setLoading(visible, message = '') {
  if (visible) {
    loadingOverlay.classList.add('visible');
    if (loadingStep && message) loadingStep.textContent = message;
  } else {
    loadingOverlay.classList.remove('visible');
  }
}

// ─── Toast notification ───────────────────────────────────────────────────────
function showToast(message, type = 'info') {
  const existing = document.getElementById('sentinel-toast');
  if (existing) existing.remove();

  const colors = { ok: '#00ff88', warn: '#ffcc00', error: '#ff2244', info: '#0ff0fc' };
  const toast = document.createElement('div');
  toast.id = 'sentinel-toast';
  toast.style.cssText = `
    position: fixed; bottom: 24px; right: 24px; z-index: 9999;
    background: #121824; border: 1px solid ${colors[type] || colors.info};
    color: ${colors[type] || colors.info}; padding: 10px 18px;
    font-family: 'Share Tech Mono', monospace; font-size: 0.72rem;
    border-radius: 6px; letter-spacing: 1px;
    box-shadow: 0 0 20px ${colors[type]}44;
    animation: slide-in 0.2s ease;
    max-width: 400px;
  `;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}
