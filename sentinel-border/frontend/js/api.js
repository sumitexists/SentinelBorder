/**
 * SentinelBorder — api.js
 * Fetch wrapper for POST /api/v1/screen and full DOM result rendering pipeline.
 */

'use strict';

const API_ENDPOINT = '/api/v1/screen';

/**
 * Submit the document (and optional live photo) to the screening API.
 * @param {File} docFile      - Identity document file
 * @param {Blob|null} liveBlob - Webcam snapshot blob, or null
 * @returns {Promise<Object>} - Parsed JSON response
 */
async function submitScreening(docFile, liveBlob) {
  const form = new FormData();
  form.append('document', docFile, docFile.name);
  if (liveBlob) {
    form.append('live_photo', liveBlob, 'snapshot.jpg');
  }

  const res = await fetch(API_ENDPOINT, {
    method: 'POST',
    body: form,
  });

  if (!res.ok) {
    let errMsg = `HTTP ${res.status}`;
    try {
      const errJson = await res.json();
      errMsg = errJson.detail || errMsg;
    } catch (_) {}
    throw new Error(errMsg);
  }

  return res.json();
}

// ─── Helper: format YYMMDD → readable date ───────────────────────────────────
function formatDate(yymmdd) {
  if (!yymmdd || yymmdd.length !== 6) return yymmdd || '—';
  const yy = parseInt(yymmdd.slice(0, 2), 10);
  const mm = yymmdd.slice(2, 4);
  const dd = yymmdd.slice(4, 6);
  // Dynamic sliding-window pivot: 20-year lookahead, matching backend logic.
  // e.g. in 2026: pivot = 46; yy 00-46 → 2000-2046; yy 47-99 → 1947-1999.
  const pivot = (new Date().getFullYear() % 100 + 20) % 100;
  const year = yy <= pivot ? 2000 + yy : 1900 + yy;
  return `${dd}/${mm}/${year}`;
}

// ─── Checksum tag updater ─────────────────────────────────────────────────────
function setCheckTag(id, passed, labelPass, labelFail) {
  const el = document.getElementById(id);
  if (!el) return;
  el.className = `check-tag ${passed ? 'pass' : 'fail'}`;
  el.textContent = `${passed ? '✓' : '✗'} ${passed ? labelPass : labelFail}`;
}

// ─── Gauge updater ────────────────────────────────────────────────────────────
function updateGauge(score) {
  const fill = document.getElementById('gauge-fill');
  const track = document.getElementById('gauge-track');
  const scoreVal = document.getElementById('score-value');

  if (!fill || !scoreVal) return;

  fill.style.width = `${score}%`;

  let color;
  if (score < 30)       color = 'var(--green)';
  else if (score < 70)  color = 'var(--yellow)';
  else                  color = 'var(--red)';

  fill.style.background = color;
  scoreVal.style.color = color;
  scoreVal.textContent = `${score} / 100`;

  if (track) track.setAttribute('aria-valuenow', score);
}

// ─── Threat badge updater ─────────────────────────────────────────────────────
function updateThreatBadge(level) {
  const badge = document.getElementById('threat-badge');
  const text  = document.getElementById('threat-level-text');
  if (!badge || !text) return;

  badge.className = `${level}`;
  const labels = {
    GREEN:  '🟢  CLEAR — LOW RISK',
    YELLOW: '🟡  REVIEW — MODERATE RISK',
    RED:    '🔴  DETAIN — HIGH RISK TAMPERING',
  };
  text.textContent = labels[level] || level;
}

// ─── Flags feed renderer ──────────────────────────────────────────────────────
function renderFlags(flags) {
  const feed = document.getElementById('flags-feed');
  if (!feed) return;

  feed.innerHTML = '';

  if (!flags || flags.length === 0) {
    feed.innerHTML = '<div class="idle-placeholder">✓ NO THREATS DETECTED — DOCUMENT CLEAR</div>';
    return;
  }

  flags.forEach(flagText => {
    const isRed    = /FAIL|MISMATCH|TAMPER|EXPIRED|DETAIN|CRITICAL/i.test(flagText);
    const isYellow = /REVIEW|WARNING|WARN|ANOMALY|MISMATCH/i.test(flagText);

    const cls  = isRed ? 'critical' : isYellow ? 'warning' : 'info';
    const icon = isRed ? '⚠' : isYellow ? '⚡' : 'ℹ';

    const item = document.createElement('div');
    item.className = `flag-item ${cls}`;
    item.innerHTML = `<span class="flag-icon">${icon}</span><span>${flagText}</span>`;
    feed.appendChild(item);
  });
}

// ─── Main result renderer ─────────────────────────────────────────────────────
function renderResult(data) {
  // ── Show credentials panel content ──────────────────────────────────────
  document.getElementById('credentials-idle').style.display = 'none';
  const cc = document.getElementById('credentials-content');
  cc.style.display = 'flex';

  const d = data.extracted_data || {};
  const setText = (id, val) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = val || '—';
    el.className = `field-value ${val ? '' : 'empty'}`;
  };

  setText('field-name',        d.name);
  setText('field-docnum',      d.doc_number);
  setText('field-nationality', d.nationality);
  setText('field-dob',         formatDate(d.dob));
  setText('field-sex',         d.sex === 'M' ? 'MALE' : d.sex === 'F' ? 'FEMALE' : d.sex);
  setText('field-expiry',      d.expiry_parsed || formatDate(d.expiry));
  setText('field-doctype',     d.doc_type);
  setText('field-country',     d.issuing_country);
  setText('field-address',     d.address);
  setText('field-engine',      (d.ocr_engine || '').toUpperCase());

  // MRZ raw
  const mrzEl = document.getElementById('mrz-raw-display');
  if (mrzEl) mrzEl.textContent = d.mrz_raw || 'NO MRZ DETECTED';

  // ── Checksum tags ─────────────────────────────────────────────────────
  const cs = d.checksums || {};
  setCheckTag('chk-docnum',    cs.doc_number_ok,  'DOC NUMBER ✓',  'DOC NUMBER ✗');
  setCheckTag('chk-dob',       cs.dob_ok,         'DOB ✓',         'DOB FORGED ✗');
  setCheckTag('chk-expiry',    cs.expiry_ok,      'EXPIRY ✓',      'EXPIRY FORGED ✗');
  setCheckTag('chk-composite', cs.composite_ok,   'COMPOSITE ✓',   'COMPOSITE ✗');
  setCheckTag('chk-expired',   !d.document_expired, 'VALID ✓',     'EXPIRED ✗');
  setCheckTag('chk-parity',    !d.viz_mrz_mismatch, 'PARITY OK ✓', 'VIZ/MRZ MISMATCH ✗');

  // ── Threat badge + gauge ──────────────────────────────────────────────
  updateThreatBadge(data.threat_level || 'GREEN');
  updateGauge(data.composite_risk_score || 0);

  // ── ELA Heatmap ───────────────────────────────────────────────────────
  const fa = data.forensic_analysis || {};
  const elaImg       = document.getElementById('ela-heatmap');
  const elaPlaceholder = document.getElementById('ela-placeholder');

  if (fa.ela_heatmap_b64) {
    elaImg.src = `data:image/jpeg;base64,${fa.ela_heatmap_b64}`;
    elaImg.style.display = 'block';
    if (elaPlaceholder) elaPlaceholder.style.display = 'none';
  }

  const elaStatRow = document.getElementById('ela-stat-row');
  const elaScoreEl = document.getElementById('ela-score-val');
  if (elaStatRow && elaScoreEl) {
    elaStatRow.style.display = 'flex';
    elaScoreEl.textContent   = `${fa.ela_score?.toFixed(3) ?? '—'} (${fa.ela_tamper_detected ? 'TAMPER DETECTED' : 'CLEAN'})`;
    elaScoreEl.className     = `stat-val ${fa.ela_tamper_detected ? 'fail' : 'ok'}`;
  }

  const edgeRow = document.getElementById('edge-stat-row');
  const edgeVal = document.getElementById('edge-stat-val');
  if (edgeRow && edgeVal) {
    edgeRow.style.display = 'flex';
    edgeVal.textContent   = `${fa.edge_score?.toFixed(1) ?? '—'}% (${fa.edge_discontinuity_detected ? 'SPLICING DETECTED' : 'CLEAN'})`;
    edgeVal.className     = `stat-val ${fa.edge_discontinuity_detected ? 'fail' : 'ok'}`;
  }

  // ── Biometric ─────────────────────────────────────────────────────────
  const bio = data.biometric_verification || {};

  const faceDoc      = document.getElementById('face-doc');
  const faceLive     = document.getElementById('face-live');
  const faceRegistry = document.getElementById('face-registry');

  if (bio.doc_face_crop_b64 && faceDoc) {
    faceDoc.src = `data:image/jpeg;base64,${bio.doc_face_crop_b64}`;
    faceDoc.style.display = 'block';
  }
  if (bio.live_face_crop_b64 && faceLive) {
    faceLive.src = `data:image/jpeg;base64,${bio.live_face_crop_b64}`;
    faceLive.style.display = 'block';
  }

  const bioRow  = document.getElementById('bio-stat-row');
  const bioDist = document.getElementById('bio-distance');
  if (bioRow && bioDist && bio.cosine_distance !== null && bio.cosine_distance !== undefined) {
    bioRow.style.display = 'flex';
    bioDist.textContent  = bio.cosine_distance.toFixed(4);
  }

  const bioStatusRow = document.getElementById('bio-status-row');
  const bioStatusVal = document.getElementById('bio-status-val');
  if (bioStatusRow && bioStatusVal && bio.status) {
    bioStatusRow.style.display = 'flex';
    const statusMap = {
      verified:     { label: 'VERIFIED ✓', cls: 'ok' },
      review:       { label: 'REVIEW ⚡',  cls: 'warn' },
      mismatch:     { label: 'MISMATCH ✗', cls: 'fail' },
      no_comparison:{ label: 'NO LIVE PHOTO', cls: '' },
    };
    const s = statusMap[bio.status] || { label: bio.status, cls: '' };
    bioStatusVal.textContent = s.label;
    bioStatusVal.className   = `stat-val ${s.cls}`;
  }

  // ── Registry Verification ───────────────────────────────────────────
  const rv = data.registry_verification || {};
  const dup = data.duplicate_passport_check || {};

  // Registry face crop
  if (rv.registry_face_crop_b64 && faceRegistry) {
    faceRegistry.src = `data:image/jpeg;base64,${rv.registry_face_crop_b64}`;
    faceRegistry.style.display = 'block';
  }

  // Registry record found badge
  const regRecordRow = document.getElementById('reg-record-row');
  const regRecordVal = document.getElementById('reg-record-val');
  if (regRecordRow && regRecordVal && rv.registry_status) {
    regRecordRow.style.display = 'flex';
    const statusMap = {
      VERIFIED:           { label: 'FOUND ✓',        cls: 'ok' },
      FACE_MISMATCH:      { label: 'MISMATCH ✗',     cls: 'fail' },
      NOT_IN_REGISTRY:    { label: 'NOT FOUND ✗',   cls: 'fail' },
      NO_EMBEDDING:       { label: 'NO PHOTO ⚡',    cls: 'warn' },
      SKIPPED:            { label: 'SKIPPED',          cls: '' },
    };
    const s = statusMap[rv.registry_status] || { label: rv.registry_status, cls: '' };
    regRecordVal.textContent = s.label;
    regRecordVal.className   = `stat-val ${s.cls}`;
  }

  // Pairwise distances
  function _showRegDist(rowId, valId, pairObj) {
    const row = document.getElementById(rowId);
    const val = document.getElementById(valId);
    if (!row || !val || !pairObj) return;
    if (pairObj.distance === null || pairObj.distance === undefined) return;
    row.style.display = 'flex';
    const d = pairObj.distance.toFixed(4);
    const matched = pairObj.match;
    val.textContent = `${d} (${matched ? 'MATCH' : 'FAIL'})`;
    val.className   = `stat-val ${matched ? 'ok' : 'fail'}`;
  }
  _showRegDist('reg-doc-reg-row', 'reg-doc-reg-val', rv.document_to_registry);
  _showRegDist('reg-reg-live-row', 'reg-reg-live-val', rv.registry_to_live);

  // Registry overall status
  const regStatusRow = document.getElementById('reg-status-row');
  const regStatusVal = document.getElementById('reg-status-val');
  if (regStatusRow && regStatusVal && rv.registry_status && rv.registry_status !== 'SKIPPED') {
    regStatusRow.style.display = 'flex';
    const isVerified = rv.overall_verified;
    regStatusVal.textContent = isVerified ? 'THREE-WAY VERIFIED ✓' : 'VERIFICATION FAILED ✗';
    regStatusVal.className   = `stat-val ${isVerified ? 'ok' : 'fail'}`;
  }

  // Duplicate passport alert
  const dupAlert = document.getElementById('dup-alert');
  const dupText  = document.getElementById('dup-alert-text');
  if (dupAlert && dupText && dup.duplicate_found) {
    const nums = (dup.duplicate_active_passports || [])
      .map(p => `${p.passport_number} (${p.issuing_country})`).join(', ');
    dupText.textContent = `DUPLICATE ACTIVE PASSPORT DETECTED: ${nums}`;
    dupAlert.style.display = 'flex';
  }

  // ── Flags ─────────────────────────────────────────────────────────────
  renderFlags(data.flags);

  // ── Processing time ───────────────────────────────────────────────────
  const procEl   = document.getElementById('proc-time');
  const procInfo = document.getElementById('processing-info');
  if (procEl && procInfo) {
    procEl.textContent    = `${data.processing_time_s?.toFixed(3) ?? '—'} s`;
    procInfo.style.display = 'flex';
  }
}
