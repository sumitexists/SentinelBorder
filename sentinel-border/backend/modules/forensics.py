"""Document-image integrity screening.

These checks produce review evidence, not proof of forgery. A high-risk visual
decision requires two independent image signals, or a QR/data mismatch.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import ExifTags, Image

from config import (COPY_MOVE_MIN_INLIERS, ELA_RECOMPRESS_QUALITY,
                    ELA_TAMPER_THRESHOLD, ELA_V2_THRESHOLD,
                    FORENSICS_V2_ENABLED, FORENSICS_SHADOW_MODE,
                    FORENSICS_USE_REGION_ANALYSIS, SUSPICIOUS_SOFTWARE_TAGS)
from utils.helpers import bytes_to_pil, get_logger, image_to_base64, pil_to_numpy

log = get_logger("forensics")


@dataclass
class ForensicsResult:
    # Existing API fields.
    ela_tamper_detected: bool = False
    ela_score: float = 0.0
    ela_heatmap_b64: str = ""
    edge_discontinuity_detected: bool = False
    edge_score: float = 0.0
    metadata_anomaly: bool = False
    metadata_flags: list[str] = field(default_factory=list)
    # New corroborating evidence.
    ela_legacy_score: float = 0.0
    ela_v2_score: float = 0.0
    ela_v2_detected: bool = False
    copy_move_detected: bool = False
    copy_move_score: float = 0.0
    qr_detected: bool = False
    qr_payload_sha256: str = ""
    qr_data_mismatch: bool = False
    region_anomalies: list[str] = field(default_factory=list)
    independent_signal_count: int = 0
    tamper_evidence_detected: bool = False
    forensic_confidence: str = "LOW"
    # Shadow-mode signals — populated when FORENSICS_SHADOW_MODE=true but
    # intentionally excluded from composite risk score until calibrated.
    text_consistency_score: float = 0.0
    text_consistency_anomaly: bool = False
    photo_boundary_score: float = 0.0
    photo_boundary_anomaly: bool = False
    field_vision_mismatch: bool = False
    field_vision_discrepancies: list[str] = field(default_factory=list)
    shadow_signal_count: int = 0
    warnings: list[str] = field(default_factory=list)


def _heatmap_and_score(diff: np.ndarray) -> tuple[float, str]:
    maximum = float(diff.max())
    scaled = (diff / maximum * 255).astype(np.uint8) if maximum else np.zeros_like(diff, dtype=np.uint8)
    gray = np.mean(scaled, axis=2).astype(np.uint8)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    image = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    return float(np.mean(diff)), image_to_base64(image, fmt="JPEG", quality=85)


def _run_ela_legacy(image: Image.Image) -> tuple[float, str]:
    """Preserve the old measurement during threshold calibration."""
    original = image.convert("RGB")
    first = io.BytesIO()
    original.save(first, format="JPEG", quality=100)
    first.seek(0)
    baseline = Image.open(first).convert("RGB")
    second = io.BytesIO()
    baseline.save(second, format="JPEG", quality=ELA_RECOMPRESS_QUALITY)
    second.seek(0)
    recompressed = Image.open(second).convert("RGB")
    return _heatmap_and_score(np.abs(np.asarray(baseline, dtype=np.float32) - np.asarray(recompressed, dtype=np.float32)))


def _run_ela_v2(image: Image.Image) -> tuple[float, str, np.ndarray]:
    """Compare source pixels directly with a controlled JPEG recompression."""
    original = image.convert("RGB")
    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=ELA_RECOMPRESS_QUALITY, optimize=False)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    diff = np.abs(np.asarray(original, dtype=np.float32) - np.asarray(recompressed, dtype=np.float32))
    score, heatmap = _heatmap_and_score(diff)
    return score, heatmap, np.mean(diff, axis=2)


def _document_regions(doc_type: str) -> dict[str, tuple[float, float, float, float]]:
    common = {"header": (0.05, 0.04, 0.90, 0.18), "identity": (0.05, 0.22, 0.90, 0.48)}
    profiles = {
        "PASSPORT": {"photo": (0.06, 0.25, 0.30, 0.45), "visual_data": (0.34, 0.25, 0.57, 0.43), "mrz": (0.04, 0.78, 0.92, 0.16)},
        "AADHAAR": {"photo": (0.06, 0.23, 0.30, 0.46), "identity_data": (0.38, 0.22, 0.52, 0.38), "uid": (0.20, 0.72, 0.68, 0.14)},
        "DRIVING_LICENCE": {"photo": (0.05, 0.22, 0.30, 0.52), "identity_data": (0.38, 0.20, 0.55, 0.48)},
        "VOTER_ID": {"photo": (0.06, 0.22, 0.31, 0.52), "identity_data": (0.39, 0.20, 0.54, 0.48)},
    }
    return {**common, **profiles.get(doc_type.upper(), {})}


def _run_region_analysis(ela_map: np.ndarray, doc_type: str) -> list[str]:
    """Only runs when FORENSICS_USE_REGION_ANALYSIS is enabled."""
    if not FORENSICS_USE_REGION_ANALYSIS:
        return []
    height, width = ela_map.shape
    median = float(np.median(ela_map))
    mad = float(np.median(np.abs(ela_map - median))) + 1e-6
    anomalies: list[str] = []
    for name, (x, y, w, h) in _document_regions(doc_type).items():
        x1, y1 = int(x * width), int(y * height)
        x2, y2 = min(width, int((x + w) * width)), min(height, int((y + h) * height))
        region = ela_map[y1:y2, x1:x2]
        if not region.size:
            continue
        region_median = float(np.median(region))
        robust_z = 0.6745 * (region_median - median) / mad
        if robust_z >= 4.0 and region_median > ELA_V2_THRESHOLD:
            anomalies.append(f"Unusually high recompression residual in {name} region (z={robust_z:.1f}).")
    return anomalies


def _run_text_consistency(image: Image.Image) -> tuple[bool, float]:
    """Detect inconsistent noise / DPI in the text-strip region (upper 60% of image).

    Strategy: compute the Laplacian variance of small 16×16 non-overlapping blocks
    across the text area.  A genuine printed document has a relatively uniform noise
    floor.  A composited document where text was digitally inserted will have blocks
    with anomalously high or low variance compared to the median.

    Returns (anomaly_detected, score) where score is the percentage of outlier blocks.
    This runs in shadow mode only — it does not affect the composite risk score.
    """
    try:
        arr = pil_to_numpy(image)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # Only examine the upper 60% — MRZ is deliberately excluded to avoid
        # false positives from the mono-spaced OCR-B font.
        text_zone = gray[: int(h * 0.60), :]
        if text_zone.size == 0:
            return False, 0.0
        block_size = 16
        variances = []
        for r in range(0, text_zone.shape[0] - block_size, block_size):
            for c in range(0, text_zone.shape[1] - block_size, block_size):
                block = text_zone[r : r + block_size, c : c + block_size].astype(np.float32)
                lap = cv2.Laplacian(block, cv2.CV_32F)
                variances.append(float(np.var(lap)))
        if not variances:
            return False, 0.0
        median = float(np.median(variances))
        mad = float(np.median(np.abs(np.array(variances) - median))) + 1e-6
        outliers = sum(0.6745 * abs(v - median) / mad >= 5.0 for v in variances)
        score = round(outliers / len(variances) * 100.0, 2)
        # Threshold: >8% outlier blocks is suspicious — expect false positives on
        # heavily compressed WhatsApp scans; calibrate before enabling in score.
        return score > 8.0, score
    except Exception as exc:
        log.debug("Text consistency check failed: %s", exc)
        return False, 0.0


def _run_photo_boundary(image: Image.Image, doc_type: str) -> tuple[bool, float]:
    """Detect hard-cut paste boundaries around the photo region.

    Strategy: crop the photo region defined in `_document_regions`, then measure
    the gradient magnitude along its perimeter.  A genuine embedded photo has a
    smooth, printed transition to the document background.  A digitally pasted
    portrait produces an abrupt step-edge along the crop boundary.

    Returns (anomaly_detected, score).  Shadow mode only.
    """
    try:
        regions = _document_regions(doc_type)
        if "photo" not in regions:
            return False, 0.0
        arr = pil_to_numpy(image)
        h, w = arr.shape[:2]
        rx, ry, rw, rh = regions["photo"]
        x1, y1 = int(rx * w), int(ry * h)
        x2, y2 = min(w, int((rx + rw) * w)), min(h, int((ry + rh) * h))
        if x2 - x1 < 20 or y2 - y1 < 20:
            return False, 0.0
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        # Compute Sobel gradient magnitude across the full image
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        # Perimeter mask: 8-px border around the photo crop
        perimeter = np.zeros_like(magnitude, dtype=bool)
        border = 8
        perimeter[y1 : y1 + border, x1:x2] = True   # top
        perimeter[y2 - border : y2, x1:x2] = True   # bottom
        perimeter[y1:y2, x1 : x1 + border] = True   # left
        perimeter[y1:y2, x2 - border : x2] = True   # right
        perimeter_vals = magnitude[perimeter]
        interior_vals = magnitude[y1:y2, x1:x2].flatten()
        if not perimeter_vals.size or not interior_vals.size:
            return False, 0.0
        ratio = float(np.mean(perimeter_vals)) / (float(np.mean(interior_vals)) + 1e-6)
        # A ratio > 3.5 means the photo border is ~3.5× sharper than its interior
        # — consistent with a hard-cut paste.  Threshold is intentionally conservative.
        score = round(ratio, 4)
        return ratio > 3.5, score
    except Exception as exc:
        log.debug("Photo boundary check failed: %s", exc)
        return False, 0.0


def _run_edge_discontinuity(image: Image.Image) -> tuple[bool, float]:
    gray = cv2.cvtColor(pil_to_numpy(image), cv2.COLOR_BGR2GRAY)
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    if min(laplacian.shape) < 80:
        return False, 0.0
    values = [float(np.mean(block)) for row in np.array_split(laplacian, 8, axis=0) for block in np.array_split(row, 8, axis=1)]
    median = float(np.median(values))
    mad = float(np.median(np.abs(np.asarray(values) - median))) + 1e-6
    outliers = sum(0.6745 * abs(value - median) / mad >= 4.5 for value in values)
    return outliers >= 2, round(outliers / len(values) * 100.0, 2)


def _run_copy_move(image: Image.Image) -> tuple[bool, float]:
    """Look for spatially separated, geometrically consistent cloned features."""
    gray = cv2.cvtColor(pil_to_numpy(image), cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = cv2.ORB_create(nfeatures=1500, fastThreshold=12).detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) < COPY_MOVE_MIN_INLIERS * 2:
        return False, 0.0
    min_separation = min(gray.shape[:2]) * 0.12
    pairs = []
    for candidates in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors, descriptors, k=3):
        if len(candidates) < 3:
            continue
        match, alternate = candidates[1], candidates[2]  # first candidate is self-match
        src, dst = keypoints[match.queryIdx].pt, keypoints[match.trainIdx].pt
        if (match.distance < 0.72 * alternate.distance and match.queryIdx < match.trainIdx
                and np.hypot(src[0] - dst[0], src[1] - dst[1]) >= min_separation):
            pairs.append(match)
    if len(pairs) < COPY_MOVE_MIN_INLIERS:
        return False, 0.0
    source = np.float32([keypoints[m.queryIdx].pt for m in pairs]).reshape(-1, 1, 2)
    destination = np.float32([keypoints[m.trainIdx].pt for m in pairs]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(source, destination, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return inliers >= COPY_MOVE_MIN_INLIERS, round(inliers / max(len(keypoints), 1) * 100.0, 2)


def _run_qr_consistency(image: Image.Image, doc_number: str) -> tuple[bool, str, bool]:
    payload, _, _ = cv2.QRCodeDetector().detectAndDecode(pil_to_numpy(image))
    if not payload:
        return False, "", False
    payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    expected = "".join(ch for ch in doc_number.upper() if ch.isalnum())
    actual = "".join(ch for ch in payload.upper() if ch.isalnum())
    # Encrypted issuer payloads are deliberately treated as neutral, not mismatched.
    return True, payload_hash, bool(expected and len(expected) >= 6 and actual and expected not in actual)


def _audit_exif(image: Image.Image) -> list[str]:
    flags: list[str] = []
    try:
        for tag_id, value in image.getexif().items():
            if any(editor in str(value).lower() for editor in SUSPICIOUS_SOFTWARE_TAGS):
                flags.append(f"EXIF '{ExifTags.TAGS.get(tag_id, str(tag_id))}' contains an editor signature: '{value}'")
    except Exception as exc:
        log.debug("EXIF audit skipped: %s", exc)
    return flags


def _audit_pdf_metadata(raw_bytes: bytes) -> list[str]:
    flags: list[str] = []
    try:
        import pikepdf
        with pikepdf.open(io.BytesIO(raw_bytes)) as pdf:
            for key in ("/Creator", "/Producer", "/Author"):
                value = pdf.docinfo.get(key)
                if value and any(editor in str(value).lower() for editor in SUSPICIOUS_SOFTWARE_TAGS):
                    flags.append(f"PDF metadata '{key}' contains an editor signature: '{value}'")
    except Exception as exc:
        log.debug("PDF metadata audit skipped: %s", exc)
    return flags


def _fuse_evidence(result: ForensicsResult) -> None:
    # Existing corroborated-evidence signals (always active).
    signals = [result.ela_v2_detected, result.edge_discontinuity_detected,
               result.copy_move_detected, bool(result.region_anomalies)]
    result.independent_signal_count = sum(signals)
    result.tamper_evidence_detected = result.qr_data_mismatch or result.independent_signal_count >= 2
    result.forensic_confidence = (
        "HIGH" if (result.qr_data_mismatch or result.independent_signal_count >= 3)
        else ("MEDIUM" if result.tamper_evidence_detected or result.metadata_anomaly else "LOW")
    )
    # Shadow signals — counted separately; do not affect tamper_evidence_detected.
    shadow_signals = [result.text_consistency_anomaly, result.photo_boundary_anomaly,
                      result.field_vision_mismatch]
    result.shadow_signal_count = sum(shadow_signals)


def run_forensics(
    raw_bytes: bytes,
    content_type: str = "image/jpeg",
    doc_type: str = "",
    doc_number: str = "",
    vision_fields: dict | None = None,
) -> ForensicsResult:
    """Run compatible and corroborated document-integrity analysis.

    Args:
        raw_bytes:     Raw document bytes.
        content_type:  MIME type of the document.
        doc_type:      Detected document type (e.g. PASSPORT, AADHAAR) for region analysis.
        doc_number:    Extracted document number, used for QR consistency check.
        vision_fields: Structured fields returned by Gemini/Ollama, used for
                       field-vision cross-check (shadow mode only).
    """
    result = ForensicsResult()
    try:
        image = bytes_to_pil(raw_bytes)
    except Exception as exc:
        result.warnings.append(f"Image load error: {exc}")
        return result

    # ── Core corroborated signals (always active) ─────────────────────────────
    try:
        legacy_score, legacy_heatmap = _run_ela_legacy(image)
        v2_score, v2_heatmap, ela_map = _run_ela_v2(image)
        result.ela_legacy_score, result.ela_v2_score = round(legacy_score, 4), round(v2_score, 4)
        result.ela_v2_detected = v2_score >= ELA_V2_THRESHOLD
        result.ela_score = result.ela_v2_score if FORENSICS_V2_ENABLED else result.ela_legacy_score
        result.ela_heatmap_b64 = v2_heatmap if FORENSICS_V2_ENABLED else legacy_heatmap
        result.ela_tamper_detected = result.ela_v2_detected if FORENSICS_V2_ENABLED else legacy_score >= ELA_TAMPER_THRESHOLD
        result.region_anomalies = _run_region_analysis(ela_map, doc_type)
    except Exception as exc:
        result.warnings.append(f"ELA analysis failed: {exc}")
    try:
        result.edge_discontinuity_detected, result.edge_score = _run_edge_discontinuity(image)
    except Exception as exc:
        result.warnings.append(f"Edge analysis failed: {exc}")
    try:
        result.copy_move_detected, result.copy_move_score = _run_copy_move(image)
    except Exception as exc:
        result.warnings.append(f"Copy-move analysis failed: {exc}")
    try:
        result.qr_detected, result.qr_payload_sha256, result.qr_data_mismatch = _run_qr_consistency(image, doc_number)
    except Exception as exc:
        result.warnings.append(f"QR analysis failed: {exc}")
    try:
        result.metadata_flags = _audit_pdf_metadata(raw_bytes) if "pdf" in content_type.lower() else _audit_exif(image)
        result.metadata_anomaly = bool(result.metadata_flags)
    except Exception as exc:
        result.warnings.append(f"Metadata audit failed: {exc}")

    # ── Shadow-mode signals (populate response; do NOT feed into risk score) ──
    if FORENSICS_SHADOW_MODE:
        try:
            result.text_consistency_anomaly, result.text_consistency_score = _run_text_consistency(image)
        except Exception as exc:
            log.debug("Text consistency signal skipped: %s", exc)
        try:
            result.photo_boundary_anomaly, result.photo_boundary_score = _run_photo_boundary(image, doc_type)
        except Exception as exc:
            log.debug("Photo boundary signal skipped: %s", exc)

    _fuse_evidence(result)

    if result.tamper_evidence_detected:
        result.warnings.append(
            f"Forensic review required: {result.independent_signal_count} corroborating "
            f"visual signal(s), confidence={result.forensic_confidence}."
        )
    if result.qr_data_mismatch:
        result.warnings.append("QR/text consistency mismatch detected — verify document issuer data.")
    result.warnings.extend(result.region_anomalies)
    result.warnings.extend(result.metadata_flags)
    return result
