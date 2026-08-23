"""
SentinelBorder — FastAPI Application Entrypoint
POST /api/v1/screen — multipart document screening pipeline
GET  /               — serves the tactical frontend SPA
"""

from __future__ import annotations

import sys
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure backend/ is on the path when running from any CWD
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    FORENSICS_SHADOW_MODE,
    THREAT_GREEN_MAX,
    THREAT_YELLOW_MAX,
    WEIGHT_CHECKSUM_FAILED,
    WEIGHT_ELA_TAMPER,
    WEIGHT_EXPIRED,
    WEIGHT_FACE_MISMATCH,
    WEIGHT_METADATA_ANOMALY,
    WEIGHT_QR_DATA_MISMATCH,
    WEIGHT_VIZ_MRZ_MISMATCH,
)
from modules.ocr_engine import run_ocr
from modules.validator import run_validation, check_field_vision_consistency
from modules.forensics import run_forensics
from modules.biometrics import run_biometrics
from utils.helpers import get_logger

log = get_logger("app")

# ─── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="SentinelBorder",
    description="Edge-native document screening & biometric triage system — SSB / MHA SIH 26188",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Composite Threat Score ────────────────────────────────────────────────────

def compute_threat_score(
    checksum_failed: bool,
    viz_mrz_mismatch: bool,
    ela_tamper: bool,
    face_mismatch: bool,
    metadata_anomaly: bool,
    document_expired: bool,
    qr_data_mismatch: bool = False,
) -> tuple[int, str]:
    """
    Threat_Score = Min(100,
      35*Checksum_Failed + 30*VIZ_MRZ_Mismatch + 35*ELA_Tamper +
      40*Face_Mismatch  + 20*Metadata_Anomaly  + 25*Document_Expired
    )
    score += WEIGHT_QR_DATA_MISMATCH * int(qr_data_mismatch)
    Returns (score: int, level: "GREEN" | "YELLOW" | "RED").
    """
    score = (
        WEIGHT_CHECKSUM_FAILED  * int(checksum_failed)  +
        WEIGHT_VIZ_MRZ_MISMATCH * int(viz_mrz_mismatch) +
        WEIGHT_ELA_TAMPER       * int(ela_tamper)        +
        WEIGHT_FACE_MISMATCH    * int(face_mismatch)     +
        WEIGHT_METADATA_ANOMALY * int(metadata_anomaly)  +
        WEIGHT_EXPIRED          * int(document_expired)
    )
    score = min(100, score)

    if score < THREAT_GREEN_MAX:
        level = "GREEN"
    elif score < THREAT_YELLOW_MAX:
        level = "YELLOW"
    else:
        level = "RED"

    return score, level


# ─── Screening Endpoint ────────────────────────────────────────────────────────

@app.post("/api/v1/screen")
async def screen_document(
    document: UploadFile = File(..., description="Identity document image (JPG/PNG/PDF)"),
    live_photo: Optional[UploadFile] = File(None, description="Live webcam snapshot (JPG/PNG)"),
):
    t_start = time.perf_counter()
    log.info("▶ Screening request received — file='%s' type='%s'",
             document.filename, document.content_type)

    # ── Read bytes ────────────────────────────────────────────────────────────
    doc_bytes = await document.read()
    live_bytes = await live_photo.read() if live_photo else None

    if not doc_bytes:
        raise HTTPException(status_code=400, detail="Document file is empty.")

    content_type = document.content_type or "image/jpeg"

    # ── Module 1: OCR & MRZ ───────────────────────────────────────────────────
    log.info("  [M1] Running OCR & MRZ extraction...")
    ocr = run_ocr(doc_bytes, content_type)

    # ── Gatekeeper: Reject Non-Government IDs ─────────────────────────────────
    if not ocr.is_government_id:
        elapsed = round(time.perf_counter() - t_start, 3)
        flags = [f"INVALID_DOCUMENT_TYPE: Detected {ocr.id_type}. {ocr.id_reasoning}"]
        flags.extend(ocr.errors)
        log.warning("◀ Screening halted — Non-government ID detected (type='%s')", ocr.id_type)
        return JSONResponse(content={
            "status": "success",
            "threat_level": "RED",
            "composite_risk_score": 100,
            "processing_time_s": elapsed,
            "flags": flags,
            "extracted_data": {
                "name": "",
                "surname": "",
                "given_names": "",
                "doc_number": "",
                "doc_type": ocr.id_type or "UNKNOWN",
                "nationality": "",
                "dob": "",
                "expiry": "",
                "expiry_parsed": "",
                "sex": "",
                "issuing_country": "",
                "address": "",
                "mrz_detected": False,
                "mrz_raw": "",
                "viz_text": ocr.viz_text[:500] if ocr.viz_text else "",
                "ocr_engine": ocr.ocr_engine_used,
                "checksums": {
                    "doc_number_ok": False,
                    "dob_ok": False,
                    "expiry_ok": False,
                    "composite_ok": False,
                    "any_failed": True,
                },
                "document_expired": False,
                "viz_mrz_mismatch": False,
            },
            "forensic_analysis": {},
            "biometric_verification": {},
        })


    # ── Module 2: Validation ──────────────────────────────────────────────────
    log.info("  [M2] Running document validation...")
    val = run_validation(ocr)

    # ── Module 3: Forensics ───────────────────────────────────────────────────
    log.info("  [M3] Running forensic analysis...")
    # Retrieve vision fields from OCR result for shadow-mode cross-check.
    vision_fields: dict = ocr.vision_extracted_data
    if FORENSICS_SHADOW_MODE and not ocr.mrz_detected and ocr.doc_number:
        # For non-MRZ docs both sources agree by construction; cross-check is a no-op.
        pass
    forensics = run_forensics(
        doc_bytes, content_type, ocr.doc_type, ocr.doc_number,
        vision_fields=vision_fields,
        text_bboxes=ocr.text_bboxes
    )

    # ── Shadow: Field-Vision Cross-check (MRZ docs with Gemini corroboration) ─
    if FORENSICS_SHADOW_MODE and ocr.mrz_detected:
        _fvc_mismatch, _fvc_discrepancies = check_field_vision_consistency(ocr, vision_fields)
        forensics.field_vision_mismatch = _fvc_mismatch
        forensics.field_vision_discrepancies = _fvc_discrepancies
        if _fvc_mismatch:
            forensics.shadow_signal_count += 1

    # ── Module 4: Biometrics ──────────────────────────────────────────────────
    log.info("  [M4] Running biometric verification...")
    bio = run_biometrics(doc_bytes, live_bytes)

    # ── Composite Threat Score ─────────────────────────────────────────────────
    face_mismatch_flag = bio.status in ("mismatch",) or (
        live_bytes is not None and not bio.face_detected_live
    )
    score, level = compute_threat_score(
        checksum_failed=val.any_checksum_failed,
        viz_mrz_mismatch=val.viz_mrz_mismatch,
        # V2 only escalates corroborated visual evidence, avoiding a single ELA
        # anomaly from a screenshot or messaging-app recompression.
        ela_tamper=forensics.tamper_evidence_detected,
        face_mismatch=face_mismatch_flag,
        metadata_anomaly=forensics.metadata_anomaly,
        document_expired=val.document_expired,
        qr_data_mismatch=forensics.qr_data_mismatch,
    )

    # ── Aggregate Flags ────────────────────────────────────────────────────────
    flags: list[str] = []
    flags.extend(val.warnings)
    flags.extend(forensics.warnings)
    flags.extend(bio.warnings)
    flags.extend(ocr.errors)

    elapsed = round(time.perf_counter() - t_start, 3)
    log.info("◀ Screening complete — score=%d level=%s elapsed=%.3fs", score, level, elapsed)

    return JSONResponse(content={
        "status": "success",
        "threat_level": level,
        "composite_risk_score": score,
        "processing_time_s": elapsed,
        "flags": flags,
        "extracted_data": {
            "name": f"{ocr.surname}, {ocr.given_names}".strip(", "),
            "surname": ocr.surname,
            "given_names": ocr.given_names,
            "doc_number": ocr.doc_number,
            "doc_type": ocr.doc_type,
            "nationality": ocr.nationality,
            "dob": ocr.dob,
            "expiry": ocr.expiry,
            "expiry_parsed": val.expiry_date_parsed,
            "sex": ocr.sex,
            "issuing_country": ocr.issuing_country,
            "address": ocr.address,
            "mrz_detected": ocr.mrz_detected,
            "mrz_raw": ocr.mrz_raw,
            "viz_text": ocr.viz_text[:500] if ocr.viz_text else "",
            "ocr_engine": ocr.ocr_engine_used,
            "checksums": {
                "doc_number_ok": val.checksum_doc_number,
                "dob_ok": val.checksum_dob,
                "expiry_ok": val.checksum_expiry,
                "composite_ok": val.checksum_composite,
                "any_failed": val.any_checksum_failed,
            },
            "document_expired": val.document_expired,
            "viz_mrz_mismatch": val.viz_mrz_mismatch,
        },
        "forensic_analysis": {
            "ela_tamper_detected": forensics.ela_tamper_detected,
            "ela_score": forensics.ela_score,
            "ela_heatmap_b64": forensics.ela_heatmap_b64,
            "edge_discontinuity_detected": forensics.edge_discontinuity_detected,
            "edge_score": forensics.edge_score,
            "metadata_anomaly": forensics.metadata_anomaly,
            "metadata_flags": forensics.metadata_flags,
            "ela_legacy_score": forensics.ela_legacy_score,
            "ela_v2_score": forensics.ela_v2_score,
            "ela_v2_detected": forensics.ela_v2_detected,
            "copy_move_detected": forensics.copy_move_detected,
            "copy_move_score": forensics.copy_move_score,
            "qr_detected": forensics.qr_detected,
            "qr_payload_sha256": forensics.qr_payload_sha256,
            "qr_data_mismatch": forensics.qr_data_mismatch,
            "region_anomalies": forensics.region_anomalies,
            "dynamic_region_anomalies": forensics.dynamic_region_anomalies,
            "vlm_tamper_detected": forensics.vlm_tamper_detected,
            "vlm_tamper_reason": forensics.vlm_tamper_reason,
            "independent_signal_count": forensics.independent_signal_count,
            "tamper_evidence_detected": forensics.tamper_evidence_detected,
            "forensic_confidence": forensics.forensic_confidence,
            # Shadow-mode keys — present but do not contribute to composite_risk_score
            # until FORENSICS_SHADOW_MODE=false after calibration.
            "text_consistency_score": forensics.text_consistency_score,
            "text_consistency_anomaly": forensics.text_consistency_anomaly,
            "photo_boundary_score": forensics.photo_boundary_score,
            "photo_boundary_anomaly": forensics.photo_boundary_anomaly,
            "field_vision_mismatch": forensics.field_vision_mismatch,
            "field_vision_discrepancies": forensics.field_vision_discrepancies,
            "shadow_signal_count": forensics.shadow_signal_count,
        },
        "biometric_verification": {
            "face_detected_doc": bio.face_detected_doc,
            "face_detected_live": bio.face_detected_live,
            "cosine_distance": bio.cosine_distance,
            "match": bio.match,
            "status": bio.status,
            "confidence_pct": bio.confidence_pct,
            "doc_face_crop_b64": bio.doc_face_crop_b64,
            "live_face_crop_b64": bio.live_face_crop_b64,
        },
    })


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/api/v1/health")
def health():
    return {"status": "online", "system": "SentinelBorder v1.0"}


# ─── Serve Frontend ───────────────────────────────────────────────────────────

_frontend_path = Path(__file__).parent.parent / "frontend"
if _frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_path), html=True), name="frontend")
