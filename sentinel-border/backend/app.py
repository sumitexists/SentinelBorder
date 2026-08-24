"""
SentinelBorder — FastAPI Application Entrypoint
POST /api/v1/screen                — multipart document screening pipeline
POST /api/v1/registry/passports    — register a passport in the local registry
GET  /api/v1/registry/passports    — list registered passports (metadata only)
GET  /api/v1/registry/verify       — verify a passport exists in the registry
GET  /api/v1/health                — system health check
GET  /                             — serves the tactical frontend SPA
"""

from __future__ import annotations

import sys
import os
import time
import uuid
import struct
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Query
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
    REGISTRY_ENABLED,
    REGISTRY_DB_PATH,
    REGISTRY_PHOTOS_DIR,
    WEIGHT_DUPLICATE_PASSPORT,
)
from modules.ocr_engine import run_ocr
from modules.validator import run_validation, check_field_vision_consistency
from modules.forensics import run_forensics
from modules.biometrics import (
    run_biometrics,
    extract_embedding,
    run_three_way_verification,
    embedding_to_bytes,
)
from modules.registry import (
    init_db,
    register_passport,
    lookup_passport,
    get_active_passports_for_person,
    list_passports,
    count_passports,
)
from utils.helpers import get_logger, image_to_base64

log = get_logger("app")

# ─── Startup: initialise registry DB ─────────────────────────────────────────

if REGISTRY_ENABLED:
    try:
        init_db(REGISTRY_DB_PATH)
        Path(REGISTRY_PHOTOS_DIR).mkdir(parents=True, exist_ok=True)
    except Exception as _db_err:
        log.error("Registry DB init failed: %s", _db_err)

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
    duplicate_passport: bool = False,
) -> tuple[int, str]:
    """
    Threat_Score = Min(100,
      35*Checksum_Failed + 30*VIZ_MRZ_Mismatch + 35*ELA_Tamper +
      40*Face_Mismatch  + 20*Metadata_Anomaly  + 25*Document_Expired +
      30*QR_Data_Mismatch + 70*Duplicate_Passport
    )
    Returns (score: int, level: "GREEN" | "YELLOW" | "RED").
    """
    score = (
        WEIGHT_CHECKSUM_FAILED  * int(checksum_failed)  +
        WEIGHT_VIZ_MRZ_MISMATCH * int(viz_mrz_mismatch) +
        WEIGHT_ELA_TAMPER       * int(ela_tamper)        +
        WEIGHT_FACE_MISMATCH    * int(face_mismatch)     +
        WEIGHT_METADATA_ANOMALY * int(metadata_anomaly)  +
        WEIGHT_EXPIRED          * int(document_expired)  +
        WEIGHT_QR_DATA_MISMATCH * int(qr_data_mismatch)  +
        WEIGHT_DUPLICATE_PASSPORT * int(duplicate_passport)
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
            "registry_verification": {},
            "duplicate_passport_check": {},
        })


    # ── Module 2: Validation ──────────────────────────────────────────────────
    log.info("  [M2] Running document validation...")
    val = run_validation(ocr)

    # ── Module 3: Forensics ───────────────────────────────────────────────────
    log.info("  [M3] Running forensic analysis...")
    vision_fields: dict = ocr.vision_extracted_data
    if FORENSICS_SHADOW_MODE and not ocr.mrz_detected and ocr.doc_number:
        pass
    forensics = run_forensics(
        doc_bytes, content_type, ocr.doc_type, ocr.doc_number,
        vision_fields=vision_fields,
        text_bboxes=ocr.text_bboxes
    )

    # ── Shadow: Field-Vision Cross-check ──────────────────────────────────────
    if FORENSICS_SHADOW_MODE and ocr.mrz_detected:
        _fvc_mismatch, _fvc_discrepancies = check_field_vision_consistency(ocr, vision_fields)
        forensics.field_vision_mismatch = _fvc_mismatch
        forensics.field_vision_discrepancies = _fvc_discrepancies
        if _fvc_mismatch:
            forensics.shadow_signal_count += 1

    # ── Module 4: Biometrics ──────────────────────────────────────────────────
    log.info("  [M4] Running biometric verification...")
    bio = run_biometrics(doc_bytes, live_bytes)

    # ── Module 5: Registry Verification ──────────────────────────────────────
    registry_verification: dict = {}
    duplicate_passport_check: dict = {}
    duplicate_active_flag = False

    registry_qualified = (
        REGISTRY_ENABLED
        and ocr.mrz_detected
        and ocr.doc_number
        and ocr.issuing_country
        and not val.any_checksum_failed
        and not val.viz_mrz_mismatch
    )

    if registry_qualified:
        log.info("  [M5] Running registry verification...")
        registry_row = lookup_passport(REGISTRY_DB_PATH, ocr.doc_number, ocr.issuing_country)

        if registry_row is None:
            # Passport not in registry → manual review
            registry_verification = {
                "registry_record_found": False,
                "registry_status": "NOT_IN_REGISTRY",
                "document_to_live":     {"match": None, "distance": None},
                "document_to_registry": {"match": None, "distance": None},
                "registry_to_live":     {"match": None, "distance": None},
                "overall_verified": False,
                "registry_face_crop_b64": "",
            }
            duplicate_passport_check = {"checked": False, "duplicate_found": False,
                                        "duplicate_active_passports": [], "threat": "NONE"}
            log.info("  [M5] Passport %s/%s not found in registry — flagging for review.",
                     ocr.doc_number, ocr.issuing_country)
        else:
            embedding_blob = registry_row.get("face_embedding")

            if not embedding_blob:
                # Record exists but no embedding stored
                registry_verification = {
                    "registry_record_found": True,
                    "registry_status": "NO_EMBEDDING",
                    "document_to_live":     {"match": None, "distance": None},
                    "document_to_registry": {"match": None, "distance": None},
                    "registry_to_live":     {"match": None, "distance": None},
                    "overall_verified": False,
                    "registry_face_crop_b64": "",
                }
                duplicate_passport_check = {"checked": False, "duplicate_found": False,
                                            "duplicate_active_passports": [], "threat": "NONE"}
            else:
                # Run three-way verification
                three_way = run_three_way_verification(doc_bytes, live_bytes, embedding_blob)

                registry_status = "VERIFIED" if three_way["overall_verified"] else "FACE_MISMATCH"
                registry_verification = {
                    "registry_record_found": True,
                    "registry_status": registry_status,
                    **{k: v for k, v in three_way.items() if k != "registry_record_found"},
                }

                # Duplicate check — only if all faces matched
                if three_way["overall_verified"]:
                    other_active = get_active_passports_for_person(
                        REGISTRY_DB_PATH,
                        registry_row["person_id"],
                        registry_row["id"],
                    )
                    if other_active:
                        duplicate_active_flag = True
                        duplicate_passport_check = {
                            "checked": True,
                            "duplicate_found": True,
                            "duplicate_active_passports": [
                                {
                                    "passport_number": r["passport_number"],
                                    "issuing_country": r["issuing_country"],
                                    "expiry_date": r["expiry_date"],
                                }
                                for r in other_active
                            ],
                            "threat": "POSSIBLE_DUPLICATE_ACTIVE_PASSPORT",
                        }
                    else:
                        duplicate_passport_check = {
                            "checked": True,
                            "duplicate_found": False,
                            "duplicate_active_passports": [],
                            "threat": "NONE",
                        }
                else:
                    duplicate_passport_check = {
                        "checked": False,
                        "duplicate_found": False,
                        "duplicate_active_passports": [],
                        "threat": "NONE",
                    }
    else:
        registry_verification = {
            "registry_record_found": False,
            "registry_status": "SKIPPED",
            "document_to_live":     {"match": None, "distance": None},
            "document_to_registry": {"match": None, "distance": None},
            "registry_to_live":     {"match": None, "distance": None},
            "overall_verified": False,
            "registry_face_crop_b64": "",
        }
        duplicate_passport_check = {"checked": False, "duplicate_found": False,
                                    "duplicate_active_passports": [], "threat": "NONE"}

    # ── Composite Threat Score ─────────────────────────────────────────────────
    face_mismatch_flag = bio.status in ("mismatch",) or (
        live_bytes is not None and not bio.face_detected_live
    )
    score, level = compute_threat_score(
        checksum_failed=val.any_checksum_failed,
        viz_mrz_mismatch=val.viz_mrz_mismatch,
        ela_tamper=forensics.tamper_evidence_detected,
        face_mismatch=face_mismatch_flag,
        metadata_anomaly=forensics.metadata_anomaly,
        document_expired=val.document_expired,
        qr_data_mismatch=forensics.qr_data_mismatch,
        duplicate_passport=duplicate_active_flag,
    )

    # ── Aggregate Flags ────────────────────────────────────────────────────────
    flags: list[str] = []
    flags.extend(val.warnings)
    flags.extend(forensics.warnings)
    flags.extend(bio.warnings)
    flags.extend(ocr.errors)

    # Registry-specific flags
    rv_status = registry_verification.get("registry_status", "SKIPPED")
    if rv_status == "NOT_IN_REGISTRY":
        flags.append(
            "REGISTRY REVIEW: Passport not found in local registry — manual verification required."
        )
    elif rv_status == "FACE_MISMATCH":
        flags.append(
            "REGISTRY MISMATCH: Document/live face does not match registry photo — "
            "possible impersonation."
        )
    elif rv_status == "NO_EMBEDDING":
        flags.append(
            "REGISTRY REVIEW: Registry record exists but no face embedding stored — "
            "manual verification required."
        )

    if duplicate_active_flag:
        dup_nums = ", ".join(
            f"{d['passport_number']} ({d['issuing_country']})"
            for d in duplicate_passport_check.get("duplicate_active_passports", [])
        )
        flags.append(
            f"CRITICAL: POSSIBLE_DUPLICATE_ACTIVE_PASSPORT — same identity holds "
            f"another active passport: {dup_nums}"
        )

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
            "text_coordinate_anomalies": forensics.text_coordinate_anomalies,
            "vlm_tamper_detected": forensics.vlm_tamper_detected,
            "vlm_tamper_reason": forensics.vlm_tamper_reason,
            "independent_signal_count": forensics.independent_signal_count,
            "tamper_evidence_detected": forensics.tamper_evidence_detected,
            "forensic_confidence": forensics.forensic_confidence,
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
        "registry_verification": registry_verification,
        "duplicate_passport_check": duplicate_passport_check,
    })


# ─── Registry Endpoints ────────────────────────────────────────────────────────

@app.post("/api/v1/registry/passports")
async def registry_register_passport(
    passport_number:  str = Form(...),
    issuing_country:  str = Form(...),
    surname:          str = Form(...),
    given_names:      str = Form(...),
    dob:              str = Form(...),
    nationality:      str = Form(...),
    issue_date:       str = Form(""),
    expiry_date:      str = Form(""),
    status:           str = Form("active"),
    person_id:        str = Form(""),
    registry_photo: Optional[UploadFile] = File(None, description="Official registry face photo"),
):
    """
    Register a passport in the local SQLite registry.
    Optionally supply a registry_photo; if provided, an ArcFace embedding
    is generated and stored alongside the photo.
    """
    if not REGISTRY_ENABLED:
        raise HTTPException(status_code=503, detail="Registry module is disabled.")

    # Basic field validation
    for field_name, value in [
        ("passport_number", passport_number),
        ("issuing_country", issuing_country),
        ("surname", surname),
        ("given_names", given_names),
        ("dob", dob),
        ("nationality", nationality),
    ]:
        if not value or not value.strip():
            raise HTTPException(status_code=422, detail=f"Field '{field_name}' is required.")

    if status not in ("active", "expired", "revoked"):
        raise HTTPException(status_code=422, detail="status must be active, expired, or revoked.")

    # Photo handling
    photo_path_str = ""
    embedding_blob: Optional[bytes] = None
    embedding_model_name = ""
    embedding_dim = 0

    if registry_photo and registry_photo.filename:
        photo_bytes = await registry_photo.read()
        if photo_bytes:
            # Save photo with a safe generated filename
            safe_pnum = "".join(c for c in passport_number if c.isalnum()).upper()
            safe_country = "".join(c for c in issuing_country if c.isalnum()).upper()
            photo_filename = f"{safe_pnum}_{safe_country}_{uuid.uuid4().hex[:8]}.jpg"
            photo_path = Path(REGISTRY_PHOTOS_DIR) / photo_filename

            # Convert and save as JPEG
            from utils.helpers import bytes_to_pil
            try:
                img = bytes_to_pil(photo_bytes)
                img.save(str(photo_path), format="JPEG", quality=95)
                photo_path_str = str(photo_path)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Photo could not be read: {exc}")

            # Generate face embedding
            emb, _, err = extract_embedding(photo_bytes)
            if emb is None:
                # Clean up saved photo
                try:
                    photo_path.unlink()
                except OSError:
                    pass
                raise HTTPException(
                    status_code=422,
                    detail=f"Face detection failed in registry photo: {err}. "
                           "Please supply a clear, front-facing photo with exactly one face."
                )
            embedding_blob = embedding_to_bytes(emb)
            embedding_model_name = "ArcFace"
            embedding_dim = len(emb)

    # Atomic DB insert
    import sqlite3 as _sqlite3
    try:
        result = register_passport(
            REGISTRY_DB_PATH,
            passport_number=passport_number.strip().upper(),
            issuing_country=issuing_country.strip().upper(),
            surname=surname.strip().upper(),
            given_names=given_names.strip().upper(),
            dob=dob.strip(),
            nationality=nationality.strip().upper(),
            issue_date=issue_date.strip(),
            expiry_date=expiry_date.strip(),
            status=status.strip(),
            photo_path=photo_path_str,
            face_embedding=embedding_blob,
            embedding_model=embedding_model_name,
            embedding_dimension=embedding_dim,
            person_id=person_id.strip() or None,
        )
    except _sqlite3.IntegrityError:
        raise HTTPException(
            status_code=409,
            detail=f"Passport {passport_number}/{issuing_country} is already registered."
        )
    except Exception as exc:
        log.error("Registry insert error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Registry error: {exc}")

    return JSONResponse(status_code=201, content={
        "status": "registered",
        "passport_number": passport_number.strip().upper(),
        "issuing_country": issuing_country.strip().upper(),
        "person_id": result["person_id"],
        "passport_id": result["passport_id"],
        "embedding_stored": embedding_blob is not None,
    })


@app.get("/api/v1/registry/passports")
def registry_list_passports(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Return a paginated list of registered passports (metadata only, no embedding bytes)."""
    if not REGISTRY_ENABLED:
        raise HTTPException(status_code=503, detail="Registry module is disabled.")
    records = list_passports(REGISTRY_DB_PATH, limit=limit, offset=offset)
    total = count_passports(REGISTRY_DB_PATH)
    return {"total": total, "offset": offset, "limit": limit, "records": records}


@app.get("/api/v1/registry/verify")
def registry_verify_passport(
    passport_number: str = Query(..., description="Passport number to look up"),
    issuing_country: str = Query(..., description="Two or three-letter issuing country code"),
):
    """
    Lightweight endpoint: confirm whether a passport exists in the registry.
    Returns { found, passport_number, person_id, status } or { found: false }.
    """
    if not REGISTRY_ENABLED:
        raise HTTPException(status_code=503, detail="Registry module is disabled.")
    row = lookup_passport(REGISTRY_DB_PATH, passport_number.strip().upper(),
                          issuing_country.strip().upper())
    if row is None:
        return {"found": False, "passport_number": passport_number, "issuing_country": issuing_country}
    return {
        "found": True,
        "passport_number": row["passport_number"],
        "issuing_country": row["issuing_country"],
        "person_id": row["person_id"],
        "status": row["status"],
        "surname": row["surname"],
        "given_names": row["given_names"],
        "embedding_stored": row["face_embedding"] is not None,
    }


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/api/v1/health")
def health():
    return {"status": "online", "system": "SentinelBorder v1.0", "registry_enabled": REGISTRY_ENABLED}


# ─── Serve Frontend ───────────────────────────────────────────────────────────

_frontend_path = Path(__file__).parent.parent / "frontend"
if _frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_path), html=True), name="frontend")
