"""
test_api_response_shape.py — Regression tests for the /api/v1/screen endpoint.

Tests cover:
  - Required top-level keys are always present
  - extracted_data keys are stable
  - forensic_analysis keys are stable (including new shadow keys)
  - biometric_verification keys are stable
  - New shadow keys appear in every response
  - Health endpoint responds correctly

These tests use the FastAPI TestClient (no real network calls).
Heavy OCR/biometric models are not loaded because conftest uses synthetic images
that PassportEye will likely not detect an MRZ in — which is the expected code path
for Aadhaar/non-MRZ documents.  The tests only verify response shape, not accuracy.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

_backend = Path(__file__).parent.parent / "backend"
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from fastapi.testclient import TestClient
from app import app  # type: ignore[import-untyped]

client = TestClient(app, raise_server_exceptions=True)


# ─── Health ──────────────────────────────────────────────────────────────────

def test_health_endpoint_responds() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"


# ─── Required top-level keys ──────────────────────────────────────────────────

_TOP_LEVEL_KEYS = {
    "status", "threat_level", "composite_risk_score",
    "processing_time_s", "flags",
    "extracted_data", "forensic_analysis", "biometric_verification",
}

def _post_synthetic_document(doc_bytes: bytes, content_type: str = "image/jpeg") -> dict:
    response = client.post(
        "/api/v1/screen",
        files={"document": ("test.jpg", io.BytesIO(doc_bytes), content_type)},
    )
    assert response.status_code == 200, f"Unexpected status: {response.status_code}\n{response.text}"
    return response.json()


def test_screen_returns_200_on_synthetic_passport(synthetic_passport_bytes: bytes) -> None:
    data = _post_synthetic_document(synthetic_passport_bytes)
    assert data["status"] == "success"


def test_screen_top_level_keys_stable(synthetic_passport_bytes: bytes) -> None:
    data = _post_synthetic_document(synthetic_passport_bytes)
    missing = _TOP_LEVEL_KEYS - set(data.keys())
    assert not missing, f"Missing top-level keys: {missing}"


def test_screen_threat_level_is_valid_string(synthetic_passport_bytes: bytes) -> None:
    data = _post_synthetic_document(synthetic_passport_bytes)
    assert data["threat_level"] in ("GREEN", "YELLOW", "RED")


def test_screen_composite_risk_score_in_range(synthetic_passport_bytes: bytes) -> None:
    data = _post_synthetic_document(synthetic_passport_bytes)
    assert 0 <= data["composite_risk_score"] <= 100


# ─── extracted_data keys ──────────────────────────────────────────────────────

_EXTRACTED_DATA_KEYS = {
    "name", "surname", "given_names", "doc_number", "doc_type",
    "nationality", "dob", "expiry", "expiry_parsed", "sex",
    "issuing_country", "address",
    "mrz_detected", "mrz_raw", "viz_text", "ocr_engine",
    "checksums", "document_expired", "viz_mrz_mismatch",
}

def test_extracted_data_keys_stable(synthetic_passport_bytes: bytes) -> None:
    data = _post_synthetic_document(synthetic_passport_bytes)
    missing = _EXTRACTED_DATA_KEYS - set(data["extracted_data"].keys())
    assert not missing, f"extracted_data missing keys: {missing}"


_CHECKSUM_KEYS = {
    "doc_number_ok", "dob_ok", "expiry_ok", "composite_ok", "any_failed"
}

def test_checksums_sub_keys_stable(synthetic_passport_bytes: bytes) -> None:
    data = _post_synthetic_document(synthetic_passport_bytes)
    missing = _CHECKSUM_KEYS - set(data["extracted_data"]["checksums"].keys())
    assert not missing, f"checksums missing keys: {missing}"


# ─── forensic_analysis keys ───────────────────────────────────────────────────

_FORENSIC_KEYS = {
    "ela_tamper_detected", "ela_score", "ela_heatmap_b64",
    "edge_discontinuity_detected", "edge_score",
    "metadata_anomaly", "metadata_flags",
    "ela_legacy_score", "ela_v2_score", "ela_v2_detected",
    "copy_move_detected", "copy_move_score",
    "qr_detected", "qr_payload_sha256", "qr_data_mismatch",
    "region_anomalies", "dynamic_region_anomalies", "text_coordinate_anomalies",
    "independent_signal_count", "tamper_evidence_detected", "forensic_confidence",
    # Shadow keys — must always be present
    "text_consistency_score", "text_consistency_anomaly",
    "photo_boundary_score", "photo_boundary_anomaly",
    "field_vision_mismatch", "field_vision_discrepancies",
    "shadow_signal_count",
}

def test_forensic_analysis_keys_stable(synthetic_passport_bytes: bytes) -> None:
    data = _post_synthetic_document(synthetic_passport_bytes)
    missing = _FORENSIC_KEYS - set(data["forensic_analysis"].keys())
    assert not missing, f"forensic_analysis missing keys: {missing}"


def test_new_shadow_keys_present_in_response(synthetic_passport_bytes: bytes) -> None:
    """Shadow keys must appear even when FORENSICS_SHADOW_MODE=true (default)."""
    data = _post_synthetic_document(synthetic_passport_bytes)
    fa = data["forensic_analysis"]
    assert "text_consistency_score" in fa
    assert "photo_boundary_score" in fa
    assert "shadow_signal_count" in fa
    assert isinstance(fa["shadow_signal_count"], int)
    assert isinstance(fa["field_vision_discrepancies"], list)


def test_forensic_confidence_is_string(synthetic_passport_bytes: bytes) -> None:
    data = _post_synthetic_document(synthetic_passport_bytes)
    assert data["forensic_analysis"]["forensic_confidence"] in ("LOW", "MEDIUM", "HIGH")


# ─── biometric_verification keys ─────────────────────────────────────────────

_BIO_KEYS = {
    "face_detected_doc", "face_detected_live",
    "cosine_distance", "match", "status",
    "confidence_pct", "doc_face_crop_b64", "live_face_crop_b64",
}

def test_biometric_keys_stable(synthetic_passport_bytes: bytes) -> None:
    data = _post_synthetic_document(synthetic_passport_bytes)
    missing = _BIO_KEYS - set(data["biometric_verification"].keys())
    assert not missing, f"biometric_verification missing keys: {missing}"


def test_biometric_status_without_live_photo(synthetic_passport_bytes: bytes) -> None:
    """When no live photo is provided, status must be 'no_comparison'."""
    data = _post_synthetic_document(synthetic_passport_bytes)
    assert data["biometric_verification"]["status"] == "no_comparison"


# ─── Aadhaar path ─────────────────────────────────────────────────────────────

def test_screen_aadhaar_returns_200(synthetic_aadhaar_bytes: bytes) -> None:
    data = _post_synthetic_document(synthetic_aadhaar_bytes)
    assert data["status"] == "success"


def test_screen_aadhaar_top_level_keys_stable(synthetic_aadhaar_bytes: bytes) -> None:
    data = _post_synthetic_document(synthetic_aadhaar_bytes)
    missing = _TOP_LEVEL_KEYS - set(data.keys())
    assert not missing, f"Missing top-level keys for Aadhaar path: {missing}"


def test_screen_aadhaar_forensic_keys_stable(synthetic_aadhaar_bytes: bytes) -> None:
    data = _post_synthetic_document(synthetic_aadhaar_bytes)
    missing = _FORENSIC_KEYS - set(data["forensic_analysis"].keys())
    assert not missing, f"forensic_analysis missing keys on Aadhaar path: {missing}"
