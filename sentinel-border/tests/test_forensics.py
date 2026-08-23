"""
test_forensics.py — Regression tests for Module 3: Forensic Tampering Detection.

Tests cover:
  - ELA legacy and ELA-v2 return non-negative numeric scores
  - Copy-move detection returns False on clean images
  - Metadata audit returns no flags on clean PNGs
  - Editor EXIF signature correctly flagged
  - Region anomaly function returns a list (may be empty on synthetics)
  - Tamper evidence requires >= 2 independent signals
  - QR consistency: no QR detected on text-only image
  - ForensicsResult API shape stability (all expected keys present)
  - Shadow-mode fields present and typed correctly
  - Text-consistency and photo-boundary signals return expected types
"""
from __future__ import annotations

import io
import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_backend = Path(__file__).parent.parent / "backend"
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from modules.forensics import (
    ForensicsResult,
    _run_ela_legacy,
    _run_ela_v2,
    _run_copy_move,
    _run_edge_discontinuity,
    _run_qr_consistency,
    _run_text_consistency,
    _run_photo_boundary,
    _audit_exif,
    _fuse_evidence,
    run_forensics,
)
from utils.helpers import bytes_to_pil


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _pil_to_bytes(img: Image.Image, fmt: str = "JPEG", quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return buf.getvalue()


def _make_clean_pil(width: int = 300, height: int = 200) -> Image.Image:
    return Image.fromarray(np.full((height, width, 3), 220, dtype=np.uint8))


# ─── ELA ─────────────────────────────────────────────────────────────────────

def test_ela_legacy_returns_non_negative_score() -> None:
    score, _heatmap = _run_ela_legacy(_make_clean_pil())
    assert isinstance(score, float)
    assert score >= 0.0


def test_ela_v2_returns_non_negative_score() -> None:
    score, _heatmap, ela_map = _run_ela_v2(_make_clean_pil())
    assert isinstance(score, float)
    assert score >= 0.0
    assert ela_map.ndim == 2


def test_ela_v2_map_shape_matches_image() -> None:
    img = _make_clean_pil(400, 300)
    _, _, ela_map = _run_ela_v2(img)
    assert ela_map.shape == (300, 400)


def test_ela_legacy_score_lower_for_clean_image() -> None:
    """A clean, uniformly generated JPEG should have a low ELA score."""
    score, _ = _run_ela_legacy(_make_clean_pil())
    # Generous upper bound — synthetic uniform images compress very efficiently.
    assert score < 20.0


# ─── Copy-move ───────────────────────────────────────────────────────────────

def test_copy_move_not_detected_on_clean_image(clean_white_jpeg: bytes) -> None:
    img = bytes_to_pil(clean_white_jpeg)
    detected, score = _run_copy_move(img)
    assert detected is False
    assert score >= 0.0


def test_copy_move_score_is_float(clean_white_jpeg: bytes) -> None:
    img = bytes_to_pil(clean_white_jpeg)
    _, score = _run_copy_move(img)
    assert isinstance(score, float)


# ─── Edge discontinuity ───────────────────────────────────────────────────────

def test_edge_discontinuity_not_detected_on_uniform_image() -> None:
    img = _make_clean_pil(400, 300)
    detected, score = _run_edge_discontinuity(img)
    assert isinstance(detected, bool)
    assert isinstance(score, float)
    # A uniform image has no edge outliers.
    assert detected is False


def test_edge_discontinuity_returns_false_on_tiny_image() -> None:
    """Images smaller than the 80-px minimum must be handled gracefully."""
    tiny = Image.fromarray(np.full((50, 50, 3), 200, dtype=np.uint8))
    detected, score = _run_edge_discontinuity(tiny)
    assert detected is False
    assert score == 0.0


# ─── QR consistency ──────────────────────────────────────────────────────────

def test_qr_not_detected_on_plain_image(clean_white_jpeg: bytes) -> None:
    img = bytes_to_pil(clean_white_jpeg)
    detected, sha, mismatch = _run_qr_consistency(img, "ABC123")
    assert detected is False
    assert sha == ""
    assert mismatch is False


# ─── Metadata audit ──────────────────────────────────────────────────────────

def test_metadata_no_flags_on_clean_png() -> None:
    img = _make_clean_pil()
    flags = _audit_exif(img)
    assert flags == []


def test_metadata_flags_editor_signature(editor_exif_jpeg: bytes) -> None:
    img = bytes_to_pil(editor_exif_jpeg)
    flags = _audit_exif(img)
    assert len(flags) > 0
    assert any("photoshop" in f.lower() or "adobe" in f.lower() for f in flags)


# ─── Text consistency (shadow signal) ────────────────────────────────────────

def test_text_consistency_returns_bool_and_float() -> None:
    img = _make_clean_pil(600, 400)
    detected, score = _run_text_consistency(img)
    assert isinstance(detected, bool)
    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0


def test_text_consistency_not_anomalous_on_clean_image() -> None:
    """A uniform image has no Laplacian variance outliers."""
    img = _make_clean_pil(600, 400)
    detected, score = _run_text_consistency(img)
    # Uniform images produce near-zero variance in every block → no outliers.
    assert detected is False


# ─── Photo boundary (shadow signal) ──────────────────────────────────────────

def test_photo_boundary_returns_false_on_unknown_doc_type() -> None:
    """No photo region is defined for UNKNOWN doc_type — must return (False, 0.0)."""
    img = _make_clean_pil(400, 300)
    detected, score = _run_photo_boundary(img, "UNKNOWN")
    assert detected is False
    assert score == 0.0


def test_photo_boundary_returns_float_score_for_passport() -> None:
    img = _make_clean_pil(600, 400)
    detected, score = _run_photo_boundary(img, "PASSPORT")
    assert isinstance(detected, bool)
    assert isinstance(score, float)
    assert score >= 0.0


# ─── Tamper evidence fusion ───────────────────────────────────────────────────

def test_tamper_requires_two_independent_signals() -> None:
    """A single signal must not set tamper_evidence_detected."""
    result = ForensicsResult(ela_v2_detected=True)
    _fuse_evidence(result)
    assert result.tamper_evidence_detected is False
    assert result.independent_signal_count == 1


def test_tamper_detected_with_two_signals() -> None:
    result = ForensicsResult(ela_v2_detected=True, copy_move_detected=True)
    _fuse_evidence(result)
    assert result.tamper_evidence_detected is True
    assert result.independent_signal_count == 2


def test_tamper_detected_on_qr_mismatch_alone() -> None:
    """A QR data mismatch is sufficient on its own."""
    result = ForensicsResult(qr_data_mismatch=True)
    _fuse_evidence(result)
    assert result.tamper_evidence_detected is True


def test_shadow_signals_do_not_affect_independent_signal_count() -> None:
    """Shadow signals go to shadow_signal_count, not independent_signal_count."""
    result = ForensicsResult(
        ela_v2_detected=False,
        text_consistency_anomaly=True,
        photo_boundary_anomaly=True,
    )
    _fuse_evidence(result)
    assert result.independent_signal_count == 0
    assert result.shadow_signal_count == 2
    assert result.tamper_evidence_detected is False


def test_forensic_confidence_high_on_three_signals() -> None:
    result = ForensicsResult(
        ela_v2_detected=True,
        copy_move_detected=True,
        edge_discontinuity_detected=True,
    )
    _fuse_evidence(result)
    assert result.forensic_confidence == "HIGH"


def test_forensic_confidence_low_on_no_signals() -> None:
    result = ForensicsResult()
    _fuse_evidence(result)
    assert result.forensic_confidence == "LOW"


# ─── ForensicsResult API shape stability ─────────────────────────────────────

# These are all the keys the API must always expose.
_REQUIRED_RESULT_FIELDS = {
    "ela_tamper_detected", "ela_score", "ela_heatmap_b64",
    "edge_discontinuity_detected", "edge_score",
    "metadata_anomaly", "metadata_flags",
    "ela_legacy_score", "ela_v2_score", "ela_v2_detected",
    "copy_move_detected", "copy_move_score",
    "qr_detected", "qr_payload_sha256", "qr_data_mismatch",
    "region_anomalies",
    "independent_signal_count", "tamper_evidence_detected", "forensic_confidence",
    # Shadow fields
    "text_consistency_score", "text_consistency_anomaly",
    "photo_boundary_score", "photo_boundary_anomaly",
    "field_vision_mismatch", "field_vision_discrepancies",
    "shadow_signal_count",
    "warnings",
}

def test_forensics_result_dataclass_has_all_required_fields() -> None:
    """Ensure no expected field was accidentally renamed or removed."""
    existing_fields = {f.name for f in dataclass_fields(ForensicsResult)}
    missing = _REQUIRED_RESULT_FIELDS - existing_fields
    assert not missing, f"ForensicsResult is missing fields: {missing}"


def test_run_forensics_returns_all_api_keys(clean_white_jpeg: bytes) -> None:
    """run_forensics on a clean image must populate every expected field."""
    result = run_forensics(clean_white_jpeg, "image/jpeg", "PASSPORT", "A1234567")
    result_dict = result.__dict__
    missing = _REQUIRED_RESULT_FIELDS - set(result_dict.keys())
    assert not missing, f"run_forensics result missing keys: {missing}"


def test_run_forensics_does_not_raise_on_minimal_input(clean_white_jpeg: bytes) -> None:
    """Must complete without raising for any supported input."""
    result = run_forensics(clean_white_jpeg)
    assert isinstance(result, ForensicsResult)


def test_run_forensics_handles_unknown_doc_type(clean_white_jpeg: bytes) -> None:
    result = run_forensics(clean_white_jpeg, "image/jpeg", "UNKNOWN", "")
    assert isinstance(result.region_anomalies, list)


def test_run_forensics_shadow_fields_are_populated(clean_white_jpeg: bytes) -> None:
    """Shadow fields must always be present — even if False/0.0."""
    result = run_forensics(clean_white_jpeg, "image/jpeg", "AADHAAR", "123456789012")
    assert hasattr(result, "text_consistency_score")
    assert hasattr(result, "photo_boundary_score")
    assert hasattr(result, "shadow_signal_count")
    assert isinstance(result.shadow_signal_count, int)
