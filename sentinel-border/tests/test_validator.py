"""
test_validator.py — Regression tests for Module 2: Document Validation Engine.

Tests cover:
  - ICAO Doc 9303 Modulo-10 checksum computation
  - All four MRZ checksums on a synthetic passport
  - Tampered-field checksum failure
  - Expiry date parsing and expired-document detection
  - VIZ ↔ MRZ parity (match and mismatch cases)
  - OCR/Vision field cross-check (check_field_vision_consistency)
"""
from __future__ import annotations

import pytest
from modules.validator import (
    icao_checksum,
    verify_checksum,
    check_viz_mrz_parity,
    check_field_vision_consistency,
    run_validation,
    is_expired,
)
from modules.ocr_engine import OCRResult


# ─── ICAO Checksum unit tests ─────────────────────────────────────────────────

# Known (field, expected_check_digit) vectors from ICAO Doc 9303 Part 3, §4.9.
_ICAO_VECTORS = [
    ("520727",  3),   # DOB from ICAO example
    ("740812",  2),
    ("ZE184226", 0),   # computed: 0
    ("AB2134<<<", 5),  # computed: 5
    ("L898902C<", 3),  # Passport number example from ICAO 9303
    ("690806",  1),
    ("940623",  6),
    ("AA00000", 0),    # All-zero numeric value
    ("ABCDEFGH", 6),   # computed: 6
    ("<<<<<<<<<", 0),  # Fillers only
]

@pytest.mark.parametrize("field,expected", _ICAO_VECTORS)
def test_icao_checksum_known_vectors(field: str, expected: int) -> None:
    assert icao_checksum(field) == expected


def test_verify_checksum_pass() -> None:
    assert verify_checksum("L898902C<", "3") is True


def test_verify_checksum_fail_wrong_digit() -> None:
    assert verify_checksum("L898902C<", "9") is False


def test_verify_checksum_fail_non_digit() -> None:
    """An invalid check-digit character must return False, not raise."""
    assert verify_checksum("L898902C<", "X") is False


def test_verify_checksum_fail_empty_digit() -> None:
    assert verify_checksum("L898902C<", "") is False


# ─── Synthetic passport: all four checksums pass ──────────────────────────────

def _make_passport_ocr() -> OCRResult:
    """
    Synthetic TD3 MRZ with all check digits computed from ICAO 9303 Modulo-10:
      Doc number field: A1234567< (9 chars) -> check digit 6
      DOB:  850101 -> check digit 9
      Expiry: 301231 -> check digit 6
      Composite: A1234567<6 + 8501019 + 3012316<<<<<<<<<<<<< -> check digit 2

    Line2 (44 chars): A1234567<6IND8501019M3012316<<<<<<<<<<<<<<<2
    No real person's data.
    """
    ocr = OCRResult(
        mrz_detected=True,
        mrz_line1="P<INDSHAH<<RAHUL<KUMAR<<<<<<<<<<<<<<<<<<<<",
        mrz_line2="A1234567<6IND8501019M3012316<<<<<<<<<<<<<<<2",
        doc_type="P",
        issuing_country="IND",
        surname="SHAH",
        given_names="RAHUL KUMAR",
        doc_number="A1234567",
        doc_number_check="6",
        nationality="IND",
        dob="850101",
        dob_check="9",
        sex="M",
        expiry="301231",
        expiry_check="6",
        optional_data="",
        composite_check="2",
        # Include both the country-code 'IND' and the full-form 'INDIAN' that appear
        # in real Indian passport VIZ text.  The parity check matches tokens from the
        # MRZ nationality field ('IND') against the VIZ token set.
        viz_text="SHAH RAHUL KUMAR IND INDIAN A1234567",
    )
    return ocr


def test_td3_doc_number_checksum_passes() -> None:
    ocr = _make_passport_ocr()
    vr = run_validation(ocr)
    assert vr.checksum_doc_number is True, "Doc number checksum should pass"


def test_td3_dob_checksum_passes() -> None:
    ocr = _make_passport_ocr()
    vr = run_validation(ocr)
    assert vr.checksum_dob is True, "DOB checksum should pass"


def test_td3_expiry_checksum_passes() -> None:
    ocr = _make_passport_ocr()
    vr = run_validation(ocr)
    assert vr.checksum_expiry is True, "Expiry checksum should pass"


def test_td3_no_checksum_failures() -> None:
    ocr = _make_passport_ocr()
    vr = run_validation(ocr)
    assert vr.any_checksum_failed is False, "No checksum should fail on a valid MRZ"


# ─── Tampered-field checksum failure ─────────────────────────────────────────

def test_tampered_expiry_breaks_checksum() -> None:
    """Changing the expiry date without updating its check digit must fail."""
    ocr = _make_passport_ocr()
    ocr.expiry = "350101"   # changed year from 30 to 35
    # expiry_check is still "9" which was for 301231 — should now fail
    vr = run_validation(ocr)
    assert vr.checksum_expiry is False
    assert vr.any_checksum_failed is True


def test_tampered_doc_number_breaks_checksum() -> None:
    ocr = _make_passport_ocr()
    ocr.doc_number = "A9999999"   # changed number
    vr = run_validation(ocr)
    assert vr.checksum_doc_number is False
    assert vr.any_checksum_failed is True


# ─── Expiry validation ────────────────────────────────────────────────────────

def test_expired_document_detected() -> None:
    assert is_expired("200101") is True   # Jan 2020 — in the past


def test_valid_document_not_expired() -> None:
    # Year 30 -> 2030 (future). Year 99 -> 1999 (past) per the validator's yy<=30 cutoff.
    assert is_expired("300101") is False   # Jan 2030 — future
    assert is_expired("991231") is True    # Dec 1999 — past (yy=99 > 30 -> 1900+yy)


def test_unparseable_expiry_treated_as_expired() -> None:
    assert is_expired("") is True
    assert is_expired("BADVAL") is True


def test_run_validation_flags_expired_doc() -> None:
    ocr = _make_passport_ocr()
    ocr.expiry = "200101"    # past
    ocr.expiry_check = str(icao_checksum("200101"))
    vr = run_validation(ocr)
    assert vr.document_expired is True
    assert any("EXPIRED" in w for w in vr.warnings)


# ─── VIZ ↔ MRZ parity ────────────────────────────────────────────────────────

def test_viz_mrz_parity_passes_on_match() -> None:
    ocr = _make_passport_ocr()
    mismatch, _ = check_viz_mrz_parity(ocr)
    assert mismatch is False


def test_viz_mrz_parity_mismatch_detected() -> None:
    ocr = _make_passport_ocr()
    # VIZ has a completely different surname — should trigger a mismatch flag.
    ocr.viz_text = "DIFFERENT NAME 1234567"
    mismatch, msgs = check_viz_mrz_parity(ocr)
    assert mismatch is True
    assert len(msgs) > 0
    assert any("Surname" in m for m in msgs)


def test_viz_mrz_parity_skipped_when_viz_empty() -> None:
    ocr = _make_passport_ocr()
    ocr.viz_text = ""
    mismatch, msgs = check_viz_mrz_parity(ocr)
    assert mismatch is False
    assert msgs == []


# ─── Field-vision cross-check ─────────────────────────────────────────────────

def test_field_vision_consistency_no_mismatch() -> None:
    ocr = _make_passport_ocr()
    vision = {"doc_number": "A1234567", "surname": "SHAH", "dob": "850101"}
    mismatch, msgs = check_field_vision_consistency(ocr, vision)
    assert mismatch is False


def test_field_vision_consistency_doc_number_mismatch() -> None:
    ocr = _make_passport_ocr()
    vision = {"doc_number": "Z9999999", "surname": "SHAH", "dob": "850101"}
    mismatch, msgs = check_field_vision_consistency(ocr, vision)
    assert mismatch is True
    assert any("Document Number" in m for m in msgs)


def test_field_vision_consistency_empty_vision_is_no_op() -> None:
    ocr = _make_passport_ocr()
    mismatch, msgs = check_field_vision_consistency(ocr, {})
    assert mismatch is False
    assert msgs == []


def test_field_vision_consistency_partial_vision_no_false_flag() -> None:
    """If vision only returns doc_number and it matches, no mismatch."""
    ocr = _make_passport_ocr()
    vision = {"doc_number": "A1234567"}
    mismatch, msgs = check_field_vision_consistency(ocr, vision)
    assert mismatch is False


# ─── MRZ not detected — validation skipped ───────────────────────────────────

def test_run_validation_skipped_when_no_mrz() -> None:
    ocr = OCRResult(mrz_detected=False)
    vr = run_validation(ocr)
    # No checksum checks run; any_checksum_failed stays False.
    assert vr.any_checksum_failed is False
    assert any("skipped" in w.lower() for w in vr.warnings)
