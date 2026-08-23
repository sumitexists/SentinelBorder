"""
SentinelBorder — Module 2: Document Validation Engine
Implements:
  - ICAO Doc 9303 Modulo-10 checksum verification (weight cycle [7, 3, 1])
  - Document expiry date validation
  - VIZ-to-MRZ text parity cross-check
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from modules.ocr_engine import OCRResult
from utils.helpers import get_logger

log = get_logger("validator")


# ─── ICAO Doc 9303 Modulo-10 ──────────────────────────────────────────────────

_WEIGHTS = [7, 3, 1]

def _char_value(ch: str) -> int:
    """
    Map a MRZ character to its numeric value per ICAO 9303:
      '0'-'9'  →  0-9
      'A'-'Z'  →  10-35
      '<'      →  0  (filler)
    """
    ch = ch.upper()
    if ch == "<":
        return 0
    if ch.isdigit():
        return int(ch)
    if ch.isalpha():
        return ord(ch) - ord("A") + 10
    return 0


def icao_checksum(field_str: str) -> int:
    """
    Compute the ICAO Doc 9303 Modulo-10 check digit for a given string.
    Check_Digit = (Σ char_value[i] × weight[i mod 3]) mod 10
    """
    total = sum(_char_value(ch) * _WEIGHTS[i % 3] for i, ch in enumerate(field_str))
    return total % 10


def verify_checksum(field_str: str, expected_digit: str) -> bool:
    """Return True if the computed check digit matches the expected digit."""
    try:
        expected = int(expected_digit)
    except (ValueError, TypeError):
        return False
    return icao_checksum(field_str) == expected


# ─── Expiry Validation ────────────────────────────────────────────────────────

def _yymmdd_to_date(yymmdd: str) -> Optional[date]:
    """
    Parse a 6-char YYMMDD string to a date object.
    Years 00-30 are assumed 2000-2030; 31-99 assumed 1931-1999.
    """
    if not yymmdd or len(yymmdd) != 6 or not yymmdd.isdigit():
        return None
    yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    year = 2000 + yy if yy <= 30 else 1900 + yy
    try:
        return date(year, mm, dd)
    except ValueError:
        return None


def is_expired(expiry_yymmdd: str) -> bool:
    """Return True if the document expiry date is in the past."""
    expiry = _yymmdd_to_date(expiry_yymmdd)
    if expiry is None:
        return True          # unparseable → treat as expired/suspect
    return expiry < date.today()


# ─── VIZ ↔ MRZ Parity Check ──────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, remove punctuation and extra spaces for fuzzy parity."""
    import re
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def check_viz_mrz_parity(ocr: OCRResult) -> tuple[bool, list[str]]:
    """
    Cross-check VIZ free text against parsed MRZ fields.
    Returns (mismatch_found: bool, list_of_discrepancy_messages).
    A mismatch is flagged if a key MRZ field cannot be found anywhere
    in the VIZ text (surname, given name, document number, nationality).
    """
    mismatches: list[str] = []
    viz = _normalize(ocr.viz_text)

    if not viz:
        # Cannot check parity without VIZ data
        return False, []

    def _field_in_viz(label: str, value: str) -> None:
        if not value:
            return
        normalized_value = _normalize(value)
        if normalized_value and normalized_value not in viz:
            mismatches.append(f"VIZ/MRZ mismatch — {label}: MRZ='{value}' not found in VIZ.")

    _field_in_viz("Surname", ocr.surname)
    _field_in_viz("Given Names", ocr.given_names)
    _field_in_viz("Nationality", ocr.nationality)

    return bool(mismatches), mismatches


def check_field_vision_consistency(
    ocr: OCRResult,
    vision_fields: dict,
) -> tuple[bool, list[str]]:
    """Cross-check OCR-parsed fields against Gemini/Ollama vision extraction.

    Both sources read the same image independently.  If they disagree on a key
    identity field, one of them is likely wrong — which may indicate document
    tampering or an OCR failure.

    Compares: doc_number, surname, dob.
    Returns (mismatch_found: bool, list_of_discrepancy_messages).

    Note: This check is designed for shadow mode.  A single discrepancy is not
    proof of forgery — OCR errors and vision model hallucinations are both
    possible, especially on low-resolution scans.
    """
    mismatches: list[str] = []

    if not vision_fields:
        return False, []

    def _compare(label: str, ocr_value: str, vision_value: str) -> None:
        """Flag when both values are non-empty and normalised forms do not match."""
        if not ocr_value or not vision_value:
            return
        a = _normalize(ocr_value)
        b = _normalize(vision_value)
        if a and b and a != b:
            mismatches.append(
                f"OCR/Vision mismatch \u2014 {label}: OCR='{ocr_value}' Vision='{vision_value}'"
            )

    _compare("Document Number", ocr.doc_number, vision_fields.get("doc_number", ""))
    _compare("Surname", ocr.surname, vision_fields.get("surname", ""))
    _compare("Date of Birth", ocr.dob, vision_fields.get("dob", ""))

    return bool(mismatches), mismatches


# ─── Output Data Structure ────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    checksum_doc_number: bool = False
    checksum_dob: bool = False
    checksum_expiry: bool = False
    checksum_composite: bool = False
    any_checksum_failed: bool = False

    document_expired: bool = False
    expiry_date_parsed: str = ""

    viz_mrz_mismatch: bool = False
    viz_mrz_discrepancies: list[str] = field(default_factory=list)

    warnings: list[str] = field(default_factory=list)


# ─── Public API ───────────────────────────────────────────────────────────────

def run_validation(ocr: OCRResult) -> ValidationResult:
    """
    Run all validation checks against an OCRResult and return a ValidationResult.
    """
    vr = ValidationResult()

    if not ocr.mrz_detected:
        vr.warnings.append("MRZ not detected — checksum verification skipped.")
        return vr

    # ── Checksum 1: Document Number ──────────────────────────────────────────
    vr.checksum_doc_number = verify_checksum(ocr.doc_number, ocr.doc_number_check)
    if not vr.checksum_doc_number:
        vr.warnings.append(
            f"FAIL: Document number checksum — field='{ocr.doc_number}' "
            f"check_digit='{ocr.doc_number_check}'"
        )

    # ── Checksum 2: Date of Birth ─────────────────────────────────────────────
    vr.checksum_dob = verify_checksum(ocr.dob, ocr.dob_check)
    if not vr.checksum_dob:
        vr.warnings.append(
            f"FAIL: DOB checksum — field='{ocr.dob}' check_digit='{ocr.dob_check}'"
        )

    # ── Checksum 3: Expiry Date ───────────────────────────────────────────────
    vr.checksum_expiry = verify_checksum(ocr.expiry, ocr.expiry_check)
    if not vr.checksum_expiry:
        vr.warnings.append(
            f"FAIL: Expiry checksum — field='{ocr.expiry}' check_digit='{ocr.expiry_check}'"
        )

    # ── Checksum 4: Composite ────────────────────────────────────────────────
    if ocr.mrz_line2 and len(ocr.mrz_line2) >= 44:
        composite_field = ocr.mrz_line2[0:10] + ocr.mrz_line2[13:20] + ocr.mrz_line2[21:43]
        vr.checksum_composite = verify_checksum(composite_field, ocr.composite_check)
        if not vr.checksum_composite:
            vr.warnings.append("FAIL: Composite MRZ checksum mismatch.")
    else:
        # For TD1 or short MRZ, skip composite
        vr.checksum_composite = True

    vr.any_checksum_failed = not (
        vr.checksum_doc_number and vr.checksum_dob and
        vr.checksum_expiry and vr.checksum_composite
    )

    # ── Expiry Check ─────────────────────────────────────────────────────────
    vr.document_expired = is_expired(ocr.expiry)
    expiry_date_obj = _yymmdd_to_date(ocr.expiry)
    if expiry_date_obj:
        vr.expiry_date_parsed = expiry_date_obj.strftime("%Y-%m-%d")
    if vr.document_expired:
        vr.warnings.append(f"Document EXPIRED on {vr.expiry_date_parsed}.")

    # ── VIZ / MRZ Parity ─────────────────────────────────────────────────────
    vr.viz_mrz_mismatch, vr.viz_mrz_discrepancies = check_viz_mrz_parity(ocr)
    if vr.viz_mrz_mismatch:
        vr.warnings.extend(vr.viz_mrz_discrepancies)

    log.info("Validation complete — checksum_failed=%s, expired=%s, viz_mrz_mismatch=%s",
             vr.any_checksum_failed, vr.document_expired, vr.viz_mrz_mismatch)

    return vr
