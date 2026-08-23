"""
test_ocr_engine.py — Regression tests for Module 1: OCR & MRZ Engine.

Tests cover:
  - Internal ICAO check-digit helper
  - TD3 (passport) MRZ field parsing
  - TD1 (ID card) MRZ field parsing
  - MRZ normalisation and OCR look-alike correction
  - Preprocessing does not crash on clean images
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_backend = Path(__file__).parent.parent / "backend"
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from modules.ocr_engine import (
    _icao_check_digit,
    _normalise_mrz_line,
    _normalise_td3_line2,
    _parse_td3,
    _parse_td1,
    _td3_quality_score,
    _preprocess,
)
from utils.helpers import bytes_to_numpy


# ─── _icao_check_digit ────────────────────────────────────────────────────────

@pytest.mark.parametrize("field,expected", [
    ("520727", 3),
    ("740812", 2),
    ("L898902C<", 3),
    ("<<<<<<<<<", 0),
])
def test_icao_check_digit_matches_validator(field: str, expected: int) -> None:
    """The standalone helper in ocr_engine must match the validator's implementation."""
    assert _icao_check_digit(field) == expected


# ─── MRZ normalisation ────────────────────────────────────────────────────────

def test_normalise_mrz_line_strips_invalid_chars() -> None:
    raw = "P<IND ABC-123 ??!!"
    normalised = _normalise_mrz_line(raw)
    assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<" for c in normalised)


def test_normalise_mrz_line_td3_truncates_to_44() -> None:
    long_line = "P<IND" + "A" * 50
    result = _normalise_mrz_line(long_line)
    assert len(result) == 44


def test_normalise_td3_line2_corrects_check_digit_positions() -> None:
    """O → 0 and I → 1 at check-digit positions (9, 19, 27, 43)."""
    # Build a 44-char line with 'O' at all check-digit positions.
    line = list("A" * 44)
    for pos in (9, 19, 27, 43):
        line[pos] = "O"
    corrected = _normalise_td3_line2("".join(line))
    for pos in (9, 19, 27, 43):
        assert corrected[pos] == "0", f"Position {pos} should be '0'"


def test_normalise_td3_line2_no_change_on_short_line() -> None:
    short = "A" * 30
    assert _normalise_td3_line2(short) == short


# ─── _parse_td3 ───────────────────────────────────────────────────────────────

# TD3 with correctly computed ICAO check digits:
#   Doc number field (9 chars): A1234567< -> check digit 6
#   DOB: 850101 -> check digit 9
#   Expiry: 301231 -> check digit 6
#   Composite -> check digit 2
#   Line1 must be exactly 44 chars (pad name field with < fillers)
_TD3_NAME = "SHAH<<RAHUL<KUMAR"
_TD3_LINE1 = ("P<IND" + _TD3_NAME).ljust(44, "<")
_TD3_LINE2 = "A1234567<6IND8501019M3012316<<<<<<<<<<<<<<<2"

def test_td3_doc_type() -> None:
    parsed = _parse_td3(_TD3_LINE1, _TD3_LINE2)
    assert parsed["doc_type"] == "P"


def test_td3_issuing_country() -> None:
    parsed = _parse_td3(_TD3_LINE1, _TD3_LINE2)
    assert parsed["issuing_country"] == "IND"


def test_td3_surname() -> None:
    parsed = _parse_td3(_TD3_LINE1, _TD3_LINE2)
    assert parsed["surname"] == "SHAH"


def test_td3_given_names() -> None:
    parsed = _parse_td3(_TD3_LINE1, _TD3_LINE2)
    assert "RAHUL" in parsed["given_names"]
    assert "KUMAR" in parsed["given_names"]


def test_td3_doc_number() -> None:
    parsed = _parse_td3(_TD3_LINE1, _TD3_LINE2)
    # TD3 positions 0-8 of line2 = 9-char doc number field; trailing < stripped.
    assert parsed["doc_number"] == "A1234567"


def test_td3_doc_number_check() -> None:
    parsed = _parse_td3(_TD3_LINE1, _TD3_LINE2)
    assert parsed["doc_number_check"] == "6"


def test_td3_nationality() -> None:
    parsed = _parse_td3(_TD3_LINE1, _TD3_LINE2)
    assert parsed["nationality"] == "IND"


def test_td3_dob() -> None:
    parsed = _parse_td3(_TD3_LINE1, _TD3_LINE2)
    assert parsed["dob"] == "850101"


def test_td3_sex() -> None:
    parsed = _parse_td3(_TD3_LINE1, _TD3_LINE2)
    assert parsed["sex"] == "M"


def test_td3_expiry() -> None:
    parsed = _parse_td3(_TD3_LINE1, _TD3_LINE2)
    assert parsed["expiry"] == "301231"


def test_td3_composite_check() -> None:
    parsed = _parse_td3(_TD3_LINE1, _TD3_LINE2)
    assert parsed["composite_check"] == "2"


# ─── _parse_td1 ───────────────────────────────────────────────────────────────

# TD1 example from ICAO 9303 Annex D (redacted / synthetic).
_TD1_LINE1 = "I<UTOD231458907<<<"
_TD1_LINE2 = "7408122F1204159UTO<<<<<<<<<<<6"
_TD1_LINE3 = "ERIKSSON<<ANNA<MARIA<<<<<<<<<<"

def test_td1_doc_type() -> None:
    parsed = _parse_td1(
        _TD1_LINE1.ljust(30, "<"),
        _TD1_LINE2.ljust(30, "<"),
        _TD1_LINE3.ljust(30, "<"),
    )
    assert parsed["doc_type"] == "I"


def test_td1_surname() -> None:
    parsed = _parse_td1(
        _TD1_LINE1.ljust(30, "<"),
        _TD1_LINE2.ljust(30, "<"),
        _TD1_LINE3.ljust(30, "<"),
    )
    assert parsed["surname"] == "ERIKSSON"


def test_td1_given_names() -> None:
    parsed = _parse_td1(
        _TD1_LINE1.ljust(30, "<"),
        _TD1_LINE2.ljust(30, "<"),
        _TD1_LINE3.ljust(30, "<"),
    )
    assert "ANNA" in parsed["given_names"]


def test_td1_dob() -> None:
    parsed = _parse_td1(
        _TD1_LINE1.ljust(30, "<"),
        _TD1_LINE2.ljust(30, "<"),
        _TD1_LINE3.ljust(30, "<"),
    )
    assert parsed["dob"] == "740812"


def test_td1_sex() -> None:
    parsed = _parse_td1(
        _TD1_LINE1.ljust(30, "<"),
        _TD1_LINE2.ljust(30, "<"),
        _TD1_LINE3.ljust(30, "<"),
    )
    assert parsed["sex"] == "F"


# ─── _td3_quality_score ───────────────────────────────────────────────────────

def test_quality_score_perfect_td3() -> None:
    score = _td3_quality_score(_TD3_LINE1, _TD3_LINE2)
    # Passes format check (10) + 3 independent checksums (3 x 10) = 40
    assert score == 40


def test_quality_score_zero_on_random_lines() -> None:
    assert _td3_quality_score("HELLO WORLD", "RANDOM DATA") == 0


def test_quality_score_zero_on_short_lines() -> None:
    assert _td3_quality_score("P<IND", "A12345") == 0


# ─── _preprocess ─────────────────────────────────────────────────────────────

def test_preprocess_does_not_crash_on_clean_image(clean_white_jpeg: bytes) -> None:
    """Preprocessing must not raise on a plain JPEG."""
    img_bgr = bytes_to_numpy(clean_white_jpeg)
    result_bgr, gray = _preprocess(img_bgr)
    assert result_bgr.ndim == 3
    assert gray.ndim == 2


def test_preprocess_resizes_wide_image() -> None:
    """Images are normalised to the canonical 1600 px width (upscaled or downscaled)."""
    wide = np.full((300, 2000, 3), 200, dtype=np.uint8)
    import cv2
    import io
    ok, buf = cv2.imencode(".jpg", wide)
    img_bgr = bytes_to_numpy(buf.tobytes())
    result_bgr, _ = _preprocess(img_bgr)
    assert result_bgr.shape[1] == 1600
