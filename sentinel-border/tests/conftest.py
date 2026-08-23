"""
conftest.py — shared pytest fixtures for SentinelBorder.

All test images are synthetically generated with PIL.
No real personal identity data is committed.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

# Make sure backend/ is importable regardless of CWD.
_backend = Path(__file__).parent.parent / "backend"
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _pil_to_bytes(img: Image.Image, fmt: str = "JPEG", quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return buf.getvalue()


def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple, text: str, fill: int = 0) -> None:
    """Draw text using the default PIL bitmap font (no external font file required)."""
    draw.text(xy, text, fill=fill)


# ─── Synthetic document factories ────────────────────────────────────────────

def make_clean_white_jpeg(width: int = 400, height: int = 300) -> bytes:
    """A plain white JPEG — baseline for forensic tests (should show no anomalies)."""
    img = Image.fromarray(np.full((height, width, 3), 240, dtype=np.uint8))
    return _pil_to_bytes(img)


def make_synthetic_passport_jpeg() -> bytes:
    """
    A passport-shaped image with a rendered MRZ zone.

    MRZ (TD3 format, two lines of 44 chars):
      Line 1: P<INDSHAH<<RAHUL<KUMAR<<<<<<<<<<<<<<<<<<<<
      Line 2: A1234567<8IND8501011M3012319<<<<<<<<<<<<<<4

    Checksums computed from ICAO 9303:
      Doc number  A1234567< → check digit 8
      DOB         850101    → check digit 1
      Expiry      301231    → check digit 9
      Composite   A1234567<8IND8501011M301231 → check digit 4

    No real person's data — all fields are synthetic.
    """
    width, height = 850, 600
    img = Image.new("RGB", (width, height), color=(245, 240, 230))
    draw = ImageDraw.Draw(img)

    # Document border
    draw.rectangle([10, 10, width - 10, height - 10], outline=(100, 80, 60), width=3)

    # Country header
    _draw_text(draw, (30, 20), "REPUBLIC OF INDIA — PASSPORT", fill=50)

    # Photo placeholder
    draw.rectangle([30, 60, 200, 250], outline=(80, 80, 80), width=2)
    _draw_text(draw, (75, 145), "PHOTO", fill=130)

    # VIZ fields
    _draw_text(draw, (220, 70),  "Surname:      SHAH",      fill=20)
    _draw_text(draw, (220, 90),  "Given Names:  RAHUL KUMAR", fill=20)
    _draw_text(draw, (220, 110), "Nationality:  INDIAN",     fill=20)
    _draw_text(draw, (220, 130), "Date of Birth: 01 JAN 1985", fill=20)
    _draw_text(draw, (220, 150), "Sex:          M",           fill=20)
    _draw_text(draw, (220, 170), "Place of Issue: NEW DELHI", fill=20)
    _draw_text(draw, (220, 190), "Date of Issue: 01 JAN 2020", fill=20)
    _draw_text(draw, (220, 210), "Date of Expiry: 31 DEC 2030", fill=20)
    _draw_text(draw, (220, 230), "Passport No:  A1234567",    fill=20)

    # MRZ zone (dark background strip)
    draw.rectangle([20, height - 120, width - 20, height - 20], fill=(30, 30, 30))
    mrz1 = "P<INDSHAH<<RAHUL<KUMAR<<<<<<<<<<<<<<<<<<<<<<"
    mrz2 = "A12345678IND8501011M3012319<<<<<<<<<<<<<<<<4"
    _draw_text(draw, (25, height - 115), mrz1, fill=240)
    _draw_text(draw, (25, height - 85),  mrz2, fill=240)

    return _pil_to_bytes(img)


def make_synthetic_aadhaar_jpeg() -> bytes:
    """
    An Aadhaar-shaped image with synthetic fields.
    UID: 1234 5678 9012  (fake — not a real Aadhaar number)
    """
    width, height = 856, 540
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.rectangle([5, 5, width - 5, height - 5], outline=(0, 102, 204), width=4)
    _draw_text(draw, (30, 15), "भारत सरकार / Government of India", fill=0)
    _draw_text(draw, (30, 35), "Unique Identification Authority of India", fill=0)

    # Photo placeholder
    draw.rectangle([30, 70, 190, 260], outline=(80, 80, 80), width=2)
    _draw_text(draw, (80, 155), "PHOTO", fill=130)

    # Fields
    _draw_text(draw, (220, 80),  "Name:  RAHUL KUMAR SHAH",   fill=10)
    _draw_text(draw, (220, 110), "DOB:   01/01/1985",         fill=10)
    _draw_text(draw, (220, 140), "Gender: MALE",              fill=10)
    _draw_text(draw, (220, 170), "Address: 12 MG ROAD, NEW DELHI 110001", fill=10)

    # UID
    _draw_text(draw, (300, 440), "1234  5678  9012",          fill=0)

    return _pil_to_bytes(img)


def make_jpeg_with_editor_exif() -> bytes:
    """JPEG with an EXIF 'Software' tag containing 'Adobe Photoshop'."""
    from PIL import ExifTags
    img = Image.new("RGB", (200, 150), color=(200, 200, 200))
    # Build a minimal EXIF blob with the Software tag (tag 305 = 0x0131)
    exif = img.getexif()
    exif[0x0131] = "Adobe Photoshop CS6"
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    return buf.getvalue()


# ─── pytest fixtures ─────────────────────────────────────────────────────────

@pytest.fixture()
def clean_white_jpeg() -> bytes:
    return make_clean_white_jpeg()


@pytest.fixture()
def synthetic_passport_bytes() -> bytes:
    return make_synthetic_passport_jpeg()


@pytest.fixture()
def synthetic_aadhaar_bytes() -> bytes:
    return make_synthetic_aadhaar_jpeg()


@pytest.fixture()
def editor_exif_jpeg() -> bytes:
    return make_jpeg_with_editor_exif()
