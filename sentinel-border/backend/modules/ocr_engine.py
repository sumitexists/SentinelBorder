"""
SentinelBorder — Module 1: OCR & MRZ Extraction Engine
Multi-engine approach:
  1. PassportEye  → dedicated MRZ detector (2/3-line zones)
  2. EasyOCR      → state-of-the-art VIZ field extraction (pre-built wheels, no MSVC)
  3. PyTesseract  → fallback for clean cropped regions

Pre-processing pipeline uses OpenCV Sobel-based de-skewing and CLAHE normalisation
to maximise accuracy on blurry, noisy border scans.
"""

from __future__ import annotations

import io
import re
import sys
import tempfile
import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from config import STRUCTURED_OCR_PROVIDER
from utils.helpers import bytes_to_numpy, bytes_to_pil, get_logger

log = get_logger("ocr_engine")


# ─── Output Data Structure ────────────────────────────────────────────────────

@dataclass
class OCRResult:
    # MRZ raw lines
    mrz_line1: str = ""
    mrz_line2: str = ""
    mrz_line3: str = ""          # only for TD1 (ID cards)
    mrz_raw: str = ""

    # Parsed MRZ fields
    doc_type: str = ""
    issuing_country: str = ""
    surname: str = ""
    given_names: str = ""
    doc_number: str = ""
    doc_number_check: str = ""
    nationality: str = ""
    dob: str = ""                # YYMMDD
    dob_check: str = ""
    sex: str = ""
    expiry: str = ""             # YYMMDD
    expiry_check: str = ""
    optional_data: str = ""
    composite_check: str = ""
    address: str = ""

    # VIZ (visual zone) free-text
    viz_text: str = ""
    text_bboxes: list = field(default_factory=list)

    # ─── LLM Shadow Mode ──────────────────────────────────────────────────────────
    vision_extracted_data: dict = field(default_factory=dict)
    vlm_tamper_detected: bool = False
    vlm_tamper_reason: str = ""

    # Confidence flag
    mrz_detected: bool = False
    ocr_engine_used: str = "none"
    errors: list[str] = field(default_factory=list)

    # Gatekeeping flags (populated for non-MRZ docs via LLM)
    is_government_id: bool = True
    id_type: str = ""
    id_reasoning: str = ""


# ─── Pre-processing ───────────────────────────────────────────────────────────

def _preprocess(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalise and de-skew a document image:
    1. Resize to canonical width (1600 px) keeping aspect ratio — wider gives
       EasyOCR more pixels per character and consistently raises accuracy.
    2. Convert to grayscale.
    3. Apply CLAHE for local contrast enhancement.
    4. Estimate skew via Hough line angles and deskew.
    Returns (processed_bgr, gray).
    """
    # 1. Resize — upscale narrow images, cap wide ones at 1600 px
    h, w = img_bgr.shape[:2]
    target_w = 1600
    if w != target_w:
        scale = target_w / w
        img_bgr = cv2.resize(
            img_bgr, (target_w, int(h * scale)), interpolation=cv2.INTER_LANCZOS4
        )

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 3. De-skew using Sobel + Hough lines
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel = np.uint8(np.absolute(sobel))
    lines = cv2.HoughLinesP(sobel, 1, np.pi / 180, threshold=80,
                             minLineLength=50, maxLineGap=10)
    if lines is not None and len(lines) > 0:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = (int(v) for v in line.flatten()[:4])
            if x2 - x1 != 0:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 45:
                    angles.append(angle)
        if angles:
            median_angle = float(np.median(angles))
            if abs(median_angle) > 0.5:
                ch, cw = gray.shape[:2]
                M = cv2.getRotationMatrix2D((cw / 2, ch / 2), median_angle, 1.0)
                img_bgr = cv2.warpAffine(img_bgr, M, (cw, ch),
                                          flags=cv2.INTER_CUBIC,
                                          borderMode=cv2.BORDER_REPLICATE)
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    return img_bgr, gray


def _sharpen_for_viz(img_bgr: np.ndarray) -> np.ndarray:
    """
    Prepare a sharpened BGR image specifically for EasyOCR VIZ extraction.

    Indian passport VIZ text is printed in a compact, high-density layout.
    An unsharp-mask pass on top of the CLAHE-normalised image significantly
    improves character-edge contrast and raises EasyOCR confidence scores.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Strong CLAHE to maximise local contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    # Unsharp mask: sharpen = original + (original - blurred) * amount
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)
    # Convert back to BGR so EasyOCR receives a 3-channel image
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


# ─── MRZ Extraction via PassportEye ──────────────────────────────────────────

def _extract_mrz_passporteye(raw_bytes: bytes) -> tuple[str, str, str]:
    """
    Use PassportEye to locate and extract the MRZ from a document image.
    Returns (line1, line2, line3) — line3 empty for TD3 passports.
    """
    try:
        from passporteye import read_mrz

        tmp_path = ""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        try:
            mrz = read_mrz(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        if mrz is None:
            return "", "", ""

        data = mrz.to_dict()
        raw = data.get("raw_text", "")
        lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]

        if len(lines) >= 2:
            return lines[0], lines[1], lines[2] if len(lines) >= 3 else ""
        return "", "", ""

    except Exception as exc:
        log.warning("PassportEye MRZ extraction failed: %s", exc)
        return "", "", ""


def _icao_check_digit(value: str) -> int:
    """Return the ICAO 9303 check digit without importing the validator module."""
    weights = (7, 3, 1)

    def char_value(char: str) -> int:
        if char == "<":
            return 0
        if char.isdigit():
            return int(char)
        return ord(char) - ord("A") + 10

    return sum(char_value(char) * weights[i % 3] for i, char in enumerate(value)) % 10


def _normalise_mrz_line(line: str) -> str:
    """Keep only ICAO MRZ characters and remove harmless trailing OCR noise."""
    line = re.sub(r"[^A-Z0-9<]", "", line.upper())
    # PassportEye occasionally appends a stray glyph after a complete TD3 name
    # line.  It is safe to drop only trailing characters beyond the fixed width.
    if line.startswith("P<") and len(line) > 44:
        return line[:44]
    return line


def _normalise_td3_line2(line: str) -> str:
    """
    Correct OCR look-alikes in TD3 Line 2 numeric zones.

    Applies digit look-alike substitution across ALL positions that must contain
    a digit per the ICAO 9303 TD3 layout:
      [0-9]   document number (+ check digit at 9)
      [13-19] date of birth  (+ check digit at 19)
      [21-27] expiry date    (+ check digit at 27)
      [43]    composite check digit

    Previously this only fixed the four check-digit positions, leaving
    O/I/S/etc. inside the actual field ranges (e.g. expiry '34O101' instead
    of '340101'), which caused wrong years and ICAO checksum failures.
    """
    if len(line) < 44:
        return line

    digit_lookalikes = str.maketrans({
        "O": "0", "Q": "0", "I": "1", "L": "1", "Z": "2",
        "S": "5", "G": "6", "B": "8",
    })
    chars = list(line)
    # Numeric zones: doc number (0-9), DOB (13-19), expiry (21-27), composite (43)
    numeric_ranges = list(range(0, 10)) + list(range(13, 20)) + list(range(21, 28)) + [43]
    for index in numeric_ranges:
        if index < len(chars):
            chars[index] = chars[index].translate(digit_lookalikes)
    return "".join(chars)


def _td3_quality_score(line1: str, line2: str) -> int:
    """Score a likely TD3 read using its fixed layout and three independent checks."""
    if len(line1) < 44 or len(line2) < 28:
        return 0
    if not re.match(r"^P<[A-Z<]{3}[A-Z<]+$", line1):
        return 0
    if not re.match(r"^[A-Z0-9<]{9}\d[A-Z<]{3}\d{6}\d[MF<]\d{6}\d", line2):
        return 0

    score = 10
    for field, check_digit in (
        (line2[0:9], line2[9]),
        (line2[13:19], line2[19]),
        (line2[21:27], line2[27]),
    ):
        if _icao_check_digit(field) == int(check_digit):
            score += 10
    return score


def _encode_mrz_candidate(img_bgr: np.ndarray, crop_to_bottom: bool = False) -> bytes:
    """Create a contrast-enhanced, enlarged JPEG suited to OCR of small MRZ text."""
    if crop_to_bottom:
        height = img_bgr.shape[0]
        img_bgr = img_bgr[int(height * 0.65):, :]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # A global contrast boost is deliberately used here instead of thresholding:
    # thresholding removes the fine anti-aliased strokes found in low-resolution
    # MRZ scans.  This mirrors the enhancement that PassportEye handles best.
    mean = float(np.mean(gray))
    gray = np.clip(mean + 2.0 * (gray.astype(np.float32) - mean), 0, 255).astype(np.uint8)

    # PassportEye is much more reliable when the character height is at least
    # about 45 px.  Cap enlargement to prevent very large uploads from growing.
    height, width = gray.shape
    scale = min(3.0, max(1.0, 2304 / max(width, 1)))
    if scale > 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    ok, encoded = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("Could not encode MRZ fallback candidate.")
    return encoded.tobytes()


def _extract_mrz_fallback(img_bgr: np.ndarray) -> tuple[str, str, str]:
    """Retry MRZ detection on enhanced full-page and lower-page orientations."""
    # The common case is a portrait passport page with the MRZ across its lower
    # edge.  This enhanced full-page pass fixes low-resolution scans without
    # assuming a particular crop.
    images: list[tuple[np.ndarray, bool]] = [
        (img_bgr, False),
        (img_bgr, True),
    ]

    # If the scan was uploaded sideways or upside-down, its MRZ becomes the
    # lower strip in one of these orientations.
    for rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
        images.append((cv2.rotate(img_bgr, rotation), True))

    best = ("", "", "")
    best_score = 0
    for candidate_img, crop_to_bottom in images:
        raw_line1, raw_line2, raw_line3 = _extract_mrz_passporteye(
            _encode_mrz_candidate(candidate_img, crop_to_bottom=crop_to_bottom)
        )
        line1 = _normalise_mrz_line(raw_line1)
        line2 = _normalise_td3_line2(_normalise_mrz_line(raw_line2))
        score = _td3_quality_score(line1, line2)
        if score > best_score:
            best = (line1, line2, _normalise_mrz_line(raw_line3))
            best_score = score
        # All three independent, fixed-field checks passed.  No later candidate
        # can be more trustworthy, so avoid unnecessary OCR work.
        if score == 40:
            break

    if best_score:
        log.info("MRZ fallback accepted a TD3 candidate with quality score %d.", best_score)
    return best


# ─── VIZ Extraction via EasyOCR ──────────────────────────────────────────────

# Module-level reader cache — initialised once, reused on subsequent requests.
_easyocr_reader = None

def _extract_viz_easyocr(img_bgr: np.ndarray) -> tuple[str, list]:
    """
    Run EasyOCR on a sharpened version of the document image to extract VIZ text.

    Improvements over the baseline:
    - Uses _sharpen_for_viz() for better character edge contrast on Indian passports.
    - Raises confidence threshold to 0.45 to reduce noise tokens.
    - Sorts detections top-to-bottom then left-to-right to preserve reading order,
      so label+value pairs (e.g. 'Surname / THAPLIYAL') stay on adjacent lines.
    - Uses a cached Reader instance to avoid reloading models on every call.
    """
    global _easyocr_reader
    try:
        import easyocr
        if _easyocr_reader is None:
            log.info("Initialising EasyOCR reader (first call, may take a moment)...")
            _easyocr_reader = easyocr.Reader(["en"], gpu=True, verbose=False)

        viz_img = _sharpen_for_viz(img_bgr)
        results = _easyocr_reader.readtext(viz_img, detail=1, paragraph=False)

        # Filter low-confidence tokens and sort by reading order (top→bottom, left→right)
        confident = [
            (bbox, text, conf)
            for (bbox, text, conf) in results
            if conf > 0.45 and text.strip()
        ]
        # bbox is a list of 4 [x,y] points; use the top-left y as primary sort key
        confident.sort(key=lambda item: (int(item[0][0][1] / 15), int(item[0][0][0])))

        lines = [text for (_bbox, text, _conf) in confident]
        bboxes = [bbox for (bbox, _text, _conf) in confident]
        return "\n".join(lines), bboxes
    except Exception as exc:
        log.warning("EasyOCR VIZ extraction failed: %s", exc)
        return "", []


# ─── VIZ Fallback via PyTesseract ────────────────────────────────────────────

def _extract_viz_tesseract(gray: np.ndarray) -> str:
    """Tesseract OCR fallback for clean images."""
    try:
        import pytesseract
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(binary, config="--psm 6 --oem 3")
        return text.strip()
    except Exception as exc:
        log.warning("Tesseract VIZ fallback failed: %s", exc)
        return ""


# ─── MRZ Parser ───────────────────────────────────────────────────────────────

_MRZ_FILLER = "<"

def _clean(s: str) -> str:
    return s.replace(_MRZ_FILLER, " ").strip()


def _parse_td3(line1: str, line2: str) -> dict:
    """Parse TD3 (passport) MRZ — 2 lines of 44 chars each."""
    # Pad if needed
    l1 = line1.ljust(44)
    l2 = line2.ljust(44)

    raw_names = l1[5:44]
    name_parts = raw_names.split("<<", 1)
    surname = _clean(name_parts[0]) if name_parts else ""
    given = _clean(name_parts[1].replace("<", " ")) if len(name_parts) > 1 else ""

    return {
        "doc_type": l1[0:2].strip("<"),
        "issuing_country": l1[2:5].strip("<"),
        "surname": surname,
        "given_names": given,
        "doc_number": l2[0:9].strip("<"),
        "doc_number_check": l2[9],
        "nationality": l2[10:13].strip("<"),
        "dob": l2[13:19],
        "dob_check": l2[19],
        "sex": l2[20],
        "expiry": l2[21:27],
        "expiry_check": l2[27],
        "optional_data": l2[28:42].strip("<"),
        "composite_check": l2[43],
    }


def _parse_td1(line1: str, line2: str, line3: str) -> dict:
    """Parse TD1 (ID card) MRZ — 3 lines of 30 chars each."""
    l1 = line1.ljust(30)
    l2 = line2.ljust(30)
    l3 = line3.ljust(30)

    raw_names = l3[0:30]
    name_parts = raw_names.split("<<", 1)
    surname = _clean(name_parts[0]) if name_parts else ""
    given = _clean(name_parts[1].replace("<", " ")) if len(name_parts) > 1 else ""

    return {
        "doc_type": l1[0:2].strip("<"),
        "issuing_country": l1[2:5].strip("<"),
        "doc_number": l1[5:14].strip("<"),
        "doc_number_check": l1[14],
        "nationality": l2[15:18].strip("<"),
        "dob": l2[0:6],
        "dob_check": l2[6],
        "sex": l2[7],
        "expiry": l2[8:14],
        "expiry_check": l2[14],
        "surname": surname,
        "given_names": given,
        "optional_data": "",
        "composite_check": l2[29],
    }


# ─── Public API ───────────────────────────────────────────────────────────────

def run_ocr(raw_bytes: bytes, content_type: str = "image/jpeg") -> OCRResult:
    """
    Full OCR pipeline. Returns an OCRResult with all parsed fields.

    For MRZ documents (passports, TD1 ID cards): PassportEye → ICAO parser.
    For non-MRZ documents (Aadhaar, PAN, Voter ID, DL):
      Step 1b uses Gemini Vision for all non-MRZ document types.
    """
    result = OCRResult()

    # Pre-process image
    try:
        # Keep the untouched decode for MRZ fallback.  The general OCR image may
        # be deskewed, but a bad Hough estimate can be harmful to tiny MRZ text.
        source_img_bgr = bytes_to_numpy(raw_bytes)
        img_bgr, gray = _preprocess(source_img_bgr)
    except Exception as exc:
        result.errors.append(f"Preprocessing failed: {exc}")
        return result

    # ── Step 1: MRZ via PassportEye ──────────────────────────────────────────
    line1, line2, line3 = _extract_mrz_passporteye(raw_bytes)
    if not (line1 and line2):
        log.info("PassportEye did not find an MRZ in the original image; retrying enhanced candidates.")
        line1, line2, line3 = _extract_mrz_fallback(source_img_bgr)

    if line1 and line2:
        result.mrz_line1 = line1
        result.mrz_line2 = line2
        result.mrz_line3 = line3
        result.mrz_raw = "\n".join(filter(None, [line1, line2, line3]))
        result.mrz_detected = True

        try:
            if line3:
                parsed = _parse_td1(line1, line2, line3)
            else:
                parsed = _parse_td3(line1, line2)

            result.doc_type = parsed.get("doc_type", "")
            result.issuing_country = parsed.get("issuing_country", "")
            result.surname = parsed.get("surname", "")
            result.given_names = parsed.get("given_names", "")
            result.doc_number = parsed.get("doc_number", "")
            result.doc_number_check = parsed.get("doc_number_check", "")
            result.nationality = parsed.get("nationality", "")
            result.dob = parsed.get("dob", "")
            result.dob_check = parsed.get("dob_check", "")
            result.sex = parsed.get("sex", "")
            result.expiry = parsed.get("expiry", "")
            result.expiry_check = parsed.get("expiry_check", "")
            result.optional_data = parsed.get("optional_data", "")
            result.composite_check = parsed.get("composite_check", "")
        except Exception as exc:
            result.errors.append(f"MRZ parsing error: {exc}")
    else:
        result.errors.append("MRZ not detected by PassportEye.")

    # ── Step 2: VIZ via EasyOCR (always — retained for display and parity checks)
    result.viz_text, result.text_bboxes = _extract_viz_easyocr(img_bgr)
    if not result.viz_text:
        log.info("EasyOCR failed, falling back to Tesseract for VIZ.")
        result.viz_text = _extract_viz_tesseract(gray)
        result.ocr_engine_used = "tesseract-fallback"
    else:
        result.ocr_engine_used = "easyocr"

    # ── Step 2b: Cross-Correction of MRZ Document Number via VIZ ─────────────
    # MRZ text is small and often suffers from look-alike errors (e.g., 'S' -> '8').
    # Some look-alikes cause ICAO checksum collisions and pass unnoticed. VIZ text
    # is larger and more reliable for these characters.
    if result.mrz_detected and result.doc_number and result.viz_text:
        import difflib
        mrz_dn = result.doc_number
        best_match = None
        best_ratio = 0.85  # requires at least ~7/8 chars to match exactly
        
        # Scan VIZ tokens for an alphanumeric string of the exact same length
        for v_tok in result.viz_text.upper().split():
            v_tok = re.sub(r"[^A-Z0-9]", "", v_tok)
            if len(v_tok) == len(mrz_dn):
                ratio = difflib.SequenceMatcher(None, mrz_dn, v_tok).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = v_tok
                    
        if best_match and best_match != mrz_dn:
            log.info("Auto-correcting MRZ doc_number '%s' -> '%s' via VIZ (ratio %.2f).", 
                     mrz_dn, best_match, best_ratio)
            result.doc_number = best_match

    # ── Step 1b: Structured Extraction via Vision LLM (Always-On) ───────────
    # We run Gemini/Ollama on all documents now. 
    # - If MRZ is present, it acts as a shadow cross-check against MRZ.
    # - If MRZ is absent, it acts as the primary data extractor.
    provider = STRUCTURED_OCR_PROVIDER
    if provider == "gemini":
        from modules.gemini_ocr import run_gemini_ocr
        structured, provider_error = run_gemini_ocr(raw_bytes, content_type, result.viz_text)
    elif provider == "ollama":
        from modules.ollama_ocr import run_ollama_ocr
        structured, provider_error = run_ollama_ocr(raw_bytes, content_type, result.viz_text)
    else:
        structured = {}
        provider_error = (
            f"Unsupported STRUCTURED_OCR_PROVIDER '{provider}'. "
            "Use 'gemini' or 'ollama'."
        )

    if structured:
        # Save the raw dictionary for shadow mode forensics
        result.vision_extracted_data = structured
        
        # Always apply gatekeeping flags
        result.is_government_id = structured.get("is_government_id", True)
        result.id_type         = structured.get("id_type", "")
        result.id_reasoning    = structured.get("reasoning", "")
        
        # Capture tampering flags from LLM
        result.vlm_tamper_detected = structured.get("visual_tampering_detected", False)
        result.vlm_tamper_reason = structured.get("tampering_reasoning", "")
        
        # If MRZ is NOT detected, the LLM fields become the primary data source
        if not result.mrz_detected:
            result.surname         = structured.get("surname", "")
            result.given_names     = structured.get("given_names", "")
            result.doc_number      = structured.get("doc_number", "")
            result.dob             = structured.get("dob", "")      # DD/MM/YYYY for non-MRZ
            result.sex             = structured.get("sex", "")
            result.expiry          = structured.get("expiry", "")   # DD/MM/YYYY or ""
            result.doc_type        = structured.get("doc_type", "")
            result.nationality     = structured.get("nationality", "IND")
            result.issuing_country = structured.get("issuing_country", "IND")
            result.address         = structured.get("address", "")
            result.ocr_engine_used = structured.get("engine", result.ocr_engine_used)
            log.info("Non-MRZ extraction complete via '%s': name='%s %s' doc_number='%s'",
                     result.ocr_engine_used, result.given_names, result.surname,
                     result.doc_number)
        else:
            log.info("Shadow extraction complete via '%s' (stored in vision_extracted_data).", 
                     structured.get("engine", provider))
                     
            # MRZ name fields don't have checksums and are noisy (e.g. padding read as S or K).
            # We auto-correct them using the clean LLM extraction if they are highly similar.
            import difflib
            def _is_cleaner_match(mrz_val: str, llm_val: str) -> bool:
                if not mrz_val or not llm_val:
                    return False
                mrz_upper = mrz_val.upper()
                llm_upper = llm_val.upper()
                if llm_upper in mrz_upper:
                    return True
                if difflib.SequenceMatcher(None, mrz_upper, llm_upper).ratio() > 0.7:
                    return True
                return False

            llm_given = structured.get("given_names", "")
            if _is_cleaner_match(result.given_names, llm_given):
                log.info("Auto-correcting MRZ given_names '%s' -> '%s' via LLM.", result.given_names, llm_given)
                result.given_names = llm_given.upper()
                
            llm_sur = structured.get("surname", "")
            if _is_cleaner_match(result.surname, llm_sur):
                log.info("Auto-correcting MRZ surname '%s' -> '%s' via LLM.", result.surname, llm_sur)
                result.surname = llm_sur.upper()
                
            llm_mrz = structured.get("mrz_raw", "")
            if llm_mrz and "<<" in llm_mrz:
                log.info("Auto-correcting MRZ raw text via LLM.")
                result.mrz_raw = llm_mrz.strip()
                
    else:
        result.errors.append(f"Vision structured extraction failed: {provider_error}")

    return result
