"""
SentinelBorder — Module 1b: Gemini Vision OCR
Extracts structured identity fields from any non-MRZ document using Gemini
Vision.  Gemini is the only non-MRZ structured extraction provider.
"""

from __future__ import annotations

import json
import re
from config import GEMINI_API_KEY, GEMINI_MODEL
from utils.helpers import get_logger

log = get_logger("gemini_ocr")


# ─── Extraction Prompt ────────────────────────────────────────────────────────

_PROMPT = """
You are an expert identity document scanner deployed at an Indian border checkpoint.

Carefully examine this identity document image and extract ALL visible text fields.

Return ONLY a valid JSON object — no markdown fences, no explanation — with exactly these keys:

{
  "is_government_id": "boolean — true if it is a formally recognized government-issued ID (e.g. Passport, National ID, Voter ID, PAN, Aadhaar, Driving Licence), false if it is a non-government/invalid ID (e.g. corporate badge, school ID, gym pass)",
  "id_type": "string — e.g. 'Corporate ID', 'Voter ID', 'Aadhaar', 'Unknown'",
  "confidence_score": "float 0.0 to 1.0",
  "reasoning": "string — brief explanation of why it is or isn't a government ID",
  "doc_type": "concise UPPER_SNAKE_CASE document type, e.g. AADHAAR, PAN, VOTER_ID, DRIVING_LICENCE, PASSPORT, VISA, NATIONAL_ID, or UNKNOWN",
  "surname": "family/last name only (string)",
  "given_names": "first + middle names (string)",
  "doc_number": "primary ID number — Aadhaar: 12 digits no spaces, PAN: 10-char alphanumeric, EPIC: 10-char, DL: as printed",
  "dob": "DD/MM/YYYY format, or just YYYY if only year is shown, or empty string",
  "sex": "M or F or empty string",
  "expiry": "DD/MM/YYYY if applicable, else empty string",
  "nationality": "3-letter ISO country code, e.g. IND",
  "issuing_country": "3-letter ISO country code, e.g. IND",
  "address": "full address text if visible, else empty string",
  "mrz_raw": "Array of strings, containing the 2 or 3 lines of the Machine Readable Zone (MRZ) at the bottom if present. Pay strict attention to the padding characters '<'. If no MRZ, return empty array.",
  "visual_tampering_detected": "boolean. Set to true if the document appears digitally altered (e.g. text looks glued on, mismatched fonts, blurred background behind text, misaligned text, etc).",
  "tampering_reasoning": "string. If visual_tampering_detected is true, briefly describe the visual anomaly.",
  "engine": "gemini"
}

Strict rules:
- AADHAAR: the 12-digit UID printed in groups of 4 is doc_number — copy every digit and omit spaces
- PAN: exactly 5 uppercase letters + 4 digits + 1 uppercase letter (e.g. ABCDE1234F)
- VOTER_ID: typically 10-character EPIC number (e.g. ABC1234567)
- For any other document type, doc_number is its primary visible identifier or registration number.
- Copy names in their displayed order. Put the family name in surname only when it is explicitly identifiable; otherwise put the complete name in given_names and leave surname empty.
- Classify the document from its visual branding and text. Do not return UNKNOWN for a recognisable document.
- If a field is not visible or not applicable, use empty string ""
- Never hallucinate or guess data not clearly visible in the image
- sex must be exactly "M", "F", or ""
- Return raw JSON only — the system will parse it directly
"""


# ─── Public API ───────────────────────────────────────────────────────────────

def _normalise_response(data: dict) -> dict:
    """Return only the API fields in the format expected by the screening API."""
    fields = (
        "doc_type", "surname", "given_names", "doc_number", "dob", "sex",
        "expiry", "nationality", "issuing_country", "address", "tampering_reasoning"
    )
    cleaned = {field: str(data.get(field, "") or "").strip() for field in fields}
    
    # Handle mrz_raw which can be a list of strings
    mrz_data = data.get("mrz_raw")
    if isinstance(mrz_data, list):
        cleaned["mrz_raw"] = "\n".join(str(line) for line in mrz_data if line).strip()
    else:
        cleaned["mrz_raw"] = str(mrz_data or "").strip()
    
    # Gatekeeping fields
    cleaned["is_government_id"] = bool(data.get("is_government_id", True))
    cleaned["id_type"] = str(data.get("id_type", "")).strip()
    cleaned["visual_tampering_detected"] = bool(data.get("visual_tampering_detected", False))
    try:
        cleaned["confidence_score"] = float(data.get("confidence_score", 1.0))
    except (ValueError, TypeError):
        cleaned["confidence_score"] = 1.0
    cleaned["reasoning"] = str(data.get("reasoning", "")).strip()

    cleaned["doc_type"] = re.sub(r"[^A-Z0-9_]+", "_", cleaned["doc_type"].upper()).strip("_")
    cleaned["doc_type"] = cleaned["doc_type"] or "UNKNOWN"

    sex = cleaned["sex"].upper()
    cleaned["sex"] = sex[0] if sex and sex[0] in ("M", "F") else ""
    cleaned["nationality"] = cleaned["nationality"].upper()
    cleaned["issuing_country"] = cleaned["issuing_country"].upper()

    if cleaned["doc_type"] == "AADHAAR":
        cleaned["doc_number"] = re.sub(r"[\s-]+", "", cleaned["doc_number"])

    cleaned["engine"] = "gemini"
    return cleaned


def run_gemini_ocr(
    raw_bytes: bytes,
    content_type: str = "image/jpeg",
    viz_text: str = "",
) -> tuple[dict, str]:
    """
    Call Gemini Vision to extract structured identity fields from a document image.

    Args:
        raw_bytes:    Raw file bytes (JPEG, PNG, or PDF).
        content_type: MIME type of the document.

    Returns:
        A pair of (extracted fields, error message).  Fields are empty only when
        Gemini could not produce a usable response; callers must not substitute
        a different OCR engine.
    """
    if not GEMINI_API_KEY:
        message = "Gemini extraction is unavailable: GEMINI_API_KEY is not configured."
        log.warning(message)
        return {}, message

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)

        # Only accepted inline MIME types for Gemini vision
        safe_mime = content_type if content_type in (
            "image/jpeg", "image/png", "image/webp",
            "image/heic", "image/heif", "application/pdf",
        ) else "image/jpeg"

        log.info("Calling Gemini model='%s' mime='%s' bytes=%d",
                 GEMINI_MODEL, safe_mime, len(raw_bytes))

        supporting_text = (
            "\n\nSupporting OCR transcript (may contain mistakes; use the image as "
            "the source of truth):\n" + viz_text[:4000]
            if viz_text else ""
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=raw_bytes, mime_type=safe_mime),
                types.Part.from_text(text=_PROMPT + supporting_text),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
                max_output_tokens=512,
            ),
        )

        raw_text = response.text.strip() if response.text else ""
        if not raw_text:
            log.warning("Gemini returned empty response.")
            return {}, "Gemini returned an empty extraction response."

        # Strip markdown fences in case model ignores the instruction
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.MULTILINE)
        raw_text = re.sub(r"\s*```\s*$", "", raw_text, flags=re.MULTILINE)
        raw_text = raw_text.strip()

        data = _normalise_response(json.loads(raw_text))

        log.info(
            "Gemini OCR ✓ — doc_type='%s' doc_number='%s' dob='%s' sex='%s'",
            data.get("doc_type"), data.get("doc_number"),
            data.get("dob"), data.get("sex"),
        )
        return data, ""

    except json.JSONDecodeError as exc:
        log.warning("Gemini OCR: JSON parse failed — %s | raw='%.200s'", exc, raw_text)
        return {}, "Gemini returned invalid JSON."
    except Exception as exc:
        log.warning("Gemini OCR failed: %s", exc)
        return {}, f"Gemini extraction failed: {exc}"
