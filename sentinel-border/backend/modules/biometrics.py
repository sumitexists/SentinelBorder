"""
SentinelBorder — Module 4: Biometric Face Verification
Uses DeepFace with:
  - RetinaFace backend for high-accuracy face detection/cropping
  - ArcFace model for 512-dim Cosine Distance facial embeddings
  - scikit-learn cosine_distances for metric computation

Operates entirely offline after initial model download.
"""

from __future__ import annotations

import io
import tempfile
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

from config import (
    DEEPFACE_DETECTOR,
    DEEPFACE_MODEL,
    FACE_MATCH_THRESHOLD,
    FACE_WARN_THRESHOLD,
)
from utils.helpers import bytes_to_pil, get_logger, image_to_base64, pil_to_numpy

log = get_logger("biometrics")


# ─── Output Data Structure ────────────────────────────────────────────────────

@dataclass
class BiometricResult:
    face_detected_doc: bool = False
    face_detected_live: bool = False
    cosine_distance: Optional[float] = None
    match: bool = False
    status: str = "no_comparison"      # "verified" | "review" | "mismatch" | "no_comparison"
    confidence_pct: float = 0.0
    doc_face_crop_b64: str = ""
    live_face_crop_b64: str = ""
    warnings: list[str] = field(default_factory=list)


# ─── Face Detection & Embedding ───────────────────────────────────────────────

def _get_embedding(img_pil: Image.Image) -> tuple[np.ndarray, Optional[Image.Image]]:
    """
    Extract ArcFace embedding for the largest face found in img_pil.
    Returns (embedding_vector, cropped_face_pil).
    Raises ValueError if no face detected.
    """
    from deepface import DeepFace

    # Save PIL to temp file — DeepFace requires a file path
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img_pil.save(tmp, format="JPEG", quality=95)
        tmp_path = tmp.name

    try:
        # Extract face + embedding
        results = DeepFace.represent(
            img_path=tmp_path,
            model_name=DEEPFACE_MODEL,
            detector_backend=DEEPFACE_DETECTOR,
            enforce_detection=True,
            align=True,
        )
        os.unlink(tmp_path)

        if not results:
            raise ValueError("No face representation returned.")

        # Use the highest-confidence detection (first result by default)
        best = results[0]
        embedding = np.array(best["embedding"], dtype=np.float32)

        # Crop face region for display
        face_region = best.get("facial_area", {})
        x = face_region.get("x", 0)
        y = face_region.get("y", 0)
        w = face_region.get("w", img_pil.width)
        h = face_region.get("h", img_pil.height)

        # Add 15% padding around the face
        pad_x = int(w * 0.15)
        pad_y = int(h * 0.15)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_pil.width, x + w + pad_x)
        y2 = min(img_pil.height, y + h + pad_y)

        crop = img_pil.crop((x1, y1, x2, y2))
        return embedding, crop

    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ─── Cosine Distance ──────────────────────────────────────────────────────────

def _cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Cosine_Distance = 1 - (v1 · v2) / (||v1|| × ||v2||)
    Clipped to [0, 1].
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 1.0
    cosine_similarity = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.clip(1.0 - cosine_similarity, 0.0, 1.0))


# ─── Public API ───────────────────────────────────────────────────────────────

def run_biometrics(
    doc_bytes: bytes,
    live_bytes: Optional[bytes],
) -> BiometricResult:
    """
    1. Extract ArcFace embedding from the document image (passport photo crop).
    2. If live_bytes provided, extract embedding from the live webcam snapshot.
    3. Compute Cosine Distance and determine match status.
    """
    br = BiometricResult()

    if live_bytes is None:
        br.warnings.append("No live photo provided — biometric comparison skipped.")
        return br

    # ── Document Face ─────────────────────────────────────────────────────────
    try:
        doc_pil = bytes_to_pil(doc_bytes)
        doc_embedding, doc_crop = _get_embedding(doc_pil)
        br.face_detected_doc = True
        if doc_crop:
            br.doc_face_crop_b64 = image_to_base64(doc_crop, quality=85)
        log.info("Document face embedding extracted (dim=%d).", len(doc_embedding))
    except Exception as exc:
        br.warnings.append(f"Document face detection failed: {exc}")
        log.warning("Doc face error: %s", exc)
        return br

    # ── Live Face ─────────────────────────────────────────────────────────────
    try:
        live_pil = bytes_to_pil(live_bytes)
        live_embedding, live_crop = _get_embedding(live_pil)
        br.face_detected_live = True
        if live_crop:
            br.live_face_crop_b64 = image_to_base64(live_crop, quality=85)
        log.info("Live face embedding extracted (dim=%d).", len(live_embedding))
    except Exception as exc:
        br.warnings.append(f"Live face detection failed: {exc}")
        log.warning("Live face error: %s", exc)
        return br

    # ── Cosine Distance & Decision ────────────────────────────────────────────
    distance = _cosine_distance(doc_embedding, live_embedding)
    br.cosine_distance = round(distance, 6)

    if distance <= FACE_MATCH_THRESHOLD:
        br.match = True
        br.status = "verified"
        # Confidence scales from 100% (distance=0) to ~50% at threshold boundary
        br.confidence_pct = round((1.0 - distance / FACE_MATCH_THRESHOLD) * 50.0 + 50.0, 1)
    elif distance <= FACE_WARN_THRESHOLD:
        br.match = False
        br.status = "review"
        br.confidence_pct = round((1.0 - (distance - FACE_MATCH_THRESHOLD) /
                                   (FACE_WARN_THRESHOLD - FACE_MATCH_THRESHOLD)) * 30.0 + 20.0, 1)
        br.warnings.append(
            f"Biometric REVIEW: Cosine distance={distance:.4f} — secondary identity "
            f"verification required."
        )
    else:
        br.match = False
        br.status = "mismatch"
        br.confidence_pct = round(max(0.0, (1.0 - distance) * 20.0), 1)
        br.warnings.append(
            f"Biometric MISMATCH: Cosine distance={distance:.4f} > threshold "
            f"{FACE_WARN_THRESHOLD} — possible impersonation."
        )

    log.info("Biometric result: distance=%.4f status=%s", distance, br.status)
    return br
