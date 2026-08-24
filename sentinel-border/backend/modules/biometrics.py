"""
SentinelBorder — Module 4: Biometric Face Verification
Uses DeepFace with:
  - RetinaFace backend for high-accuracy face detection/cropping
  - ArcFace model for 512-dim Cosine Distance facial embeddings
  - scikit-learn cosine_distances for metric computation

Operates entirely offline after initial model download.

Public API (backward-compatible):
  run_biometrics(doc_bytes, live_bytes) -> BiometricResult

New reusable helpers for three-way verification:
  extract_embedding(image_bytes) -> tuple[np.ndarray | None, Image | None, str]
  compare_embeddings(emb1, emb2)  -> dict {"match": bool, "distance": float}
  run_three_way_verification(doc_bytes, live_bytes, registry_emb_bytes) -> dict
"""

from __future__ import annotations

import io
import struct
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


# ─── Embedding serialisation helpers ─────────────────────────────────────────

def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Serialise a float32 numpy array to raw bytes for SQLite BLOB storage."""
    arr = embedding.astype(np.float32)
    return struct.pack(f"{len(arr)}f", *arr)


def bytes_to_embedding(blob: bytes) -> np.ndarray:
    """Deserialise raw bytes (stored in SQLite) back to a float32 numpy array."""
    count = len(blob) // 4
    arr = struct.unpack(f"{count}f", blob)
    return np.array(arr, dtype=np.float32)


# ─── New reusable public helpers ──────────────────────────────────────────────

def extract_embedding(
    image_bytes: bytes,
) -> tuple[Optional[np.ndarray], Optional[Image.Image], str]:
    """
    Public helper: detect exactly one face in image_bytes and return its
    ArcFace embedding plus the cropped face PIL image.

    Returns:
        (embedding, crop, error_message)
        On success: (np.ndarray, Image, "")
        On failure: (None, None, "human-readable error")
    """
    try:
        img_pil = bytes_to_pil(image_bytes)
        embedding, crop = _get_embedding(img_pil)
        return embedding, crop, ""
    except Exception as exc:
        return None, None, str(exc)


def compare_embeddings(
    emb1: np.ndarray,
    emb2: np.ndarray,
) -> dict:
    """
    Compare two ArcFace embeddings.
    Returns: {"match": bool, "distance": float}
    """
    distance = round(_cosine_distance(emb1, emb2), 6)
    match = distance <= FACE_MATCH_THRESHOLD
    return {"match": match, "distance": distance}


def run_three_way_verification(
    doc_bytes: bytes,
    live_bytes: Optional[bytes],
    registry_embedding_bytes: bytes,
) -> dict:
    """
    Perform three-way biometric verification:
      document ↔ registry, registry ↔ live, document ↔ live.

    Returns the additive registry_verification block dict:
    {
      "registry_record_found": True,
      "document_to_live":     {"match": bool, "distance": float},
      "document_to_registry": {"match": bool, "distance": float},
      "registry_to_live":     {"match": bool, "distance": float},
      "overall_verified":     bool,
      "registry_face_crop_b64": str,
      "error": str  # empty on success
    }
    """
    result: dict = {
        "registry_record_found": True,
        "document_to_live":     {"match": False, "distance": None},
        "document_to_registry": {"match": False, "distance": None},
        "registry_to_live":     {"match": False, "distance": None},
        "overall_verified":     False,
        "registry_face_crop_b64": "",
        "error": "",
    }

    # Deserialise registry embedding
    try:
        reg_emb = bytes_to_embedding(registry_embedding_bytes)
    except Exception as exc:
        result["error"] = f"Registry embedding corrupt: {exc}"
        return result

    # Extract doc embedding
    doc_emb, doc_crop, doc_err = extract_embedding(doc_bytes)
    if doc_emb is None:
        result["error"] = f"Document face not detected: {doc_err}"
        return result

    # Document ↔ Registry
    result["document_to_registry"] = compare_embeddings(doc_emb, reg_emb)

    # Live photo checks
    if live_bytes:
        live_emb, _, live_err = extract_embedding(live_bytes)
        if live_emb is not None:
            result["document_to_live"]  = compare_embeddings(doc_emb, live_emb)
            result["registry_to_live"]  = compare_embeddings(reg_emb, live_emb)
        else:
            result["error"] = f"Live face not detected: {live_err}"
    else:
        # No live photo — only doc ↔ registry can be done
        result["document_to_live"]  = {"match": None, "distance": None}
        result["registry_to_live"]  = {"match": None, "distance": None}

    # Overall: all available comparisons must pass
    comparisons = [result["document_to_registry"]]
    if live_bytes:
        comparisons += [result["document_to_live"], result["registry_to_live"]]
    result["overall_verified"] = all(c.get("match") is True for c in comparisons)

    return result


# ─── Public API (backward-compatible) ────────────────────────────────────────

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
