"""
SentinelBorder — Utility Helpers
Base64 image codecs, numpy↔JPEG conversions, structured logger.
"""

import base64
import io
import logging
import sys
from typing import Optional

import numpy as np
from PIL import Image

# ─── Logger ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ─── Image Conversion Helpers ─────────────────────────────────────────────────

def bytes_to_pil(raw: bytes) -> Image.Image:
    """Convert raw bytes to a PIL Image (RGB)."""
    return Image.open(io.BytesIO(raw)).convert("RGB")


def bytes_to_numpy(raw: bytes) -> np.ndarray:
    """Convert raw bytes to a NumPy BGR array (OpenCV-compatible)."""
    import cv2
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes with OpenCV.")
    return img


def pil_to_bytes(img: Image.Image, fmt: str = "JPEG", quality: int = 95) -> bytes:
    """Encode a PIL Image to raw bytes."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return buf.getvalue()


def numpy_to_bytes(arr: np.ndarray, ext: str = ".jpg") -> bytes:
    """Encode a NumPy array to JPEG bytes."""
    import cv2
    ok, buf = cv2.imencode(ext, arr)
    if not ok:
        raise ValueError("cv2.imencode failed.")
    return buf.tobytes()


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a NumPy BGR array to a PIL RGB image."""
    import cv2
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert a PIL image to a NumPy BGR array."""
    import cv2
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ─── Base64 Helpers ───────────────────────────────────────────────────────────

def image_to_base64(img: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    """Return a Base64-encoded string of a PIL image (no data-URI prefix)."""
    raw = pil_to_bytes(img, fmt=fmt, quality=quality)
    return base64.b64encode(raw).decode("utf-8")


def numpy_to_base64(arr: np.ndarray) -> str:
    """Return a Base64-encoded JPEG string from a NumPy array."""
    raw = numpy_to_bytes(arr)
    return base64.b64encode(raw).decode("utf-8")


def base64_to_pil(b64: str) -> Image.Image:
    """Decode a Base64 string to a PIL Image."""
    raw = base64.b64decode(b64)
    return bytes_to_pil(raw)


# ─── Misc ─────────────────────────────────────────────────────────────────────

def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))
