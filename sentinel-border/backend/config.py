"""
SentinelBorder — System Configuration
All scoring weights, detection thresholds, and operational constants.
"""

import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}

# Load .env — checks backend/ first, then project root (whichever exists)
try:
    from dotenv import load_dotenv
    _backend_env = Path(__file__).parent / ".env"          # backend/.env  ← user's file
    _root_env    = Path(__file__).parent.parent / ".env"   # project-root/.env
    if _backend_env.exists():
        load_dotenv(_backend_env, override=False)
    elif _root_env.exists():
        load_dotenv(_root_env, override=False)
except ImportError:
    pass  # python-dotenv not installed yet; env vars still work


# ─── ELA (Error Level Analysis) ───────────────────────────────────────────────
ELA_RECOMPRESS_QUALITY: int = 90      # JPEG quality used for re-compression
ELA_TAMPER_THRESHOLD: float = 15.0    # Mean pixel diff above this = tamper flag
# V2 preserves source pixels, adds regional checks, and requires corroboration
# before affecting the composite risk score.
FORENSICS_V2_ENABLED: bool = _env_bool("FORENSICS_V2_ENABLED", True)
ELA_V2_THRESHOLD: float = float(os.getenv("ELA_V2_THRESHOLD", "12.0"))
COPY_MOVE_MIN_INLIERS: int = int(os.getenv("COPY_MOVE_MIN_INLIERS", "10"))
# Shadow mode: run new signals but do not include them in the composite risk score.
# Set to false only after threshold calibration against labelled data.
FORENSICS_SHADOW_MODE: bool = _env_bool("FORENSICS_SHADOW_MODE", True)
# Enable document-region-aware ELA anomaly checks (requires OCR doc_type).
FORENSICS_USE_REGION_ANALYSIS: bool = _env_bool("FORENSICS_USE_REGION_ANALYSIS", True)
# Points added to composite score per edge-discontinuity event (0 = shadow only).
FORENSICS_EDGE_RISK_WEIGHT: int = int(os.getenv("FORENSICS_EDGE_RISK_WEIGHT", "0"))
# Minimum independent corroborating signals before new tamper evidence affects score.
FORENSICS_V2_CORROBORATION_MIN: int = int(os.getenv("FORENSICS_V2_CORROBORATION_MIN", "2"))

# ─── Biometric Thresholds (Cosine Distance) ────────────────────────────────────
FACE_MATCH_THRESHOLD: float = 0.55    # <= 0.45 → Verified Match
FACE_WARN_THRESHOLD: float = 0.6     # <= 0.55 → Secondary Review; > 0.55 → Mismatch

# ─── DeepFace Model Config ─────────────────────────────────────────────────────
DEEPFACE_MODEL: str = "ArcFace"       # ArcFace | Facenet512
DEEPFACE_DETECTOR: str = "retinaface" # retinaface | opencv | mtcnn

# ─── Composite Threat Score Weights ────────────────────────────────────────────
WEIGHT_CHECKSUM_FAILED: int = 35
WEIGHT_VIZ_MRZ_MISMATCH: int = 30
WEIGHT_ELA_TAMPER: int = 35
WEIGHT_FACE_MISMATCH: int = 40
WEIGHT_METADATA_ANOMALY: int = 20
WEIGHT_EXPIRED: int = 25
WEIGHT_QR_DATA_MISMATCH: int = 30

# ─── Threat Level Bands ────────────────────────────────────────────────────────
THREAT_GREEN_MAX: int = 30    # < 30 → GREEN
THREAT_YELLOW_MAX: int = 70   # 30–69 → YELLOW; >= 70 → RED

# ─── Known editor signatures in EXIF/PDF metadata ──────────────────────────────
SUSPICIOUS_SOFTWARE_TAGS: list[str] = [
    "adobe photoshop",
    "gimp",
    "canva",
    "pixlr",
    "paint.net",
    "affinity photo",
    "inkscape",
    "lightroom",
    "snapseed",
]

# ─── Accepted upload MIME types ────────────────────────────────────────────────
ALLOWED_CONTENT_TYPES: list[str] = [
    "image/jpeg",
    "image/png",
    "application/pdf",
]

# ─── Structured Vision OCR Provider ───────────────────────────────────────────
# Choose "gemini" or "ollama" in .env.  Both providers receive the source image
# and the local OCR transcript; the image remains authoritative.
STRUCTURED_OCR_PROVIDER: str = os.getenv("STRUCTURED_OCR_PROVIDER", "gemini").strip().lower()

# Gemini Vision
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")  # user-specified: 2.5-flash

# Ollama Vision (local-only)
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")

# ─── Passport Registry Module ──────────────────────────────────────────────────
# Set REGISTRY_ENABLED=false to disable the registry lookup step entirely.
REGISTRY_ENABLED: bool = _env_bool("REGISTRY_ENABLED", True)
# Path to the SQLite database file (auto-created on first startup).
REGISTRY_DB_PATH: str = os.getenv(
    "REGISTRY_DB_PATH",
    str(Path(__file__).parent / "data" / "registry.db"),
)
# Directory where registry face photos are stored.
REGISTRY_PHOTOS_DIR: str = os.getenv(
    "REGISTRY_PHOTOS_DIR",
    str(Path(__file__).parent / "data" / "registry_photos"),
)
# Weight added to composite score when a second active passport is confirmed
# for the same identity (POSSIBLE_DUPLICATE_ACTIVE_PASSPORT).
WEIGHT_DUPLICATE_PASSPORT: int = int(os.getenv("WEIGHT_DUPLICATE_PASSPORT", "70"))
