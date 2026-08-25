"""
SentinelBorder — Registry Module
Isolated SQLite passport registry: schema management, registration,
exact lookup, and duplicate-passport detection.

Uses only Python built-in sqlite3 — no extra dependencies.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.helpers import get_logger

log = get_logger("registry")


# ─── DB initialisation ────────────────────────────────────────────────────────

def init_db(db_path: str) -> None:
    """
    Create the people + passports tables if they don't exist.
    Safe to call multiple times (idempotent).
    Enforces UNIQUE(passport_number, issuing_country).
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript("""
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS people (
                person_id  TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS passports (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id           TEXT    NOT NULL REFERENCES people(person_id),
                passport_number     TEXT    NOT NULL,
                issuing_country     TEXT    NOT NULL,
                surname             TEXT    NOT NULL,
                given_names         TEXT    NOT NULL,
                dob                 TEXT    NOT NULL,
                nationality         TEXT    NOT NULL,
                issue_date          TEXT,
                expiry_date         TEXT,
                status              TEXT    NOT NULL DEFAULT 'active',
                photo_path          TEXT,
                face_embedding      BLOB,
                embedding_model     TEXT,
                embedding_dimension INTEGER,
                created_at          TEXT    NOT NULL,
                UNIQUE(passport_number, issuing_country)
            );
        """)
    log.info("Registry DB initialised at %s", db_path)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── Registration ─────────────────────────────────────────────────────────────

def register_passport(
    db_path: str,
    *,
    passport_number: str,
    issuing_country: str,
    surname: str,
    given_names: str,
    dob: str,
    nationality: str,
    issue_date: str = "",
    expiry_date: str = "",
    status: str = "active",
    photo_path: str = "",
    face_embedding: Optional[bytes] = None,
    embedding_model: str = "",
    embedding_dimension: int = 0,
    person_id: Optional[str] = None,
) -> dict:
    """
    Atomically insert a person + passport record.
    Returns { "person_id": str, "passport_id": int, "passport_number": str }.
    Raises sqlite3.IntegrityError if (passport_number, issuing_country) already exists.
    """
    pid = person_id or str(uuid.uuid4())
    now = _now()

    with _connect(db_path) as conn:
        # Upsert person (if person_id is reused across passports)
        conn.execute(
            "INSERT OR IGNORE INTO people (person_id, created_at) VALUES (?, ?)",
            (pid, now),
        )
        cur = conn.execute(
            """
            INSERT INTO passports
                (person_id, passport_number, issuing_country, surname, given_names,
                 dob, nationality, issue_date, expiry_date, status,
                 photo_path, face_embedding, embedding_model, embedding_dimension, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                pid, passport_number, issuing_country.upper(), surname, given_names,
                dob, nationality, issue_date, expiry_date, status,
                photo_path, face_embedding, embedding_model, embedding_dimension, now,
            ),
        )
        passport_id = cur.lastrowid

    log.info(
        "Registered passport %s/%s → person_id=%s id=%d",
        passport_number, issuing_country, pid, passport_id,
    )
    return {"person_id": pid, "passport_id": passport_id, "passport_number": passport_number}


# ─── Lookup ───────────────────────────────────────────────────────────────────

def lookup_passport(
    db_path: str,
    passport_number: str,
    issuing_country: str,
) -> Optional[dict]:
    """
    Exact lookup by (passport_number, issuing_country).
    Returns a plain dict (all columns including face_embedding bytes) or None.
    """
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM passports
            WHERE passport_number = ? AND issuing_country = ?
            LIMIT 1
            """,
            (passport_number, issuing_country.upper()),
        ).fetchone()

    if row is None:
        return None
    return dict(row)


# ─── Duplicate detection ──────────────────────────────────────────────────────

def get_active_passports_for_person(
    db_path: str,
    person_id: str,
    exclude_passport_id: int,
) -> list[dict]:
    """
    Return all **active** passports for person_id, excluding the one just verified.
    Used to detect duplicate active passport fraud.
    """
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, passport_number, issuing_country, status, expiry_date
            FROM passports
            WHERE person_id = ? AND id != ? AND status = 'active'
            """,
            (person_id, exclude_passport_id),
        ).fetchall()
    return [dict(r) for r in rows]


# ─── 1-to-N Search Helpers ────────────────────────────────────────────────────

def get_all_face_embeddings(db_path: str) -> list[tuple[int, str, str, bytes]]:
    """
    Fetch all non-null face embeddings from the passports table.
    Returns a list of (passport_id, passport_number, issuing_country, face_embedding_bytes).
    """
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, passport_number, issuing_country, face_embedding FROM passports WHERE face_embedding IS NOT NULL"
        ).fetchall()
    return [(r["id"], r["passport_number"], r["issuing_country"], r["face_embedding"]) for r in rows]


# ─── List all (metadata only, no embedding bytes) ─────────────────────────────

def list_passports(
    db_path: str,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """
    Return paginated list of all passports — metadata only, no embedding blob.
    """
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, person_id, passport_number, issuing_country,
                   surname, given_names, dob, nationality,
                   issue_date, expiry_date, status,
                   photo_path, embedding_model, embedding_dimension, created_at
            FROM passports
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


def count_passports(db_path: str) -> int:
    with _connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM passports").fetchone()[0]
