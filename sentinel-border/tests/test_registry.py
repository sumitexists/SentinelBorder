"""
tests/test_registry.py — Registry module unit + integration tests.

These tests use:
- An in-memory / temp-file SQLite DB (no real registry.db touched)
- Random float32 arrays as mock ArcFace embeddings (no DeepFace required)
- FastAPI TestClient for endpoint tests
- pytest.mark.skipif for DeepFace-dependent paths

All tests are independent of any external API keys or model files.
"""

from __future__ import annotations

import io
import os
import sqlite3
import struct
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
_backend = Path(__file__).parent.parent / "backend"
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

# ── Helpers ───────────────────────────────────────────────────────────────────

def _rand_emb(dim: int = 512) -> np.ndarray:
    """Return a random unit-normalised float32 embedding vector."""
    v = np.random.rand(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _emb_to_blob(emb: np.ndarray) -> bytes:
    return struct.pack(f"{len(emb)}f", *emb)


def _blob_to_emb(blob: bytes) -> np.ndarray:
    count = len(blob) // 4
    return np.array(struct.unpack(f"{count}f", blob), dtype=np.float32)


def _white_jpeg_bytes(w: int = 200, h: int = 200) -> bytes:
    img = Image.fromarray(
        np.full((h, w, 3), 220, dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — SQLite registry module (no DeepFace)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_db(tmp_path):
    """Provide a fresh registry.db in a temp directory for each test."""
    from modules.registry import init_db
    db = str(tmp_path / "registry.db")
    init_db(db)
    return db


class TestSchemaInit:
    def test_init_creates_tables(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "people" in tables
        assert "passports" in tables

    def test_init_idempotent(self, tmp_db):
        """Calling init_db twice must not raise or duplicate anything."""
        from modules.registry import init_db
        init_db(tmp_db)  # second call
        conn = sqlite3.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM passports").fetchone()[0]
        conn.close()
        assert count == 0


class TestRegisterPassport:
    def _register(self, db, pnum="A1234567", country="IND", person_id=None, status="active"):
        from modules.registry import register_passport
        return register_passport(
            db,
            passport_number=pnum,
            issuing_country=country,
            surname="SMITH",
            given_names="JOHN",
            dob="850101",
            nationality="IND",
            status=status,
            person_id=person_id,
        )

    def test_register_returns_ids(self, tmp_db):
        result = self._register(tmp_db)
        assert "person_id" in result
        assert "passport_id" in result
        assert result["passport_number"] == "A1234567"

    def test_register_persists_in_db(self, tmp_db):
        self._register(tmp_db)
        conn = sqlite3.connect(tmp_db)
        row = conn.execute("SELECT * FROM passports WHERE passport_number='A1234567'").fetchone()
        conn.close()
        assert row is not None

    def test_unique_constraint_raises(self, tmp_db):
        self._register(tmp_db)
        with pytest.raises(sqlite3.IntegrityError):
            self._register(tmp_db)  # same passport_number + issuing_country

    def test_same_person_multiple_passports(self, tmp_db):
        """Same person_id can have multiple passports with different numbers."""
        pid = str(uuid.uuid4())
        self._register(tmp_db, pnum="A1111111", person_id=pid)
        self._register(tmp_db, pnum="A2222222", person_id=pid)
        conn = sqlite3.connect(tmp_db)
        count = conn.execute(
            "SELECT COUNT(*) FROM passports WHERE person_id=?", (pid,)
        ).fetchone()[0]
        conn.close()
        assert count == 2

    def test_embedding_stored_as_blob(self, tmp_db):
        from modules.registry import register_passport
        emb = _rand_emb(512)
        register_passport(
            tmp_db,
            passport_number="B9999999",
            issuing_country="IND",
            surname="DOE",
            given_names="JANE",
            dob="900505",
            nationality="IND",
            face_embedding=_emb_to_blob(emb),
            embedding_model="ArcFace",
            embedding_dimension=512,
        )
        conn = sqlite3.connect(tmp_db)
        blob = conn.execute(
            "SELECT face_embedding FROM passports WHERE passport_number='B9999999'"
        ).fetchone()[0]
        conn.close()
        assert blob is not None
        recovered = _blob_to_emb(blob)
        assert recovered.shape == (512,)
        assert np.allclose(recovered, emb)


class TestLookupPassport:
    def test_lookup_hit(self, tmp_db):
        from modules.registry import register_passport, lookup_passport
        register_passport(
            tmp_db, passport_number="X1234567", issuing_country="IND",
            surname="JONES", given_names="ALICE", dob="750303", nationality="IND",
        )
        row = lookup_passport(tmp_db, "X1234567", "IND")
        assert row is not None
        assert row["surname"] == "JONES"

    def test_lookup_miss(self, tmp_db):
        from modules.registry import lookup_passport
        assert lookup_passport(tmp_db, "ZZZZZZZZ", "XYZ") is None

    def test_lookup_case_insensitive_country(self, tmp_db):
        from modules.registry import register_passport, lookup_passport
        register_passport(
            tmp_db, passport_number="CI000001", issuing_country="IND",
            surname="TEST", given_names="USER", dob="800808", nationality="IND",
        )
        assert lookup_passport(tmp_db, "CI000001", "ind") is not None  # lower-case


class TestDuplicateDetection:
    def _make_two_passports(self, db):
        from modules.registry import register_passport
        pid = str(uuid.uuid4())
        r1 = register_passport(
            db, passport_number="DUP001", issuing_country="IND",
            surname="X", given_names="Y", dob="800101", nationality="IND",
            person_id=pid, status="active",
        )
        r2 = register_passport(
            db, passport_number="DUP002", issuing_country="IND",
            surname="X", given_names="Y", dob="800101", nationality="IND",
            person_id=pid, status="active",
        )
        return pid, r1["passport_id"], r2["passport_id"]

    def test_finds_second_active_passport(self, tmp_db):
        from modules.registry import get_active_passports_for_person
        pid, id1, id2 = self._make_two_passports(tmp_db)
        others = get_active_passports_for_person(tmp_db, pid, exclude_passport_id=id1)
        assert len(others) == 1
        assert others[0]["passport_number"] == "DUP002"

    def test_excludes_self(self, tmp_db):
        from modules.registry import get_active_passports_for_person
        pid, id1, _ = self._make_two_passports(tmp_db)
        others = get_active_passports_for_person(tmp_db, pid, exclude_passport_id=id1)
        assert all(r["id"] != id1 for r in others)

    def test_expired_not_returned(self, tmp_db):
        from modules.registry import register_passport, get_active_passports_for_person
        pid = str(uuid.uuid4())
        r_active = register_passport(
            tmp_db, passport_number="EXP001", issuing_country="IND",
            surname="X", given_names="Y", dob="800101", nationality="IND",
            person_id=pid, status="active",
        )
        register_passport(
            tmp_db, passport_number="EXP002", issuing_country="IND",
            surname="X", given_names="Y", dob="800101", nationality="IND",
            person_id=pid, status="expired",
        )
        others = get_active_passports_for_person(tmp_db, pid, r_active["passport_id"])
        assert len(others) == 0, "Expired passports must not appear as duplicate threat"


class TestListAndCount:
    def test_count_empty(self, tmp_db):
        from modules.registry import count_passports
        assert count_passports(tmp_db) == 0

    def test_list_returns_metadata(self, tmp_db):
        from modules.registry import register_passport, list_passports
        register_passport(
            tmp_db, passport_number="L0000001", issuing_country="IND",
            surname="LIST", given_names="TEST", dob="910909", nationality="IND",
        )
        records = list_passports(tmp_db)
        assert len(records) == 1
        # face_embedding BLOB must NOT appear in list output
        assert "face_embedding" not in records[0]


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Biometrics helpers (no DeepFace required)
# ──────────────────────────────────────────────────────────────────────────────

class TestEmbeddingSerialization:
    def test_round_trip(self):
        from modules.biometrics import embedding_to_bytes, bytes_to_embedding
        emb = _rand_emb(512)
        blob = embedding_to_bytes(emb)
        recovered = bytes_to_embedding(blob)
        assert recovered.shape == (512,)
        assert np.allclose(emb, recovered, atol=1e-6)

    def test_empty_blob_returns_empty_array(self):
        from modules.biometrics import bytes_to_embedding
        arr = bytes_to_embedding(b"")
        assert arr.shape == (0,)


class TestCompareEmbeddings:
    def test_identical_embeddings_match(self):
        from modules.biometrics import compare_embeddings
        emb = _rand_emb(512)
        result = compare_embeddings(emb, emb)
        assert result["match"] is True
        assert result["distance"] < 0.01

    def test_orthogonal_embeddings_mismatch(self):
        from modules.biometrics import compare_embeddings
        # Two maximally different unit vectors
        emb1 = np.zeros(512, dtype=np.float32); emb1[0] = 1.0
        emb2 = np.zeros(512, dtype=np.float32); emb2[1] = 1.0
        result = compare_embeddings(emb1, emb2)
        assert result["match"] is False
        assert result["distance"] > 0.5

    def test_distance_is_float(self):
        from modules.biometrics import compare_embeddings
        result = compare_embeddings(_rand_emb(), _rand_emb())
        assert isinstance(result["distance"], float)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FastAPI endpoint tests (TestClient, no real DB or DeepFace)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def client(tmp_path, monkeypatch):
    """
    FastAPI TestClient with registry pointing at a fresh temp DB.
    Monkeypatches config so the app uses the temp paths.
    """
    db_path    = str(tmp_path / "test_registry.db")
    photos_dir = str(tmp_path / "photos")
    Path(photos_dir).mkdir()

    # Patch config values before importing app
    monkeypatch.setenv("REGISTRY_DB_PATH",    db_path)
    monkeypatch.setenv("REGISTRY_PHOTOS_DIR", photos_dir)
    monkeypatch.setenv("REGISTRY_ENABLED",    "true")

    # Re-import config + app with patched env
    import importlib
    import config as cfg
    cfg.REGISTRY_DB_PATH    = db_path
    cfg.REGISTRY_PHOTOS_DIR = photos_dir
    cfg.REGISTRY_ENABLED    = True

    from modules.registry import init_db
    init_db(db_path)

    import app as _app
    _app.REGISTRY_DB_PATH    = db_path
    _app.REGISTRY_PHOTOS_DIR = photos_dir
    _app.REGISTRY_ENABLED    = True

    from fastapi.testclient import TestClient
    return TestClient(_app.app, raise_server_exceptions=True)


class TestRegistryEndpoints:

    def _post_passport(self, client, *, pnum="A1234567", country="IND",
                       surname="SMITH", given_names="JOHN", dob="850101",
                       nationality="IND", status="active"):
        return client.post("/api/v1/registry/passports", data={
            "passport_number": pnum,
            "issuing_country": country,
            "surname":         surname,
            "given_names":     given_names,
            "dob":             dob,
            "nationality":     nationality,
            "status":          status,
        })

    def test_register_returns_201(self, client):
        res = self._post_passport(client)
        assert res.status_code == 201
        body = res.json()
        assert body["passport_number"] == "A1234567"
        assert "person_id" in body

    def test_register_missing_field_returns_422(self, client):
        res = client.post("/api/v1/registry/passports", data={
            "passport_number": "A9999999",
            # missing issuing_country, surname, given_names, dob, nationality
        })
        assert res.status_code == 422

    def test_register_duplicate_returns_409(self, client):
        self._post_passport(client)
        res = self._post_passport(client)  # same passport again
        assert res.status_code == 409

    def test_register_invalid_status_returns_422(self, client):
        res = self._post_passport(client, pnum="X9999999", status="invalid_status")
        assert res.status_code == 422

    def test_list_passports_empty(self, client):
        res = client.get("/api/v1/registry/passports")
        assert res.status_code == 200
        assert res.json()["total"] == 0

    def test_list_passports_after_register(self, client):
        self._post_passport(client)
        res = client.get("/api/v1/registry/passports")
        assert res.json()["total"] == 1

    def test_verify_found(self, client):
        self._post_passport(client)
        res = client.get("/api/v1/registry/verify?passport_number=A1234567&issuing_country=IND")
        assert res.status_code == 200
        assert res.json()["found"] is True

    def test_verify_not_found(self, client):
        res = client.get("/api/v1/registry/verify?passport_number=ZZZZZZZZ&issuing_country=XYZ")
        assert res.status_code == 200
        assert res.json()["found"] is False

    def test_verify_case_insensitive(self, client):
        self._post_passport(client)
        res = client.get("/api/v1/registry/verify?passport_number=a1234567&issuing_country=ind")
        assert res.json()["found"] is True

    def test_health_includes_registry_flag(self, client):
        res = client.get("/api/v1/health")
        assert res.status_code == 200
        assert "registry_enabled" in res.json()


class TestScreenEndpointRegistryIntegration:
    """
    Verify that /api/v1/screen always returns the registry_verification
    and duplicate_passport_check keys, and that they are correctly structured
    when MRZ is absent or registry is skipped.
    These tests do not require a working OCR/DeepFace — they verify response shape.
    """

    def _make_white_jpeg_file(self) -> tuple[bytes, str]:
        return _white_jpeg_bytes(), "image/jpeg"

    def test_screen_returns_registry_keys(self, client):
        """
        Even a trivial non-passport image must return the two registry keys
        (they will be empty dicts for non-government IDs, present otherwise).
        """
        img_bytes = _white_jpeg_bytes()
        res = client.post(
            "/api/v1/screen",
            files={"document": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        # We don't care about the threat level — just that the keys exist
        assert res.status_code == 200
        body = res.json()
        assert "registry_verification" in body
        assert "duplicate_passport_check" in body
