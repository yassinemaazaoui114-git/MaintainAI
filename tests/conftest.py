"""Every test runs against a throwaway copy of factory_maintenance.db, never the real one."""

import hashlib
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REAL_DB = ROOT / "factory_maintenance.db"
sys.path.insert(0, str(ROOT))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="session", autouse=True)
def real_db_untouched():
    before = _digest(REAL_DB)
    yield
    assert _digest(REAL_DB) == before, "a test wrote to the real database"


@pytest.fixture
def temp_db(tmp_path, monkeypatch) -> Path:
    db = tmp_path / "factory_maintenance.db"
    shutil.copy(REAL_DB, db)
    monkeypatch.setenv("MAINTAINAI_DB", str(db))  # read by db_utils.db_path() on every query
    return db


@pytest.fixture
def query(temp_db):
    def run(sql, params=()):
        with sqlite3.connect(temp_db) as conn:
            return conn.execute(sql, params).fetchall()
    return run
