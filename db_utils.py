"""
Shared database helpers for the MaintainAI app.
All pages import from here so there's one source of truth for the DB path
and query logic (avoids the exact kind of drift this project has been
careful to avoid elsewhere).
"""
import os
import re
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

# next to app.py by default; MAINTAINAI_DB points the app at another copy (the tests do)
DEFAULT_DB = Path(__file__).resolve().with_name("maintainai.db")


def db_path() -> Path:
    return Path(os.environ.get("MAINTAINAI_DB", DEFAULT_DB))


@st.cache_resource(show_spinner=False)
def _connect(path: str) -> sqlite3.Connection:
    return sqlite3.connect(path, check_same_thread=False)


def get_connection() -> sqlite3.Connection:
    return _connect(str(db_path()))


def run_query(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = get_connection()
    return pd.read_sql_query(sql, conn, params=params)


def run_write(sql: str, params: tuple):
    conn = get_connection()
    conn.execute(sql, params)
    conn.commit()


def get_machine_ids():
    df = run_query("SELECT machine_id FROM Machines ORDER BY machine_id")
    return df["machine_id"].tolist()


def next_machine_id():
    nums = []
    for i in get_machine_ids():
        m = re.match(r"^MCH-(\d+)$", i)
        if m:
            nums.append(int(m.group(1)))
    n = (max(nums) + 1) if nums else 1
    return f"MCH-{n:03d}"
