"""
Shared database helpers for the Smart Predictive Maintenance app.
All pages import from here so there's one source of truth for the DB path
and query logic (avoids the exact kind of drift this project has been
careful to avoid elsewhere).
"""
import sqlite3
import pandas as pd
import streamlit as st

DB_PATH = "factory_maintenance.db"  # expects the .db file in the same folder as app.py


@st.cache_resource
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


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


def get_machine_types():
    return ["CNC Machine", "Hydraulic Press", "Conveyor Belt", "Air Compressor",
            "Injection Molding Machine", "Welding Robot", "Packaging Machine",
            "Industrial Boiler", "Centrifugal Pump", "Diesel Generator", "Other"]


def get_locations():
    return ["Production Line A", "Production Line B", "Production Line C",
            "Packaging Area", "Utility Room", "Warehouse", "Other"]


def next_machine_id():
    import re
    ids = get_machine_ids()
    nums = []
    for i in ids:
        m = re.match(r"^MCH-(\d+)$", i)
        if m:
            nums.append(int(m.group(1)))
    n = (max(nums) + 1) if nums else 1
    return f"MCH-{n:03d}"
