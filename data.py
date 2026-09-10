"""Real data + model loaders, replacing the sample/placeholder versions.

Reads from factory_maintenance.db (via db_utils, the same tested DB layer
used elsewhere in this project) and scores machines with the actual trained
model in risk_model.pkl — not a placeholder formula.
"""

from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from db_utils import run_query, run_write

MACHINE_COLS = ["machine_id", "machine_type", "location", "install_date", "operating_hours"]


@st.cache_resource
def _model_bundle():
    return joblib.load("risk_model.pkl")


def load_machines() -> pd.DataFrame:
    """Base machine info + the AI feature set, joined into one DataFrame.

    'failures' and 'downtime' are computed live from Failure_Log (via the
    Machine_Risk_Features view), never stored on the machine itself — so
    they can't drift out of sync with the actual log.
    """
    base = run_query("""
        SELECT machine_id, type AS machine_type, location, install_date, operating_hours
        FROM Machines ORDER BY machine_id
    """)
    feats = run_query("SELECT * FROM Machine_Risk_Features")
    feats = feats.rename(columns={
        "failure_count": "failures",
        "total_downtime_hours": "downtime",
        "maintenance_frequency_per_year": "maintenance_frequency",
        "avg_temperature_recent": "temperature",
        "avg_vibration_recent": "vibration",
    })[["machine_id", "age_years", "days_since_maintenance", "failures",
        "downtime", "maintenance_frequency", "temperature", "vibration"]]

    df = base.merge(feats, on="machine_id", how="left")
    df["temperature"] = df["temperature"].fillna(df["temperature"].mean())
    df["vibration"] = df["vibration"].fillna(df["vibration"].mean())
    df["maintenance_frequency"] = df["maintenance_frequency"].fillna(0)
    df["days_since_maintenance"] = df["days_since_maintenance"].fillna(9999)
    return df


def load_maintenance() -> pd.DataFrame:
    return run_query("""
        SELECT machine_id, date, type, reason, duration_hours AS duration
        FROM Maintenance_Log ORDER BY date DESC
    """)


def load_failures() -> pd.DataFrame:
    return run_query("""
        SELECT machine_id, date, failure_type, cause,
               downtime_hours AS downtime, repair_hours AS repair_time
        FROM Failure_Log ORDER BY date DESC
    """)


# ── model seam: the real trained model, not a placeholder formula ─────────────
def score_risk(machines: pd.DataFrame) -> pd.Series:
    bundle = _model_bundle()
    model, features = bundle["model"], bundle["features"]

    X = machines.rename(columns={
        "failures": "failure_count", "downtime": "total_downtime"
    })[features]  # exact column order the model was trained on

    return pd.Series(model.predict(X).round(1), index=machines.index).clip(0, 100)


def monthly_failures(failures: pd.DataFrame, months: int = 12) -> tuple[list[str], list[int]]:
    d = failures.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    counts = d.dropna(subset=["date"]).set_index("date").resample("ME").size()
    idx = pd.date_range(end=counts.index.max() if len(counts) else pd.Timestamp.today(),
                        periods=months, freq="ME")
    counts = counts.reindex(idx, fill_value=0)
    return [i.strftime("%b") for i in counts.index], counts.astype(int).tolist()


# ── writes: persist to the real database, not just session state ─────────────
def add_machine(machine_id: str, machine_type: str, location: str, install_date, operating_hours: int):
    run_write(
        "INSERT INTO Machines (machine_id, type, location, install_date, operating_hours) VALUES (?,?,?,?,?)",
        (machine_id, machine_type, location, str(install_date), int(operating_hours))
    )


def add_maintenance(machine_id: str, date, mtype: str, reason: str, duration: float):
    run_write(
        "INSERT INTO Maintenance_Log (machine_id, date, type, reason, duration_hours) VALUES (?,?,?,?,?)",
        (machine_id, str(date), mtype, reason, float(duration))
    )


def add_failure(machine_id: str, date, failure_type: str, cause: str, downtime: float, repair_time: float):
    run_write(
        "INSERT INTO Failure_Log (machine_id, date, failure_type, cause, downtime_hours, repair_hours) VALUES (?,?,?,?,?,?)",
        (machine_id, str(date), failure_type, cause, float(downtime), float(repair_time))
    )
