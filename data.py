"""Real data + model loaders, replacing the sample/placeholder versions.

Reads from factory_maintenance.db (via db_utils, the same tested DB layer
used elsewhere in this project) and scores machines with the actual trained
model in risk_model.pkl — not a placeholder formula. The model's inputs come
from risk_features, the same code the training script uses.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import risk_features as rf
from db_utils import get_connection, run_query, run_write

MODEL_PATH = Path(__file__).resolve().with_name("risk_model.pkl")
MACHINE_COLS = ["machine_id", "machine_type", "location", "install_date", "operating_hours"]


@st.cache_resource(show_spinner=False)
def _model_bundle():
    return joblib.load(MODEL_PATH)


def thresholds() -> tuple[int, int]:
    t = _model_bundle()["thresholds"]
    return t["warning"], t["critical"]


def load_machines() -> pd.DataFrame:
    """Base machine info + the AI feature set, joined into one DataFrame.

    'failures' and 'downtime' (trailing 12 months) are computed live from
    Failure_Log, never stored on the machine itself — so they can't drift out
    of sync with the actual log. Features are as of the latest record date
    (see risk_features), exposed as df.attrs["as_of"].
    """
    base = run_query("""
        SELECT machine_id, type AS machine_type, location, install_date, operating_hours
        FROM Machines ORDER BY machine_id
    """)
    tables = rf.load_tables(get_connection())
    feats = rf.current_features(tables).rename(columns={
        "failure_count": "failures", "total_downtime": "downtime",
    }).drop(columns="operating_hours")

    df = base.merge(feats, on="machine_id", how="left")
    df.attrs["as_of"] = rf.reference_date(tables)
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
def _model_input(machines: pd.DataFrame) -> pd.DataFrame:
    return machines.rename(columns={
        "failures": "failure_count", "downtime": "total_downtime"
    })[_model_bundle()["features"]]  # exact column order the model was trained on


def score_risk(machines: pd.DataFrame) -> pd.Series:
    model = _model_bundle()["model"]
    return pd.Series(model.predict(_model_input(machines)).round(1), index=machines.index).clip(0, 100)


# ── explanations: SHAP contributions in plain words ──────────────────────────
@st.cache_resource(show_spinner=False)
def _explainer():
    import shap  # deferred: only needed once the dashboard renders
    return shap.TreeExplainer(_model_bundle()["model"])


# (raises risk, lowers risk) wording per feature; v is the machine's value
_PHRASES = {
    "days_since_maintenance": (lambda v: f"{v:.0f} days since maintenance",
                               lambda v: f"serviced {v:.0f} days ago"),
    "failure_count": (lambda v: f"{v:.0f} failure{'s' if v != 1 else ''} in 12 months",
                      lambda v: "no failures in 12 months" if v == 0 else f"only {v:.0f} failure{'s' if v != 1 else ''} in 12 months"),
    "total_downtime": (lambda v: f"{v:.1f} h downtime in 12 months",
                       lambda v: f"low downtime ({v:.1f} h in 12 months)"),
    "vibration": (lambda v: f"high vibration ({v:.1f} mm/s)",
                  lambda v: f"normal vibration ({v:.1f} mm/s)"),
    "temperature": (lambda v: f"high temperature ({v:.0f} °C)",
                    lambda v: f"normal temperature ({v:.0f} °C)"),
    "age_years": (lambda v: f"older machine ({v:.1f} years)",
                  lambda v: f"newer machine ({v:.1f} years)"),
    "operating_hours": (lambda v: f"{v:,.0f} operating hours",
                        lambda v: f"low running hours ({v:,.0f} h)"),
    "maintenance_frequency": (lambda v: f"only {v:.1f} services/yr",
                              lambda v: f"regular servicing ({v:.1f}/yr)"),
}


def explain_risk(machines: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """SHAP contributions per machine (rows) and feature (columns), in score points.

    The model's average score (the returned base value) plus a row's contributions
    equals that machine's predicted score.
    """
    X = _model_input(machines)
    ex = _explainer()
    vals = np.asarray(ex.shap_values(X))
    base = float(np.ravel(ex.expected_value)[0])
    return base, pd.DataFrame(vals, index=machines.index, columns=X.columns)


def factor_text(feature: str, value: float, contribution: float) -> str:
    up, down = _PHRASES[feature]
    return (up if contribution > 0 else down)(float(value))


def top_factors(machines: pd.DataFrame, contribs: pd.DataFrame, n: int = 2,
                min_points: float = 1.0) -> pd.Series:
    """Plain-words summary of the n factors pushing each machine's risk up most."""
    X = _model_input(machines)
    out = {}
    for i, row in contribs.iterrows():
        up = row[row >= min_points].sort_values(ascending=False).head(n)
        out[i] = [factor_text(f, X.at[i, f], c) for f, c in up.items()]
    return pd.Series(out)


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
