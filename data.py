"""Data + model layer for the app.

Reads from maintainai.db (via db_utils) and scores machines with the trained model in
risk_model.pkl. The model's inputs come from risk_features, the same code the training
script uses. The risk score is the model's probability, in %, that a machine has a
component failure in the next 7 days.
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
MACHINE_MODELS = ["model1", "model2", "model3", "model4"]
MAINT_TYPES = ["Scheduled", "Corrective"]


@st.cache_resource(show_spinner=False)
def _model_bundle():
    return joblib.load(MODEL_PATH)


def thresholds() -> tuple[int, int]:
    t = _model_bundle()["thresholds"]
    return t["warning"], t["critical"]


def model_name() -> str:
    return {"RandomForestClassifier": "Random Forest classifier",
            "HistGradientBoostingClassifier": "Gradient boosting classifier"}.get(
        type(_model_bundle()["model"]).__name__, type(_model_bundle()["model"]).__name__)


def load_machines() -> pd.DataFrame:
    """Machine register + the model's inputs as of the latest record (df.attrs["as_of"]).

    Everything but the register itself is computed live from the logs, so it can't
    drift out of sync with them.
    """
    tables = rf.load_tables(get_connection())
    feats = rf.current_features(tables)
    df = tables["machines"][["machine_id", "model"]].merge(feats, on="machine_id", how="left")
    df = df.sort_values("machine_id", ignore_index=True)
    df.attrs["as_of"] = rf.reference_date(tables)
    return df


def load_maintenance() -> pd.DataFrame:
    return run_query("""
        SELECT machine_id, date, component, type, COALESCE(note, '') AS note
        FROM Maintenance_Log ORDER BY date DESC, id DESC
    """)


def load_failures() -> pd.DataFrame:
    return run_query("""
        SELECT machine_id, date, component, COALESCE(note, '') AS note
        FROM Failure_Log ORDER BY date DESC, id DESC
    """)


# ── scoring ──────────────────────────────────────────────────────────────────
def _model_input(machines: pd.DataFrame) -> pd.DataFrame:
    return machines[_model_bundle()["features"]]  # exact column order the model was trained on


def score_risk(machines: pd.DataFrame) -> pd.Series:
    """P(failure in the next 7 days), in %."""
    p = _model_bundle()["model"].predict_proba(_model_input(machines))[:, 1]
    return pd.Series((p * 100).round(1), index=machines.index)


# ── explanations: SHAP contributions in plain words ──────────────────────────
@st.cache_resource(show_spinner=False)
def _explainer():
    import shap  # deferred: only needed once a page renders
    return shap.TreeExplainer(_model_bundle()["model"])


@st.cache_resource(show_spinner=False)
def _shap_cache() -> dict:
    """SHAP rows by model input, shared across sessions (the model never changes at runtime)."""
    return {}


FEATURE_LABELS = {
    "age_years": "Machine age", "model_no": "Machine model",
    **{f"days_since_{c}": f"Days since {c} replaced" for c in rf.COMPONENTS},
    **{s: f"{s.capitalize()}, last day" for s in rf.SENSORS},
    **{f"{s}_7d": f"{s.capitalize()}, 7-day mean" for s in rf.SENSORS},
    "errors_7d": "Error codes, last 7 days",
}


def factor_text(feature: str, value: float, contribution: float, fleet: dict) -> str:
    """Plain words for one factor. `fleet` holds fleet medians, to say high or low."""
    v = float(value)
    if feature.startswith("days_since_"):
        c = feature.removeprefix("days_since_")
        return f"{c} not replaced for {v:.0f} days" if contribution > 0 else f"{c} replaced {v:.0f} days ago"
    if feature == "errors_7d":
        n = f"{v:.0f} error code{'s' if v != 1 else ''}"
        return f"{n} in 7 days" if v else "no error codes in 7 days"
    if feature == "age_years":
        return f"machine age {v:.0f} years"
    if feature == "model_no":
        return f"machine model{v:.0f}"
    base = feature.removesuffix("_7d")
    when = "7-day mean" if feature.endswith("_7d") else "last day"
    ref = fleet.get(feature, v)
    level = "normal" if abs(v - ref) <= 0.005 * abs(ref) else ("high" if v > ref else "low")
    return f"{level} {base} ({when} {v:.1f}, fleet {ref:.1f})"


def fleet_medians(machines: pd.DataFrame) -> dict:
    return _model_input(machines).median().to_dict()


def explain_risk(machines: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """SHAP contributions per machine (rows) and feature (columns), in percentage points.

    The fleet's average risk (the returned base value) plus a row's contributions
    equals that machine's risk score.
    """
    X = _model_input(machines)
    ex, cache = _explainer(), _shap_cache()
    # exact SHAP for 100 machines takes seconds; a saved record changes one machine's
    # inputs, so only rows not seen before are computed
    keys = [tuple(r) for r in X.itertuples(index=False)]
    new = list(dict.fromkeys(k for k in keys if k not in cache))
    if new:
        vals = ex.shap_values(pd.DataFrame(new, columns=X.columns))
        vals = np.asarray(vals[1] if isinstance(vals, list) else vals)
        if vals.ndim == 3:  # (rows, features, classes): keep the "failure" class
            vals = vals[..., 1]
        cache.update(zip(new, vals * 100))
    base = float(np.ravel(ex.expected_value)[-1])
    return base * 100, pd.DataFrame([cache[k] for k in keys], index=machines.index, columns=X.columns)


def explain_factors(machines: pd.DataFrame, contribs: pd.DataFrame, i,
                    fleet: dict | None = None) -> list[tuple[str, float]]:
    """(plain words, contribution) for every factor of machine row i. Pass `fleet` (medians
    of the whole fleet) when `machines` is not the fleet, e.g. a what-if frame."""
    X, fleet = _model_input(machines), fleet or fleet_medians(machines)
    return [(factor_text(f, X.at[i, f], c, fleet), c) for f, c in contribs.loc[i].items()]


def top_factors(machines: pd.DataFrame, contribs: pd.DataFrame, n: int = 2,
                min_points: float = 1.0) -> pd.Series:
    """Plain-words summary of the n factors pushing each machine's risk up most."""
    X, fleet = _model_input(machines), fleet_medians(machines)
    out = {}
    for i, row in contribs.iterrows():
        up = row[row >= min_points].sort_values(ascending=False).head(n)
        out[i] = [factor_text(f, X.at[i, f], c, fleet) for f, c in up.items()]
    return pd.Series(out)


def global_importance(contribs: pd.DataFrame) -> pd.Series:
    """Mean absolute SHAP contribution per feature across the fleet, largest first."""
    return contribs.abs().mean().sort_values(ascending=False)


# ── what-if ──────────────────────────────────────────────────────────────────
def what_if(machines: pd.DataFrame, machine_id: str, replace=(), sensors_normal: bool = False,
            clear_errors: bool = False) -> pd.DataFrame:
    """Two rows for one machine, "now" and "after": the model inputs with the simulated
    actions applied. Nothing is written to the database.

    replace         components replaced today: their days-since-replacement drop to 0
    sensors_normal  every sensor reading back at the fleet median (e.g. the fault fixed)
    clear_errors    no error codes in the last 7 days
    """
    now = machines[machines["machine_id"] == machine_id].iloc[[0]].copy()
    after = now.copy()
    for c in replace:
        after[f"days_since_{c}"] = 0.0
    if sensors_normal:
        fleet = fleet_medians(machines)
        for s in rf.SENSORS:
            after[s] = fleet[s]
            after[f"{s}_7d"] = fleet[f"{s}_7d"]
    if clear_errors:
        after["errors_7d"] = 0.0
    out = pd.concat([now, after], ignore_index=True)
    out.index = ["now", "after"]
    return out


def monthly_failures(failures: pd.DataFrame, months: int = 12) -> tuple[list[str], list[int]]:
    d = failures.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    counts = d.dropna(subset=["date"]).set_index("date").resample("ME").size()
    idx = pd.date_range(end=counts.index.max() if len(counts) else pd.Timestamp.today(),
                        periods=months, freq="ME")
    counts = counts.reindex(idx, fill_value=0)
    return [i.strftime("%b") for i in counts.index], counts.astype(int).tolist()


# ── writes: persist to the real database, not just session state ─────────────
def add_machine(machine_id: str, model: str, age_years: int):
    run_write("INSERT INTO Machines (machine_id, model, age_years) VALUES (?,?,?)",
              (machine_id, model, int(age_years)))


def add_maintenance(machine_id: str, date, component: str, mtype: str, note: str = ""):
    run_write(
        "INSERT INTO Maintenance_Log (machine_id, date, component, type, note) VALUES (?,?,?,?,?)",
        (machine_id, str(date), component, mtype, note.strip() or None))


def add_failure(machine_id: str, date, component: str, note: str = "", replaced: bool = True):
    """Log a failure; with replaced=True also log the corrective replacement, as the
    source data does for (almost) every failure."""
    run_write("INSERT INTO Failure_Log (machine_id, date, component, note) VALUES (?,?,?,?)",
              (machine_id, str(date), component, note.strip() or None))
    if replaced:
        add_maintenance(machine_id, date, component, "Corrective", "Replaced after failure")
