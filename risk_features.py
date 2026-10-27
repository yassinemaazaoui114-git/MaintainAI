"""Feature engineering and the proxy risk formula, shared by the app and the trainer.

One module so the features the model is trained on and the features the app scores
are computed by the same code (training/train_risk_model.py, data.py, the notebook).

Features are computed "as of" a date, using only records dated on or before it:

    age_years               years since install_date
    operating_hours         hour-meter reading (back-estimated for past dates, see below)
    days_since_maintenance  days since the last Maintenance_Log entry
    failure_count           failures in the trailing 12 months
    total_downtime          failure downtime hours in the trailing 12 months
    maintenance_frequency   maintenance events per year, trailing 12 months
    temperature             mean sensor temperature, trailing 56 days (deg C)
    vibration               mean sensor vibration, trailing 56 days (mm/s)

"Now" for the live app is the date of the latest log or sensor record, not the wall
clock: the demo dataset is frozen in time, and measuring against today's date would
make every machine drift towards Critical as the calendar moves on.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

FEATURES = [
    "age_years", "operating_hours", "days_since_maintenance", "failure_count",
    "total_downtime", "maintenance_frequency", "temperature", "vibration",
]
WINDOW_DAYS = 365   # failures, downtime and maintenance frequency
SENSOR_DAYS = 56    # sensor averages


# ── raw data ─────────────────────────────────────────────────────────────────
def load_tables(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    q = lambda sql: pd.read_sql_query(sql, conn)
    t = {
        "machines": q("SELECT machine_id, install_date, operating_hours FROM Machines"),
        "maintenance": q("SELECT machine_id, date FROM Maintenance_Log"),
        "failures": q("SELECT machine_id, date, downtime_hours FROM Failure_Log"),
        "sensors": q("SELECT machine_id, date, temperature, vibration FROM Sensor_Readings"),
    }
    t["machines"]["install_date"] = pd.to_datetime(t["machines"]["install_date"], errors="coerce")
    for k in ("maintenance", "failures", "sensors"):
        t[k]["date"] = pd.to_datetime(t[k]["date"], errors="coerce")
        t[k] = t[k].dropna(subset=["date"])
    return t


def reference_date(t: dict[str, pd.DataFrame]) -> pd.Timestamp:
    """Latest dated record across the logs and sensor readings ("now" for the app)."""
    dates = [t[k]["date"].max() for k in ("maintenance", "failures", "sensors") if len(t[k])]
    dates = [d for d in dates if pd.notna(d)]
    return max(dates) if dates else pd.Timestamp.today().normalize()


# ── features ─────────────────────────────────────────────────────────────────
def features_at(t: dict[str, pd.DataFrame], as_of, ref: pd.Timestamp | None = None,
                all_machines: bool = False) -> pd.DataFrame:
    """One row per machine installed on or before `as_of`: machine_id + FEATURES.

    With all_machines=True (the live app) a machine whose install date is later than
    `as_of` — e.g. one just added with today's date — is kept and treated as brand new.

    Machines.operating_hours is a single meter reading taken at the reference date,
    so for earlier dates the hours are back-estimated assuming the machine accrued
    them evenly since installation (an assumption; there is no hour-meter history).
    """
    as_of = pd.Timestamp(as_of)
    ref = reference_date(t) if ref is None else ref
    m = t["machines"]
    if not all_machines:
        m = m[m["install_date"].fillna(as_of) <= as_of]
    m = m.copy()
    mid = m["machine_id"]

    def upto(df, days=None):
        keep = df["date"] <= as_of
        if days is not None:
            keep &= df["date"] > as_of - pd.Timedelta(days=days)
        return df[keep]

    install = m["install_date"].fillna(as_of).clip(upper=as_of)
    out = pd.DataFrame({"machine_id": mid.values})
    out["age_years"] = ((as_of - install).dt.days / 365.25).clip(lower=0).round(2).values

    life_now = (ref - install).dt.days
    life_then = (as_of - install).dt.days.clip(lower=0)
    share = np.where(life_now > 0, (life_then / life_now.where(life_now > 0, 1)).clip(upper=1), 1.0)
    out["operating_hours"] = (m["operating_hours"].astype(float) * share).round(0).values

    last = upto(t["maintenance"]).groupby("machine_id")["date"].max()
    since = (as_of - mid.map(last)).dt.days
    out["days_since_maintenance"] = since.fillna((as_of - install).dt.days).clip(lower=0).values

    f12 = upto(t["failures"], WINDOW_DAYS).groupby("machine_id")
    out["failure_count"] = mid.map(f12.size()).fillna(0).astype(int).values
    out["total_downtime"] = mid.map(f12["downtime_hours"].sum()).fillna(0).round(1).values

    # annualise over the part of the window the machine actually existed (min. 90 days)
    m12 = upto(t["maintenance"], WINDOW_DAYS).groupby("machine_id").size()
    span = (as_of - install).dt.days.clip(lower=90, upper=WINDOW_DAYS)
    out["maintenance_frequency"] = (mid.map(m12).fillna(0) * 365 / span).round(2).values

    s = upto(t["sensors"])
    recent = s[s["date"] > as_of - pd.Timedelta(days=SENSOR_DAYS)].groupby("machine_id")[["temperature", "vibration"]].mean()
    latest = s.sort_values("date").groupby("machine_id")[["temperature", "vibration"]].last()
    for col in ("temperature", "vibration"):
        v = mid.map(recent[col]).fillna(mid.map(latest[col]))
        fleet = recent[col].median() if len(recent) else (s[col].median() if len(s) else 0.0)
        out[col] = v.fillna(fleet).round(2).values
    return out


def current_features(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ref = reference_date(t)
    return features_at(t, ref, ref, all_machines=True)


def weekly_snapshots(t: dict[str, pd.DataFrame], start=None, end=None) -> pd.DataFrame:
    """Every machine's features as of each week. Starts one full window after the
    first record by default, so the 12-month counts are never truncated."""
    ref = reference_date(t)
    first = min(t[k]["date"].min() for k in ("maintenance", "failures", "sensors"))
    start = pd.Timestamp(start) if start is not None else first + pd.Timedelta(days=WINDOW_DAYS)
    end = pd.Timestamp(end) if end is not None else ref
    frames = []
    # anchored on `end` so the last snapshot is exactly the live feature set
    for d in pd.date_range(end=end, periods=(end - start).days // 7 + 1, freq="7D"):
        f = features_at(t, d, ref)
        f.insert(1, "snapshot_date", d)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


# ── proxy target ─────────────────────────────────────────────────────────────
# There are no real failure labels, so the training target is this designed formula.
# Each feature is mapped onto 0-1 between a fixed "healthy" and "bad" value (clipped),
# then combined with weights that sum to 1:  risk = 100 * sum(weight * normalised).
# Ranges are fixed engineering judgements, not fitted to the data, so the formula
# doesn't move when records are added.
#
#   feature                 weight  0 (healthy) -> 1 (bad)   rationale
PROXY_TERMS = {
    "failure_count":          (0.22, 0, 6),      # recent failures: strongest sign of trouble
    "days_since_maintenance": (0.16, 0, 180),    # overdue service; 6 months = fully overdue
    "vibration":              (0.16, 2.0, 8.0),  # mm/s; ISO 10816-style zones, ~7 mm/s is severe
    "temperature":            (0.12, 40, 85),    # deg C above a normal running temperature
    "total_downtime":         (0.10, 0, 150),    # hours lost to failures in 12 months
    "age_years":              (0.10, 0, 12),     # wear-out with age
    "operating_hours":        (0.08, 0, 40000),  # cumulative running hours
    "maintenance_frequency":  (0.06, 6, 0),      # inverted: 6+ services/yr = 0, none = 1
}
PROXY_NOISE_SD = 4.0  # unexplained variation, in score points


def proxy_components(X: pd.DataFrame) -> pd.DataFrame:
    """Each feature normalised to 0-1 (1 = worst) as the proxy formula sees it."""
    return pd.DataFrame(
        {f: ((X[f] - lo) / (hi - lo)).clip(0, 1) for f, (_, lo, hi) in PROXY_TERMS.items()},
        index=X.index,
    )


def proxy_risk(X: pd.DataFrame, noise_sd: float = 0.0, seed: int | None = None) -> pd.Series:
    """The designed risk score, 0-100. With noise_sd > 0 this is the training target."""
    w = pd.Series({f: wt for f, (wt, _, _) in PROXY_TERMS.items()})
    score = 100 * proxy_components(X).mul(w).sum(axis=1)
    if noise_sd:
        score += np.random.default_rng(seed).normal(0, noise_sd, len(score))
    return score.clip(0, 100).round(1)
