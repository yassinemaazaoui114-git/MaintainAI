"""Model inputs and labels, shared by the app and the trainer.

One module so the features the model is trained on and the features the app scores
are computed by the same code (training/train_risk_model.py, data.py, notebook 01).

Features describe a machine at the end of a day, using only records dated on or before it:

    age_years            machine age (static in the source data)
    model_no             machine model 1-4, as a number
    days_since_comp1..4  days since that component was last replaced (scheduled or after a failure)
    volt, rotate, pressure, vibration          that day's mean sensor reading
    volt_7d, rotate_7d, pressure_7d, vibration_7d   mean over the last 7 days
    errors_7d            error codes logged in the last 7 days

Label: did the machine have a component failure in the next 7 days, (d, d + 7]?

"Now" for the live app is the date of the latest record, not the wall clock: the
demo data ends on 2015-12-31, and measuring against today's date would make every
component look years overdue.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]
SENSORS = ["volt", "rotate", "pressure", "vibration"]
FEATURES = (["age_years", "model_no"] + [f"days_since_{c}" for c in COMPONENTS]
            + SENSORS + [f"{s}_7d" for s in SENSORS] + ["errors_7d"])
HORIZON_DAYS = 7   # the risk score is P(failure within this many days)
WINDOW_DAYS = 7    # sensor means and error counts
FAILURE_DAYS = 365  # "failures in the last 12 months", shown in the app (not a model input)


# ── raw data ─────────────────────────────────────────────────────────────────
def load_tables(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    q = lambda sql: pd.read_sql_query(sql, conn)
    t = {
        "machines": q("SELECT machine_id, model, age_years FROM Machines"),
        "maintenance": q("SELECT machine_id, date, component FROM Maintenance_Log"),
        "failures": q("SELECT machine_id, date, component FROM Failure_Log"),
        "errors": q("SELECT machine_id, date, error_code FROM Error_Log"),
        "sensors": q(f"SELECT machine_id, date, {', '.join(SENSORS)} FROM Sensor_Readings"),
    }
    for k in ("maintenance", "failures", "errors", "sensors"):
        t[k]["date"] = pd.to_datetime(t[k]["date"], errors="coerce")
        t[k] = t[k].dropna(subset=["date"])
    return t


def reference_date(t: dict[str, pd.DataFrame]) -> pd.Timestamp:
    """Latest dated record across the logs and sensor readings ("now" for the app)."""
    dates = [t[k]["date"].max() for k in ("maintenance", "failures", "errors", "sensors") if len(t[k])]
    dates = [d for d in dates if pd.notna(d)]
    return max(dates) if dates else pd.Timestamp.today().normalize()


def first_full_day(t: dict[str, pd.DataFrame]) -> pd.Timestamp:
    """First day with a full 7-day sensor and error window behind it."""
    return t["sensors"]["date"].min() + pd.Timedelta(days=WINDOW_DAYS - 1)


# ── features ─────────────────────────────────────────────────────────────────
def _daily_grid(ids, start, end) -> pd.DataFrame:
    days = pd.date_range(start, end, freq="D")
    return pd.MultiIndex.from_product([ids, days], names=["machine_id", "date"]).to_frame(index=False)


def daily_features(t: dict[str, pd.DataFrame], start, end) -> pd.DataFrame:
    """machine_id, date + FEATURES for every machine and every day in [start, end].

    Machines without records (e.g. one just added) still get a row: a component never
    replaced counts as old as the machine, and missing sensor values take that day's
    fleet median.
    """
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    m = t["machines"]
    grid = _daily_grid(m["machine_id"], start, end)
    out = grid.merge(m, on="machine_id", how="left")
    out["model_no"] = out["model"].astype(str).str.extract(r"(\d+)")[0].astype(float).fillna(0).astype(int)
    out["age_years"] = out["age_years"].astype(float)
    out = out.drop(columns="model").sort_values("date", kind="stable")

    # days since each component was last replaced, on or before the day
    for c in COMPONENTS:
        rep = (t["maintenance"].loc[t["maintenance"]["component"] == c, ["machine_id", "date"]]
               .rename(columns={"date": "last"}).sort_values("last"))
        out = pd.merge_asof(out, rep, left_on="date", right_on="last", by="machine_id")
        never = (out["age_years"] * 365.25).round()
        out[f"days_since_{c}"] = (out["date"] - out.pop("last")).dt.days.astype(float).fillna(never)

    # sensors and errors on a regular daily grid that reaches back one window
    back = _daily_grid(m["machine_id"], start - pd.Timedelta(days=WINDOW_DAYS - 1), end)
    s = t["sensors"]
    s = back.merge(s[s["date"] <= end], on=["machine_id", "date"], how="left")
    g = s.groupby("machine_id")[SENSORS]
    roll = g.rolling(WINDOW_DAYS, min_periods=1).mean().reset_index(level=0, drop=True)
    latest = g.ffill()  # that day's reading, else the latest one before it
    s[SENSORS] = latest
    s[[f"{x}_7d" for x in SENSORS]] = roll.reindex(s.index).fillna(latest).to_numpy()
    e = t["errors"].groupby(["machine_id", "date"]).size().rename("n").reset_index()
    s = s.merge(e, on=["machine_id", "date"], how="left").fillna({"n": 0})
    s["errors_7d"] = s.groupby("machine_id")["n"].transform(lambda x: x.rolling(WINDOW_DAYS, min_periods=1).sum())
    cols = SENSORS + [f"{x}_7d" for x in SENSORS] + ["errors_7d"]
    out = out.merge(s[["machine_id", "date"] + cols], on=["machine_id", "date"], how="left")
    for col in cols:
        out[col] = out[col].fillna(out.groupby("date")[col].transform("median"))

    out = out.sort_values(["date", "machine_id"], ignore_index=True)
    out[SENSORS + [f"{x}_7d" for x in SENSORS]] = out[SENSORS + [f"{x}_7d" for x in SENSORS]].round(2)
    return out[["machine_id", "date"] + FEATURES]


def features_at(t: dict[str, pd.DataFrame], as_of) -> pd.DataFrame:
    """One row per machine: machine_id + FEATURES at the end of `as_of`."""
    return daily_features(t, as_of, as_of).drop(columns="date")


def failures_last_12mo(t: dict[str, pd.DataFrame], as_of) -> pd.Series:
    as_of = pd.Timestamp(as_of)
    f = t["failures"]
    f = f[(f["date"] <= as_of) & (f["date"] > as_of - pd.Timedelta(days=FAILURE_DAYS))]
    return f.groupby("machine_id").size()


def current_features(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Live features as of the latest record, plus failures in the last 12 months (display only)."""
    ref = reference_date(t)
    out = features_at(t, ref)
    out["failures_12mo"] = out["machine_id"].map(failures_last_12mo(t, ref)).fillna(0).astype(int)
    return out


# ── labels ───────────────────────────────────────────────────────────────────
def failure_labels(t: dict[str, pd.DataFrame], rows: pd.DataFrame,
                   horizon: int = HORIZON_DAYS) -> np.ndarray:
    """1 where the machine has a failure in (date, date + horizon], else 0."""
    events = set(zip(t["failures"]["machine_id"], t["failures"]["date"]))
    y = np.zeros(len(rows), dtype=int)
    for k in range(1, horizon + 1):
        ahead = rows["date"] + pd.Timedelta(days=k)
        y |= np.fromiter(((m, d) in events for m, d in zip(rows["machine_id"], ahead)), bool, len(rows))
    return y


def training_rows(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every machine-day whose 7-day look-back and 7-day look-ahead are both complete."""
    end = reference_date(t) - pd.Timedelta(days=HORIZON_DAYS)
    rows = daily_features(t, first_full_day(t), end)
    rows["failure_next_7d"] = failure_labels(t, rows)
    return rows
