"""Build maintainai.db from the Microsoft Azure Predictive Maintenance sample data.

    python training/build_azure_db.py                        # reads data/azure_pdm/
    python training/build_azure_db.py --src path/to/csvs --out maintainai.db

The five raw CSVs (PdM_machines, PdM_maint, PdM_failures, PdM_errors, PdM_telemetry)
are not committed (data/ is git-ignored); only the small database this script writes is.
Source: https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance
(license not clearly stated; used here as a non-commercial, educational demo).

What changes on the way in:
- machineID 1..100 becomes MCH-001..MCH-100.
- Hourly telemetry (876,100 rows) becomes one row per machine per day: the mean volt,
  rotate, pressure and vibration of that day's readings.
- Timestamps become dates. Records after 2015-12-31 are dropped: the only ones are
  the first hours of 2016-01-01 (7 telemetry hours per machine, 2 errors, 7
  replacements), a day without a full day of readings.
- A replacement that happens on the same day as a failure of the same component on
  that machine is typed "Corrective"; every other replacement "Scheduled".
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LAST_DAY = pd.Timestamp("2015-12-31")
SENSORS = ["volt", "rotate", "pressure", "vibration"]
FILES = ["PdM_machines", "PdM_maint", "PdM_failures", "PdM_errors", "PdM_telemetry"]

SCHEMA = """
CREATE TABLE Machines (
    machine_id TEXT PRIMARY KEY,
    model      TEXT NOT NULL,
    age_years  INTEGER NOT NULL
);
CREATE TABLE Maintenance_Log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id TEXT NOT NULL REFERENCES Machines(machine_id),
    date       TEXT NOT NULL,
    component  TEXT NOT NULL,
    type       TEXT NOT NULL,          -- Scheduled | Corrective
    note       TEXT
);
CREATE TABLE Failure_Log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id TEXT NOT NULL REFERENCES Machines(machine_id),
    date       TEXT NOT NULL,
    component  TEXT NOT NULL,
    note       TEXT
);
CREATE TABLE Error_Log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id TEXT NOT NULL REFERENCES Machines(machine_id),
    date       TEXT NOT NULL,
    error_code TEXT NOT NULL
);
CREATE TABLE Sensor_Readings (     -- daily means of the hourly telemetry
    machine_id TEXT NOT NULL REFERENCES Machines(machine_id),
    date       TEXT NOT NULL,
    volt       REAL,
    rotate     REAL,
    pressure   REAL,
    vibration  REAL,
    PRIMARY KEY (machine_id, date)
);
CREATE INDEX ix_maint_machine ON Maintenance_Log(machine_id, date);
CREATE INDEX ix_fail_machine ON Failure_Log(machine_id, date);
CREATE INDEX ix_err_machine ON Error_Log(machine_id, date);
"""

# Model inputs as of the latest sensor day, for anyone querying the database directly.
# The app computes the same values in Python (risk_features.py); a test keeps them equal.
VIEW = """
CREATE VIEW Machine_Risk_Features AS
WITH asof AS (SELECT MAX(date) AS d FROM Sensor_Readings)
SELECT
    m.machine_id, m.model, m.age_years,
    {days_since},
    (SELECT COUNT(*) FROM Error_Log e WHERE e.machine_id = m.machine_id
        AND e.date > date(asof.d, '-7 day') AND e.date <= asof.d) AS errors_7d,
    {latest},
    {mean7},
    (SELECT COUNT(*) FROM Failure_Log f WHERE f.machine_id = m.machine_id
        AND f.date > date(asof.d, '-365 day') AND f.date <= asof.d) AS failures_12mo
FROM Machines m, asof
""".format(
    days_since=",\n    ".join(
        f"CAST(julianday(asof.d) - julianday((SELECT MAX(date) FROM Maintenance_Log ml "
        f"WHERE ml.machine_id = m.machine_id AND ml.component = 'comp{i}' AND ml.date <= asof.d)) "
        f"AS INTEGER) AS days_since_comp{i}" for i in range(1, 5)),
    latest=",\n    ".join(
        f"(SELECT {s} FROM Sensor_Readings r WHERE r.machine_id = m.machine_id AND r.date <= asof.d "
        f"ORDER BY r.date DESC LIMIT 1) AS {s}" for s in SENSORS),
    mean7=",\n    ".join(
        f"(SELECT AVG({s}) FROM Sensor_Readings r WHERE r.machine_id = m.machine_id "
        f"AND r.date > date(asof.d, '-7 day') AND r.date <= asof.d) AS {s}_7d" for s in SENSORS),
)


def _read(src: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(src / f"{name}.csv")
    if "datetime" in df:
        df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()
    if "machineID" in df:
        df["machine_id"] = df["machineID"].map("MCH-{:03d}".format)
    return df


def build(src: Path, out: Path) -> dict:
    missing = [f for f in FILES if not (src / f"{f}.csv").exists()]
    if missing:
        raise SystemExit(f"missing in {src}: {', '.join(missing)}")

    machines = _read(src, "PdM_machines")
    maint, fails, errs = (_read(src, n) for n in ("PdM_maint", "PdM_failures", "PdM_errors"))
    tele = _read(src, "PdM_telemetry")
    dropped = {n: int((d["date"] > LAST_DAY).sum())
               for n, d in (("maintenance", maint), ("failures", fails), ("errors", errs), ("telemetry_hours", tele))}
    maint, fails, errs, tele = (d[d["date"] <= LAST_DAY] for d in (maint, fails, errs, tele))

    failed = set(zip(fails["machine_id"], fails["date"], fails["failure"]))
    maint = maint.assign(type=[("Corrective" if k in failed else "Scheduled")
                               for k in zip(maint["machine_id"], maint["date"], maint["comp"])])
    daily = tele.groupby(["machine_id", "date"], as_index=False)[SENSORS].mean()
    daily[SENSORS] = daily[SENSORS].round(2)

    day = lambda s: s.dt.strftime("%Y-%m-%d")
    out.unlink(missing_ok=True)
    with sqlite3.connect(out) as conn:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO Machines VALUES (?,?,?)",
                         zip(machines["machine_id"], machines["model"], machines["age"].astype(int)))
        m = maint.sort_values(["date", "machine_id", "comp"])
        conn.executemany("INSERT INTO Maintenance_Log (machine_id, date, component, type) VALUES (?,?,?,?)",
                         zip(m["machine_id"], day(m["date"]), m["comp"], m["type"]))
        f = fails.sort_values(["date", "machine_id", "failure"])
        conn.executemany("INSERT INTO Failure_Log (machine_id, date, component) VALUES (?,?,?)",
                         zip(f["machine_id"], day(f["date"]), f["failure"]))
        e = errs.sort_values(["datetime", "machine_id"])
        conn.executemany("INSERT INTO Error_Log (machine_id, date, error_code) VALUES (?,?,?)",
                         zip(e["machine_id"], day(e["date"]), e["errorID"]))
        conn.executemany("INSERT INTO Sensor_Readings VALUES (?,?,?,?,?,?)",
                         zip(daily["machine_id"], day(daily["date"]),
                             *(daily[s] for s in SENSORS)))
        conn.executescript(VIEW)
    with sqlite3.connect(out) as conn:
        conn.execute("VACUUM")
    return {
        "machines": len(machines), "maintenance": len(maint),
        "corrective": int((maint["type"] == "Corrective").sum()), "failures": len(fails),
        "errors": len(errs), "sensor_days": len(daily), "dropped_after_2015": dropped,
        "size_mb": round(out.stat().st_size / 1e6, 2),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "azure_pdm")
    ap.add_argument("--out", type=Path, default=ROOT / "maintainai.db")
    args = ap.parse_args(argv)
    print(build(args.src, args.out))


if __name__ == "__main__":
    main()
