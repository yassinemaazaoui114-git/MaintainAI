"""Train the MaintainAI risk model: P(component failure in the next 7 days).

    python training/train_risk_model.py            # evaluate, then save risk_model.pkl
    python training/train_risk_model.py --no-save  # evaluate only

Data: maintainai.db, built from the Microsoft Azure Predictive Maintenance sample data
by training/build_azure_db.py. The failure events are real records in that public
dataset (which Microsoft generated); they were not designed by this project.

One row per machine per day (features from risk_features.py), labelled 1 when the
machine has a failure in the next 7 days. The split is by time, so the model is always
tested on months it has not seen:

    train       2015-01-07 .. 2015-08-24   fit candidate models
    validation  2015-09-01 .. 2015-10-31   pick the model, check calibration, set thresholds
    test        2015-11-01 .. 2015-12-24   touched once, by the model refit on train + validation

Rows in the 7 days before each boundary are dropped from the earlier block, because
their labels look into the next block. The model the app uses is refit on every
labelled day; the numbers reported are the test numbers above.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import risk_features as rf  # noqa: E402

SEED = 42
LABEL = "failure_next_7d"
VAL_START, TEST_START = pd.Timestamp("2015-09-01"), pd.Timestamp("2015-11-01")
CRITICAL = 50          # "more likely than not to fail within 7 days"
WARNING_RECALL = 0.80  # Warning = highest threshold that still flags 80% of pre-failure days (validation)
MAX_ECE = 0.05         # calibrate only if validation probabilities are off by more than 5 points
RESULTS = ROOT / "results" / "risk_model"

# small grids, judged on validation PR-AUC only. The forest has 100 trees: 300 scored the
# same on validation (0.637 vs 0.632 with leaf 20) but made SHAP for the fleet 5x slower,
# and the app explains every machine on each data change.
CANDIDATES = {
    "Baseline (failure rate)": [lambda: DummyClassifier(strategy="prior")],
    "Logistic regression": [lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))],
    "Random Forest": [
        (lambda leaf=leaf, mf=mf: RandomForestClassifier(
            n_estimators=100, min_samples_leaf=leaf, max_features=mf, n_jobs=-1, random_state=SEED))
        for leaf in (10, 20, 40) for mf in (0.33, 0.5)],
    "Gradient boosting": [
        (lambda lr=lr, nodes=nodes: HistGradientBoostingClassifier(
            learning_rate=lr, max_leaf_nodes=nodes, max_iter=300, early_stopping=False, random_state=SEED))
        for lr in (0.05, 0.1) for nodes in (15, 31)],
}


def build_dataset(db_path: Path = ROOT / "maintainai.db") -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return rf.training_rows(rf.load_tables(conn))


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    gap = pd.Timedelta(days=rf.HORIZON_DAYS + 1)  # a row's label looks 7 days ahead
    return {
        "train": df[df["date"] <= VAL_START - gap],
        "val": df[(df["date"] >= VAL_START) & (df["date"] < TEST_START)],
        "train+val": df[df["date"] <= TEST_START - gap],
        "test": df[df["date"] >= TEST_START],
    }


def proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error: average gap between predicted and observed rates."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def at_threshold(y, p, threshold: float) -> dict:
    flag = p * 100 >= threshold
    tp = int((flag & (y == 1)).sum())
    return {"threshold": threshold, "flagged": int(flag.sum()), "true_alarms": tp,
            "false_alarms": int(flag.sum()) - tp,
            "precision": round(tp / flag.sum(), 3) if flag.any() else None,
            "recall": round(tp / (y == 1).sum(), 3)}


def scores(y, p, thresholds: dict) -> dict:
    out = {"pr_auc": round(float(average_precision_score(y, p)), 3),
           "roc_auc": round(float(roc_auc_score(y, p)), 3) if p.std() > 0 else 0.5,
           "brier": round(float(brier_score_loss(y, p)), 4)}
    for name, th in thresholds.items():
        a = at_threshold(y, p, th)
        out[f"precision_{name}"], out[f"recall_{name}"] = a["precision"], a["recall"]
    return out


def failure_events_caught(t, test: pd.DataFrame, p: np.ndarray, threshold: float) -> dict:
    """Failure events (machine-days with a failure) after the test start, and how many had
    an alert at or above `threshold` on at least one of the 7 days before."""
    flagged = set(zip(test.loc[p * 100 >= threshold, "machine_id"], test.loc[p * 100 >= threshold, "date"]))
    last = test["date"].max() + pd.Timedelta(days=rf.HORIZON_DAYS)
    f = t["failures"]
    ev = f[(f["date"] > TEST_START) & (f["date"] <= last)][["machine_id", "date"]].drop_duplicates()
    # lead time = days between the earliest alert in that window and the failure
    lead = [max((k for k in range(1, rf.HORIZON_DAYS + 1) if (m, d - pd.Timedelta(days=k)) in flagged), default=0)
            for m, d in zip(ev["machine_id"], ev["date"])]
    lead = np.array(lead)
    return {"events": int(len(ev)), "caught": int((lead > 0).sum()),
            "lead_days_median": float(np.median(lead[lead > 0])) if (lead > 0).any() else None,
            "caught_2plus_days_ahead": int((lead >= 2).sum()),
            "first": str(ev["date"].min().date()), "last": str(ev["date"].max().date())}


def evaluate(df: pd.DataFrame, tables) -> dict:
    s = split(df)
    X = {k: v[rf.FEATURES] for k, v in s.items()}
    y = {k: v[LABEL].to_numpy() for k, v in s.items()}

    # 1. pick the best setting per family, then the best family, on validation PR-AUC
    best = {}
    for family, makers in CANDIDATES.items():
        for make in makers:
            m = make().fit(X["train"], y["train"])
            ap = average_precision_score(y["val"], proba(m, X["val"]))
            if family not in best or ap > best[family][1]:
                best[family] = (make, ap, m)
    selected = max((f for f in best if not f.startswith("Baseline")), key=lambda f: best[f][1])
    make, _, sel_train_only = best[selected]
    p_val = proba(sel_train_only, X["val"])

    # 2. calibration and thresholds, from validation only
    val_ece = ece(y["val"], p_val)
    warning = max(t for t in range(5, CRITICAL, 5) if at_threshold(y["val"], p_val, t)["recall"] >= WARNING_RECALL)
    thresholds = {"warning": warning, "critical": CRITICAL}

    # 3. test once: every family's chosen setting, refit on train + validation
    results, p_test = [], {}
    for family, (mk, ap_val, _) in best.items():
        m = mk().fit(X["train+val"], y["train+val"])
        p_test[family] = proba(m, X["test"])
        results.append({"model": family, "val_pr_auc": round(float(ap_val), 3),
                        **scores(y["test"], p_test[family], thresholds)})
        if family == selected:
            params = {k: v for k, v in m.get_params().items() if v is not None and k in
                      ("n_estimators", "min_samples_leaf", "max_features", "learning_rate", "max_leaf_nodes")}
    p = p_test[selected]

    # calibration check on test, for information: isotonic fitted on validation predictions
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_val, y["val"])
    p_tr_only = proba(sel_train_only, X["test"])
    rel = pd.DataFrame({"p": p, "y": y["test"]})
    rel["bin"] = np.minimum((rel["p"] * 10).astype(int), 9)
    reliability = (rel.groupby("bin").agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
                   .round(3).reset_index().to_dict("records"))

    test = s["test"]
    return {
        "target": f"failure of any component within {rf.HORIZON_DAYS} days (machine-day rows)",
        "dataset": {
            "rows": int(len(df)), "machines": int(df["machine_id"].nunique()),
            "first_day": str(df["date"].min().date()), "last_day": str(df["date"].max().date()),
            "positive_rate": round(float(df[LABEL].mean()), 3),
            "failures": int(len(tables["failures"])),
        },
        "split": {k: {"rows": int(len(v)), "first_day": str(v["date"].min().date()),
                      "last_day": str(v["date"].max().date()), "positive_rate": round(float(v[LABEL].mean()), 3)}
                  for k, v in s.items()},
        "selected": {"model": selected, "params": params, "selection": "validation PR-AUC"},
        "results": results,
        "thresholds": thresholds,
        "threshold_choice": {
            "critical": f"{CRITICAL}%: the model thinks a failure in the next 7 days is more likely than not",
            "warning": f"highest multiple of 5 below Critical that still flags {WARNING_RECALL:.0%} of "
                       f"the days before a failure (validation)",
            "validation": {name: at_threshold(y["val"], p_val, th) for name, th in thresholds.items()},
        },
        "test_alerts": {name: at_threshold(y["test"], p, th) for name, th in thresholds.items()},
        "test_failure_events": {name: failure_events_caught(tables, test, p, th) for name, th in thresholds.items()},
        "calibration": {
            "validation_ece": round(val_ece, 3), "max_ece": MAX_ECE,
            "calibrated": val_ece > MAX_ECE,
            "test_brier_raw_train_only": round(float(brier_score_loss(y["test"], p_tr_only)), 4),
            "test_brier_isotonic_train_only": round(float(brier_score_loss(y["test"], iso.predict(p_tr_only))), 4),
            "test_ece": round(ece(y["test"], p), 3),
            "test_reliability": reliability,
        },
        "_make_selected": make,
        "_p_test": p_test, "_y_test": y["test"], "_p_val": p_val, "_y_val": y["val"],
        "_test_predictions": pd.DataFrame({"machine_id": test["machine_id"].values,
                                           "date": test["date"].dt.strftime("%Y-%m-%d").values,
                                           LABEL: y["test"], "risk": (p * 100).round(1)}),
    }


def train_final(df: pd.DataFrame, report: dict) -> dict:
    """Refit the selected setting on every labelled day for the app."""
    if report["calibration"]["calibrated"]:
        raise SystemExit("validation ECE above the limit: add a calibration step before saving")
    model = report["_make_selected"]().fit(df[rf.FEATURES], df[LABEL])
    if hasattr(model, "n_jobs"):
        model.n_jobs = None  # single-threaded at inference: 100 rows, no pool start-up
    return {"model": model, "features": list(rf.FEATURES), "thresholds": dict(report["thresholds"])}


def save_results(report: dict, df: pd.DataFrame, out_dir: Path = RESULTS) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = {k: v for k, v in report.items() if not k.startswith("_")}
    rep["final_model"] = {"trained_on": date.today().isoformat(), "rows": int(len(df)),
                          "first_day": str(df["date"].min().date()), "last_day": str(df["date"].max().date())}
    rep["scikit_learn"] = sklearn.__version__
    (out_dir / "metrics.json").write_text(json.dumps(rep, indent=2))
    report["_test_predictions"].to_csv(out_dir / "test_predictions.csv", index=False)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", type=Path, default=ROOT / "maintainai.db")
    ap.add_argument("--out", type=Path, default=ROOT / "risk_model.pkl")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)

    df = build_dataset(args.db)
    with sqlite3.connect(args.db) as conn:
        tables = rf.load_tables(conn)
    report = evaluate(df, tables)
    d, sp = report["dataset"], report["split"]
    print(f"{d['rows']} machine-days | {d['machines']} machines | {d['first_day']}..{d['last_day']} | "
          f"positive rate {d['positive_rate']}")
    print("  " + " | ".join(f"{k} {v['rows']} ({v['first_day']}..{v['last_day']})" for k, v in sp.items()))
    print(f"  selected: {report['selected']['model']} {report['selected']['params']}")
    for r in report["results"]:
        print(f"  {r['model']:<24} val PR-AUC {r['val_pr_auc']:.3f} | test PR-AUC {r['pr_auc']:.3f} "
              f"ROC {r['roc_auc']:.3f} Brier {r['brier']:.4f} | crit P {r['precision_critical']} R {r['recall_critical']}")
    th, c = report["thresholds"], report["calibration"]
    print(f"  thresholds {th} | val ECE {c['validation_ece']} (calibrated: {c['calibrated']}) | "
          f"test Brier raw {c['test_brier_raw_train_only']} vs isotonic {c['test_brier_isotonic_train_only']}")
    for name in ("critical", "warning"):
        a, e = report["test_alerts"][name], report["test_failure_events"][name]
        print(f"  test {name:>8}: {e['caught']}/{e['events']} failure events alerted in the 7 days before "
              f"({e['caught_2plus_days_ahead']} at least 2 days ahead, median lead {e['lead_days_median']} d) | "
              f"{a['flagged']} machine-days flagged, {a['false_alarms']} false alarms (precision {a['precision']})")
    if not args.no_save:
        save_results(report, df)
        joblib.dump(train_final(df, report), args.out, compress=3)
        print(f"saved {args.out.name} ({args.out.stat().st_size / 1e6:.1f} MB) + results/risk_model/")


if __name__ == "__main__":
    main()
