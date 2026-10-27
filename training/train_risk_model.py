"""Train the MaintainAI risk model on weekly snapshots of factory_maintenance.db.

    python training/train_risk_model.py            # evaluate, then save risk_model.pkl
    python training/train_risk_model.py --no-save  # evaluate only

What this model is (and is not): there are no real failure labels, so the target is
the designed proxy formula in risk_features.py (weighted, normalised features plus
Gaussian noise). A good score here shows the pipeline can learn a known relationship
from the data it will see in the app. It says nothing about predicting real failures.

Evaluation holds out whole machines (GroupShuffleSplit): weekly snapshots of the same
machine are near-duplicates, so a random row split would leak them across train/test.
The random split is reported too, only for comparison with the earlier model.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import risk_features as rf  # noqa: E402

SEED = 42
THRESHOLDS = {"warning": 40, "critical": 70}
RESULTS = ROOT / "results" / "risk_model"


def build_dataset(db_path: Path = ROOT / "factory_maintenance.db", seed: int = SEED) -> pd.DataFrame:
    """Weekly feature snapshots + the noisy proxy target (`risk`) + the noise-free one."""
    with sqlite3.connect(db_path) as conn:
        tables = rf.load_tables(conn)
    snaps = rf.weekly_snapshots(tables)
    snaps["risk_clean"] = rf.proxy_risk(snaps[rf.FEATURES])
    snaps["risk"] = rf.proxy_risk(snaps[rf.FEATURES], noise_sd=rf.PROXY_NOISE_SD, seed=seed)
    return snaps


def make_model(seed: int = SEED) -> RandomForestRegressor:
    # picked from a small grid (min_samples_leaf 1/2/5, max_features 1.0/0.5, max_depth 8)
    # by 5-fold group CV MAE: 3.89 vs 4.16 for the defaults with min_samples_leaf=2
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                 n_jobs=-1, random_state=seed)


def metrics(y_true, y_pred) -> dict:
    y_pred = np.clip(y_pred, 0, 100)
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "R2": round(float(r2_score(y_true, y_pred)), 3),
    }


def machine_split(df: pd.DataFrame, seed: int = SEED, test_size: float = 0.2):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, groups=df["machine_id"]))
    return df.iloc[tr], df.iloc[te]


def evaluate(df: pd.DataFrame, seed: int = SEED) -> dict:
    X, y = df[rf.FEATURES], df["risk"]
    train, test = machine_split(df, seed)
    Xtr, ytr, Xte, yte = train[rf.FEATURES], train["risk"], test[rf.FEATURES], test["risk"]

    models = {
        "Mean baseline": DummyRegressor(),
        "Linear regression": LinearRegression(),
        "Random Forest": make_model(seed),
    }
    held_out = {}
    for name, m in models.items():
        held_out[name] = metrics(yte, m.fit(Xtr, ytr).predict(Xte))
    rf_model = models["Random Forest"]
    pred = np.clip(rf_model.predict(Xte), 0, 100)

    cv = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=df["machine_id"]):
        cv.append(metrics(y.iloc[te], make_model(seed).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])))
    cv_summary = {k: {"mean": round(float(np.mean([c[k] for c in cv])), 3),
                      "std": round(float(np.std([c[k] for c in cv])), 3)} for k in cv[0]}

    rtr, rte = train_test_split(df, test_size=0.2, random_state=seed)
    random_split = metrics(rte["risk"], make_model(seed).fit(rtr[rf.FEATURES], rtr["risk"]).predict(rte[rf.FEATURES]))

    # best achievable: the noise-free formula scored against the noisy target
    ceiling = metrics(yte, test["risk_clean"])

    importances = (pd.Series(rf_model.feature_importances_, index=rf.FEATURES)
                   .sort_values(ascending=False).round(4))
    return {
        "dataset": {
            "snapshots": int(len(df)), "machines": int(df["machine_id"].nunique()),
            "weeks": int(df["snapshot_date"].nunique()),
            "first_snapshot": str(df["snapshot_date"].min().date()),
            "last_snapshot": str(df["snapshot_date"].max().date()),
            "train_machines": int(train["machine_id"].nunique()),
            "test_machines": int(test["machine_id"].nunique()),
            "test_rows": int(len(test)),
        },
        "held_out_machines": held_out,
        "group_5fold_cv_random_forest": cv_summary,
        "random_row_split_random_forest": random_split,
        "noise_ceiling": ceiling,
        "feature_importance": importances.to_dict(),
        "proxy_weights": {f: w for f, (w, _, _) in rf.PROXY_TERMS.items()},
        "proxy_noise_sd": rf.PROXY_NOISE_SD,
        "_test_predictions": pd.DataFrame({"machine_id": test["machine_id"].values,
                                           "actual": yte.values, "predicted": pred.round(1)}),
    }


def train_final(df: pd.DataFrame, seed: int = SEED) -> dict:
    """Refit on every snapshot for the app; the evaluation above is what we report."""
    model = make_model(seed).fit(df[rf.FEATURES], df["risk"])
    model.n_jobs = None  # single-threaded at inference: 50 rows, no pool start-up
    return {"model": model, "features": list(rf.FEATURES), "thresholds": dict(THRESHOLDS)}


def save_results(report: dict, out_dir: Path = RESULTS) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = {k: v for k, v in report.items() if not k.startswith("_")}
    rep["scikit_learn"] = sklearn.__version__
    (out_dir / "metrics.json").write_text(json.dumps(rep, indent=2))
    report["_test_predictions"].to_csv(out_dir / "test_predictions.csv", index=False)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", type=Path, default=ROOT / "factory_maintenance.db")
    ap.add_argument("--out", type=Path, default=ROOT / "risk_model.pkl")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)

    df = build_dataset(args.db)
    report = evaluate(df)
    d, h = report["dataset"], report["held_out_machines"]
    print(f"{d['snapshots']} snapshots | {d['machines']} machines | {d['weeks']} weeks "
          f"({d['first_snapshot']} to {d['last_snapshot']}); test = {d['test_machines']} unseen machines")
    for name, m in h.items():
        print(f"  {name:<18} MAE {m['MAE']:>5}  RMSE {m['RMSE']:>5}  R2 {m['R2']:>6}")
    cv = report["group_5fold_cv_random_forest"]
    print(f"  RF 5-fold group CV  MAE {cv['MAE']['mean']} +- {cv['MAE']['std']}  R2 {cv['R2']['mean']} +- {cv['R2']['std']}")
    print(f"  RF random row split MAE {report['random_row_split_random_forest']['MAE']}  R2 {report['random_row_split_random_forest']['R2']}  (leaky; for comparison only)")
    print(f"  noise ceiling      MAE {report['noise_ceiling']['MAE']}  R2 {report['noise_ceiling']['R2']}")
    if not args.no_save:
        save_results(report)
        joblib.dump(train_final(df), args.out, compress=3)
        print(f"saved {args.out.name} ({args.out.stat().st_size / 1e6:.1f} MB) + results/risk_model/")


if __name__ == "__main__":
    main()
