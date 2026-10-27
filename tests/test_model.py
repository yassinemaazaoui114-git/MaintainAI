"""The model bundle loads cleanly and scores 0-100; features and explanations are consistent."""

import sqlite3
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest

import risk_features as rf
from conftest import REAL_DB, ROOT


@pytest.fixture(scope="module")
def bundle():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # e.g. InconsistentVersionWarning from a scikit-learn mismatch
        return joblib.load(ROOT / "risk_model.pkl")


@pytest.fixture(scope="module")
def tables():
    with sqlite3.connect(REAL_DB) as conn:
        return rf.load_tables(conn)


def test_bundle_format(bundle):
    assert set(bundle) == {"model", "features", "thresholds"}
    assert bundle["features"] == rf.FEATURES
    assert bundle["thresholds"] == {"warning": 40, "critical": 70}


def test_predicts_0_to_100_on_fleet(bundle, tables):
    X = rf.current_features(tables)[bundle["features"]]
    pred = bundle["model"].predict(X)
    assert len(pred) == 50
    assert np.all((pred >= 0) & (pred <= 100))
    assert pred.max() - pred.min() > 30  # it separates healthy from at-risk machines


def test_predicts_0_to_100_on_extremes(bundle):
    lo = {f: 0.0 for f in rf.FEATURES}
    hi = {"age_years": 40, "operating_hours": 200_000, "days_since_maintenance": 2000, "failure_count": 50,
          "total_downtime": 5000, "maintenance_frequency": 0, "temperature": 150, "vibration": 40}
    pred = bundle["model"].predict(pd.DataFrame([lo, hi])[bundle["features"]])
    assert np.all((pred >= 0) & (pred <= 100))
    assert pred[1] > pred[0]


def test_proxy_formula_weights_sum_to_one():
    assert sum(w for w, _, _ in rf.PROXY_TERMS.values()) == pytest.approx(1.0)
    assert set(rf.PROXY_TERMS) == set(rf.FEATURES)


def test_features_ignore_future_records(tables):
    as_of = pd.Timestamp("2026-03-01")
    before = rf.features_at(tables, as_of)
    future = {k: v.copy() for k, v in tables.items()}
    future["failures"] = pd.concat([future["failures"], pd.DataFrame(
        {"machine_id": ["MCH-001"], "date": [as_of + pd.Timedelta(days=1)], "downtime_hours": [99.0]})])
    pd.testing.assert_frame_equal(before, rf.features_at(future, as_of))


def test_last_snapshot_matches_live_features(tables):
    snaps = rf.weekly_snapshots(tables)
    last = snaps[snaps["snapshot_date"] == snaps["snapshot_date"].max()].drop(columns="snapshot_date")
    pd.testing.assert_frame_equal(last.reset_index(drop=True), rf.current_features(tables).reset_index(drop=True))


def test_shap_contributions_add_up_to_prediction(temp_db):
    import data as D
    m = D.load_machines()
    base, contrib = D.explain_risk(m)
    raw = D._model_bundle()["model"].predict(D._model_input(m))
    np.testing.assert_allclose(base + contrib.sum(axis=1).values, raw, atol=1e-6)
    drivers = D.top_factors(m, contrib)
    assert all(isinstance(d, list) and len(d) <= 2 for d in drivers)
