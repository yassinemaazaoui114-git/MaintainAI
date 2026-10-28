"""The model bundle loads cleanly and scores 0-100; features, labels, explanations and the
what-if are consistent and never use the future."""

import sqlite3
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest

import risk_features as rf
from conftest import REAL_DB, ROOT

sys.path.insert(0, str(ROOT / "training"))
import train_risk_model as T  # noqa: E402


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
    t = bundle["thresholds"]
    assert set(t) == {"warning", "critical"} and 0 < t["warning"] < t["critical"] < 100


def test_database_matches_the_source(tables):
    assert len(tables["machines"]) == 100
    assert len(tables["failures"]) == 761
    assert tables["sensors"]["date"].min() == pd.Timestamp("2015-01-01")
    assert rf.reference_date(tables) == pd.Timestamp("2015-12-31")
    assert len(tables["sensors"]) == 100 * 365  # one row per machine per day, not hourly


def test_predicts_probabilities_on_fleet(bundle, tables):
    X = rf.current_features(tables)[bundle["features"]]
    p = bundle["model"].predict_proba(X)[:, 1]
    assert len(p) == 100
    assert np.all((p >= 0) & (p <= 1))
    assert p.max() - p.min() > 0.3  # it separates healthy from at-risk machines


def test_features_ignore_future_records(tables):
    as_of = pd.Timestamp("2015-06-30")
    before = rf.features_at(tables, as_of)
    future = {k: v.copy() for k, v in tables.items()}
    nxt = as_of + pd.Timedelta(days=1)
    future["maintenance"] = pd.concat([future["maintenance"], pd.DataFrame(
        {"machine_id": ["MCH-001"], "date": [nxt], "component": ["comp1"]})])
    future["errors"] = pd.concat([future["errors"], pd.DataFrame(
        {"machine_id": ["MCH-001"], "date": [nxt], "error_code": ["error1"]})])
    future["sensors"] = future["sensors"].assign(
        vibration=np.where(future["sensors"]["date"] > as_of, 999.0, future["sensors"]["vibration"]))
    pd.testing.assert_frame_equal(before, rf.features_at(future, as_of))


def test_daily_rows_match_single_day_features(tables):
    d = pd.Timestamp("2015-10-15")
    rows = rf.daily_features(tables, d - pd.Timedelta(days=3), d)
    one = rows[rows["date"] == d].drop(columns="date").reset_index(drop=True)
    pd.testing.assert_frame_equal(one, rf.features_at(tables, d).reset_index(drop=True))


def test_labels_look_7_days_ahead():
    t = {"failures": pd.DataFrame({"machine_id": ["M"], "date": [pd.Timestamp("2015-06-10")]})}
    rows = pd.DataFrame({"machine_id": "M", "date": pd.to_datetime(
        ["2015-06-02", "2015-06-03", "2015-06-09", "2015-06-10"])})
    # 8 days before: no; 7 and 1 day before: yes; the failure day itself looks past it
    assert rf.failure_labels(t, rows).tolist() == [0, 1, 1, 0]


def test_time_split_has_no_overlap():
    df = pd.DataFrame({"date": pd.date_range("2015-01-07", "2015-12-24"), "machine_id": "M"})
    s = T.split(df)
    gap = pd.Timedelta(days=rf.HORIZON_DAYS)
    assert s["train"]["date"].max() + gap < s["val"]["date"].min()
    assert s["train+val"]["date"].max() + gap < s["test"]["date"].min()


def test_sql_view_matches_python_features(tables):
    with sqlite3.connect(REAL_DB) as conn:
        view = pd.read_sql_query("SELECT * FROM Machine_Risk_Features", conn).set_index("machine_id")
    py = rf.current_features(tables).set_index("machine_id")
    cols = [f for f in rf.FEATURES if f != "model_no"] + ["failures_12mo"]
    np.testing.assert_allclose(py[cols].astype(float), view.loc[py.index, cols].astype(float), atol=0.01)


def test_shap_contributions_add_up_to_risk(temp_db):
    import data as D
    m = D.load_machines()
    base, contrib = D.explain_risk(m)
    np.testing.assert_allclose(base + contrib.sum(axis=1).values, D.score_risk(m).values, atol=0.06)
    drivers = D.top_factors(m, contrib)
    assert all(isinstance(d, list) and len(d) <= 2 for d in drivers)
    assert D.global_importance(contrib).index[0] in rf.FEATURES


def test_what_if_only_changes_the_chosen_inputs(temp_db):
    import data as D
    m = D.load_machines()
    m["risk"] = D.score_risk(m)
    mid = m.sort_values("risk").iloc[-1]["machine_id"]
    w = D.what_if(m, mid, replace=["comp2"])
    changed = [f for f in rf.FEATURES if w.at["now", f] != w.at["after", f]]
    assert changed in ([], ["days_since_comp2"]) and w.at["after", "days_since_comp2"] == 0
    full = D.score_risk(D.what_if(m, mid, replace=rf.COMPONENTS, sensors_normal=True, clear_errors=True))
    assert full["after"] < full["now"]


def test_what_if_compares_against_the_whole_fleet(temp_db):
    import data as D
    m = D.load_machines()
    w = D.what_if(m, m["machine_id"].iloc[0], sensors_normal=True)
    _, contrib = D.explain_risk(w)
    words = [t for t, _ in D.explain_factors(w, contrib, "after", D.fleet_medians(m))]
    sensor_words = [t for t in words if any(s in t for s in rf.SENSORS)]
    assert sensor_words and all(t.startswith("normal ") for t in sensor_words)
