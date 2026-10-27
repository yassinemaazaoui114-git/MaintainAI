"""Industry-styled predictive-maintenance dashboard on Streamlit.

Run:  streamlit run app.py
Layout: display surfaces are rendered as HTML embeds (ui.py); every input is a
native Streamlit widget, so the Python model stays in charge.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import data as D
import theme_css
import ui
from db_utils import next_machine_id

st.set_page_config(page_title="Predictive Maintenance", layout="wide",
                   initial_sidebar_state="expanded")
theme_css.inject()


# ── state ────────────────────────────────────────────────────────────────────
def _init():
    if "machines" not in st.session_state:
        st.session_state.machines = D.load_machines()
        st.session_state.maintenance = D.load_maintenance()
        st.session_state.failures = D.load_failures()


_init()
M: pd.DataFrame = st.session_state.machines
MAINT: pd.DataFrame = st.session_state.maintenance
FAIL: pd.DataFrame = st.session_state.failures

# ── sidebar ──────────────────────────────────────────────────────────────────
WARN, CRIT = D.thresholds()  # stored with the model: Warning 40, Critical 70

_SCREENS = ["Dashboard", "Machines", "Maintenance Log", "Failure Log", "Benchmarks"]
RESULTS = Path(__file__).resolve().parent / "results"
_BENCH = ["ai4i2020", "cmapss_fd001"]
_counts = {
    "Dashboard": len(M),
    "Machines": len(M),
    "Maintenance Log": len(MAINT),
    "Failure Log": len(FAIL),
    "Benchmarks": sum((RESULTS / "benchmarks" / b / "metrics.json").exists() for b in _BENCH),
}

with st.sidebar:
    st.markdown(
        "<div class='sb-head'>"
        "<div class='sb-title'>MaintainAI</div>"
        "<div class='sb-sub'>Predictive Maintenance</div>"
        "</div>"
        "<div class='sb-seclabel'>Monitor</div>",
        unsafe_allow_html=True)
    # the bold markdown count is pushed flush right by theme_css; the key keeps the
    # selection when a count changes (otherwise saving a record jumped back to Dashboard)
    screen = st.radio("Monitor", _SCREENS, format_func=lambda s: f"{s} **{_counts[s]}**",
                      label_visibility="collapsed", key="screen")
    _as_of = M.attrs.get("as_of")
    _last = _as_of.strftime("%d %b %Y") if _as_of is not None else "—"
    st.markdown(
        "<div class='sb-info'>"
        "<div class='row'><span class='k'>Model</span><span class='v'>Random Forest Regressor</span></div>"
        f"<div class='row'><span class='k'>Data as of</span><span class='v'>{_last}</span></div>"
        f"<div class='row'><span class='k'>Total records</span><span class='v'>{len(M)+len(MAINT)+len(FAIL)}</span></div>"
        "</div>",
        unsafe_allow_html=True)

# ── scoring ──────────────────────────────────────────────────────────────────
M = M.copy()
M["risk"] = D.score_risk(M)
M["status"] = [ui.status_of(r, WARN, CRIT)[0] for r in M["risk"]]
BASE, CONTRIB = D.explain_risk(M)
M["drivers"] = D.top_factors(M, CONTRIB)


# ── form feedback ────────────────────────────────────────────────────────────
# st.rerun() discards anything drawn in the current run, so a success message has
# to be carried over to the next run to be seen at all.
def _flash(msg: str):
    st.session_state.flash = msg
    st.rerun()


def _show_flash():
    if msg := st.session_state.pop("flash", None):
        st.success(msg)


# ── screens ──────────────────────────────────────────────────────────────────
def dashboard():
    counts = M["status"].value_counts()
    st.iframe(
        ui.metric_row(len(M), int(counts.get("Normal", 0)),
                      int(counts.get("Warning", 0)), int(counts.get("Critical", 0))),
        height="content")

    crit_rows = (M[M["status"] == "Critical"].sort_values("risk", ascending=False)
                 .rename(columns={"machine_id": "id", "machine_type": "type"})
                 [["id", "type", "risk"]].to_dict("records"))
    st.iframe(ui.alarm_banner(crit_rows), height="content")

    left, right = st.columns([1.3, 1])
    with left:
        st.iframe(ui.risk_chart(M, warn=WARN, crit=CRIT), height="content")
    with right:
        labels, counts_m = D.monthly_failures(FAIL)
        st.iframe(ui.month_chart(labels, counts_m), height="content")

    st.iframe(ui.health_table(M, warn=WARN, crit=CRIT),
                    height="content")

    ranked = M.sort_values("risk", ascending=False)
    pick, _ = st.columns([1, 2.2])
    mid = pick.selectbox("Explain a machine's score", ranked["machine_id"].tolist(),
                         format_func=lambda m: f"{m} · risk {M.set_index('machine_id').at[m, 'risk']:.0f}")
    i = M.index[M["machine_id"] == mid][0]
    X = M.rename(columns={"failures": "failure_count", "downtime": "total_downtime"})
    factors = [(D.factor_text(f, X.at[i, f], c), c) for f, c in CONTRIB.loc[i].items()]
    st.iframe(ui.explain_panel(mid, M.at[i, "risk"], BASE, factors, warn=WARN, crit=CRIT),
              height="content")


def machines():
    tbl, form = st.columns([1.6, 1])
    with tbl:
        st.iframe(
            ui.data_table(
                M[["machine_id", "machine_type", "location", "install_date", "operating_hours", "status"]]
                 .assign(operating_hours=lambda d: d["operating_hours"].map("{:,.0f}".format)),
                "Asset register",
                ["ID", "Type", "Location", "Install date", "Operating hours", "Status"],
                right=[4]),
            height="content")
    with form:
        st.markdown("#### Add machine")
        _show_flash()
        with st.form("add_machine", clear_on_submit=True):
            a, b = st.columns(2)
            mid = a.text_input("Machine ID", value=next_machine_id())
            mtype = b.text_input("Type", placeholder="Centrifugal Pump")
            loc = a.text_input("Location", placeholder="Line A")
            inst = b.date_input("Install date")
            hours = st.number_input("Operating hours (meter reading)", min_value=0, step=100)
            if st.form_submit_button("Add machine"):
                if not mid.strip():
                    st.error("Machine ID is required.")
                elif mid in M["machine_id"].values:
                    st.error(f"Machine ID '{mid}' already exists — choose a different ID.")
                else:
                    D.add_machine(mid, mtype or "Unspecified", loc or "—", inst, hours)
                    st.session_state.machines = D.load_machines()
                    _flash(f"{mid} added to the register.")


def maintenance_log():
    tbl, form = st.columns([1.6, 1])
    with tbl:
        f1, f2 = st.columns(2)
        mach = f1.selectbox("Machine", ["All machines"] + M["machine_id"].tolist())
        typ = f2.selectbox("Type", ["All types"] + sorted(MAINT["type"].unique()))
        d = MAINT
        if mach != "All machines":
            d = d[d["machine_id"] == mach]
        if typ != "All types":
            d = d[d["type"] == typ]
        st.iframe(
            ui.data_table(d, "Maintenance records",
                          ["Machine", "Date", "Type", "Reason", "Duration (h)"],
                          right=[4], chip_cols=[2],
                          footnote=f"{len(d)} records · {d['duration'].sum():.1f} h maintenance time"),
            height="content")
    with form:
        st.markdown("#### Log maintenance")
        _show_flash()
        with st.form("log_maint", clear_on_submit=True):
            a, b = st.columns(2)
            mid = a.selectbox("Machine", M["machine_id"].tolist())
            when = b.date_input("Date")
            typ2 = a.selectbox("Type", ["Preventive", "Corrective", "Inspection", "Calibration"])
            dur = b.number_input("Duration (h)", min_value=0.0, step=0.5)
            reason = st.text_area("Reason", placeholder="Bearing replacement following vibration alert")
            if st.form_submit_button("Log maintenance"):
                if not reason.strip():
                    st.error("Add a reason before logging.")
                else:
                    D.add_maintenance(mid, when, typ2, reason, dur)
                    st.session_state.maintenance = D.load_maintenance()
                    st.session_state.machines = D.load_machines()  # features (and risk) changed
                    _flash(f"Record logged for {mid}.")


def failure_log():
    tbl, form = st.columns([1.6, 1])
    with tbl:
        f1, f2 = st.columns(2)
        mach = f1.selectbox("Machine", ["All machines"] + M["machine_id"].tolist(), key="fm")
        typ = f2.selectbox("Failure type", ["All types"] + sorted(FAIL["failure_type"].unique()), key="ft")
        d = FAIL
        if mach != "All machines":
            d = d[d["machine_id"] == mach]
        if typ != "All types":
            d = d[d["failure_type"] == typ]
        st.iframe(
            ui.data_table(d, "Failure records",
                          ["Machine", "Date", "Failure type", "Cause", "Downtime (h)", "Repair (h)"],
                          right=[4, 5], alarm_cols=[2],
                          footnote=f"{len(d)} records · {d['downtime'].sum():.1f} h downtime · {d['repair_time'].sum():.1f} h wrench time"),
            height="content")
    with form:
        st.markdown("#### Log failure")
        _show_flash()
        with st.form("log_fail", clear_on_submit=True):
            a, b = st.columns(2)
            mid = a.selectbox("Machine", M["machine_id"].tolist())
            when = b.date_input("Date")
            ftype = a.selectbox("Failure type",
                                ["Mechanical", "Electrical", "Thermal", "Hydraulic", "Control"])
            down = b.number_input("Downtime (h)", min_value=0.0, step=0.5)
            rep = a.number_input("Repair time (h)", min_value=0.0, step=0.5)
            cause = b.text_input("Cause", placeholder="Seal wear")
            if st.form_submit_button("Log failure"):
                if not cause.strip():
                    st.error("Add a cause before logging.")
                else:
                    D.add_failure(mid, when, ftype, cause, down, rep)
                    st.session_state.failures = D.load_failures()
                    st.session_state.machines = D.load_machines()  # features (and risk) changed
                    _flash(f"Failure logged for {mid}.")


def _metrics(*parts: str) -> dict | None:
    p = RESULTS.joinpath(*parts, "metrics.json")
    return json.loads(p.read_text()) if p.exists() else None


def _figure(*parts: str, caption: str = ""):
    p = RESULTS.joinpath(*parts)
    if p.exists():
        st.image(str(p), caption=caption or None, width="stretch")


def benchmarks():
    st.iframe(ui.notice(
        "Simulated benchmark datasets",
        "Both benchmarks below use public datasets that were <b>generated by simulators</b> "
        "(AI4I 2020 by documented failure rules, C-MAPSS by NASA's engine simulation), not recorded "
        "from real factories or aircraft. They show that the modelling workflow holds up on standard "
        "predictive-maintenance problems. They are not a claim about real-world accuracy. The in-app risk score "
        "is a separate model, trained on this app's own synthetic data (see the bottom of this page)."),
        height="content")

    ai = _metrics("benchmarks", "ai4i2020")
    st.markdown("#### AI4I 2020 · machine-failure classification")
    if ai:
        sel = ai["selected"]
        st.caption(f"{ai['rows']:,} production runs, {ai['failure_rate']:.1%} failures · test set: "
                   f"{ai['test_rows']:,} runs, {ai['test_failures']} failures · failure-mode columns excluded (no leakage) · "
                   f"class weights + F1-tuned threshold · selected on training CV: **{sel['model']}, {sel['features']} features**")
        t = pd.DataFrame(ai["results"])
        t = t[t["model"] != "Baseline (prior)"].copy()
        num = ["pr_auc", "roc_auc", "f1", "precision", "recall"]
        t[num] = t[num].map("{:.3f}".format)
        st.iframe(ui.data_table(
            t[["model", "features", "pr_auc", "roc_auc", "f1", "precision", "recall"]],
            "Test-set results", ["Model", "Features", "PR-AUC", "ROC-AUC", "F1", "Precision", "Recall"],
            right=[2, 3, 4, 5, 6],
            footnote=f"Baseline PR-AUC (always predict the failure rate) = {ai['failure_rate']:.3f}"),
            height="content")
        a, b = st.columns([2.2, 1])
        with a:
            _figure("benchmarks", "ai4i2020", "pr_curves.png")
        with b:
            cm = sel["confusion_matrix"]
            _figure("benchmarks", "ai4i2020", "confusion_matrix.png",
                    caption=f"{cm['tp']} of {cm['tp'] + cm['fn']} failures caught, {cm['fp']} false alarms")
        st.caption(f"Citation: {ai['citation']} · [dataset]({ai['url']})")
    else:
        st.info("Run notebooks/02_benchmark_ai4i2020.ipynb to produce these results.")

    cm_ = _metrics("benchmarks", "cmapss_fd001")
    st.markdown("#### NASA C-MAPSS FD001 · remaining useful life")
    if cm_:
        st.caption(f"{cm_['train_engines']} training engines run to failure · {cm_['test_engines']} test engines "
                   f"(official split + RUL file) · RUL capped at {cm_['rul_cap']} · {cm_['window']}-cycle rolling "
                   f"sensor features · selected on training CV: **{cm_['selected']}** · lower is better for both metrics")
        t = pd.DataFrame(cm_["results"])
        t[["test_rmse", "test_rmse_capped_truth"]] = t[["test_rmse", "test_rmse_capped_truth"]].map("{:.2f}".format)
        t["nasa_score"] = t["nasa_score"].map("{:,.0f}".format)
        st.iframe(ui.data_table(
            t[["model", "test_rmse", "test_rmse_capped_truth", "nasa_score"]],
            "Test-set results (last observed cycle of each test engine)",
            ["Model", "RMSE (cycles)", "RMSE vs capped truth", "NASA score"], right=[1, 2, 3]),
            height="content")
        _figure("benchmarks", "cmapss_fd001", "rul_predictions.png")
        st.caption(f"Citation: {cm_['citation']}")
    else:
        st.info("Run notebooks/03_benchmark_nasa_cmapss.ipynb to produce these results.")

    rm = _metrics("risk_model")
    st.markdown("#### In-app risk model · synthetic data, proxy target")
    if rm:
        h, cv = rm["held_out_machines"], rm["group_5fold_cv_random_forest"]
        rows = [(n, m["MAE"], m["RMSE"], m["R2"]) for n, m in h.items()]
        rows.append(("Random Forest · 5-fold group CV", cv["MAE"]["mean"], cv["RMSE"]["mean"], cv["R2"]["mean"]))
        rows.append(("Noise ceiling (the formula itself)", *rm["noise_ceiling"].values()))
        st.caption(f"{rm['dataset']['snapshots']:,} weekly snapshots of {rm['dataset']['machines']} synthetic machines · "
                   f"evaluated on machines the model never saw. The target is a designed formula, so these numbers "
                   f"show the pipeline works, not that it predicts real failures.")
        rows = [(n, f"{mae:.2f}", f"{rmse:.2f}", f"{r2:.3f}") for n, mae, rmse, r2 in rows]
        st.iframe(ui.data_table(pd.DataFrame(rows), "Held-out machines", ["Model", "MAE", "RMSE", "R²"],
                                right=[1, 2, 3]), height="content")
    else:
        st.info("Run training/train_risk_model.py to produce these results.")


_SUBTITLES = {
    "Dashboard": "Fleet health overview",
    "Machines": "Asset register",
    "Maintenance Log": "Service history",
    "Failure Log": "Breakdown history",
    "Benchmarks": "Model results on public datasets",
}
st.markdown(
    f"<div style='display:flex;align-items:baseline;gap:14px;margin-bottom:2px'>"
    f"<span style=\"font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:34px;"
    f"letter-spacing:.02em;text-transform:uppercase\">{screen}</span>"
    f"<span style='font-size:14px;color:#6b6f73'>{_SUBTITLES[screen]}</span></div>",
    unsafe_allow_html=True)
{"Dashboard": dashboard, "Machines": machines,
 "Maintenance Log": maintenance_log, "Failure Log": failure_log, "Benchmarks": benchmarks}[screen]()

st.iframe(ui.status_strip([
    ("Historian", "OK"), ("Model", "Random Forest Risk Regressor"),
    ("Records", str(len(M) + len(MAINT) + len(FAIL))),
]), height="content")
