"""Industry-styled predictive-maintenance dashboard on Streamlit.

Run:  streamlit run app.py
Layout: display surfaces are rendered as HTML embeds (ui.py); every input is a
native Streamlit widget, so the Python model stays in charge.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

import data as D
import risk_features as rf
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
AS_OF: date = M.attrs["as_of"].date()  # "now" for the app: the latest record (31 Dec 2015)
FIRST_DAY = date(2014, 1, 1)           # the oldest records are replacements from mid-2014

# ── sidebar ──────────────────────────────────────────────────────────────────
WARN, CRIT = D.thresholds()  # stored with the model: Warning 20, Critical 50

_SCREENS = ["Dashboard", "Machines", "Maintenance Log", "Failure Log", "Model"]
RESULTS = Path(__file__).resolve().parent / "results"
_counts = {
    "Dashboard": len(M),
    "Machines": len(M),
    "Maintenance Log": len(MAINT),
    "Failure Log": len(FAIL),
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
    screen = st.radio("Monitor", _SCREENS,
                      format_func=lambda s: f"{s} **{_counts[s]}**" if s in _counts else s,
                      label_visibility="collapsed", key="screen")
    st.markdown(
        "<div class='sb-info'>"
        f"<div class='row'><span class='k'>Model</span><span class='v'>{D.model_name()}</span></div>"
        f"<div class='row'><span class='k'>Data as of</span><span class='v'>{AS_OF:%d %b %Y}</span></div>"
        f"<div class='row'><span class='k'>Total records</span><span class='v'>{len(M)+len(MAINT)+len(FAIL):,}</span></div>"
        "</div>",
        unsafe_allow_html=True)

# ── scoring ──────────────────────────────────────────────────────────────────
M = M.copy()
M["risk"] = D.score_risk(M)
M["status"] = [ui.status_of(r, WARN, CRIT)[0] for r in M["risk"]]
BASE, CONTRIB = D.explain_risk(M)
M["drivers"] = D.top_factors(M, CONTRIB)
RANKED = M.sort_values("risk", ascending=False)["machine_id"].tolist()
_RISK = M.set_index("machine_id")["risk"]


def _machine_label(m: str) -> str:
    return f"{m} · risk {_RISK[m]:.0f}%"


# ── form feedback ────────────────────────────────────────────────────────────
# st.rerun() discards anything drawn in the current run, so a success message has
# to be carried over to the next run to be seen at all.
def _flash(msg: str):
    st.session_state.flash = msg
    st.rerun()


def _show_flash():
    if msg := st.session_state.pop("flash", None):
        st.success(msg)


def _reload(*names: str):
    for n in names:
        st.session_state[n] = getattr(D, f"load_{n}")()
    st.session_state.machines = D.load_machines()  # features (and risk) changed


def _explain(mid: str):
    i = M.index[M["machine_id"] == mid][0]
    st.iframe(ui.explain_panel(mid, M.at[i, "risk"], BASE, D.explain_factors(M, CONTRIB, i),
                               warn=WARN, crit=CRIT), height="content")


# ── screens ──────────────────────────────────────────────────────────────────
# health-table columns between machine ID and risk: (header, column, format, right-aligned)
HEALTH_COLS = [
    ("Model", "model", str, False),
    ("Age (y)", "age_years", "{:.0f}".format, True),
    ("Errors 7 d", "errors_7d", "{:.0f}".format, True),
    ("Failures 12 mo", "failures_12mo", "{:.0f}".format, True),
]


def dashboard():
    counts = M["status"].value_counts()
    st.iframe(
        ui.metric_row(len(M), int(counts.get("Normal", 0)),
                      int(counts.get("Warning", 0)), int(counts.get("Critical", 0))),
        height="content")

    crit_rows = (M[M["status"] == "Critical"].sort_values("risk", ascending=False)
                 .rename(columns={"machine_id": "id", "model": "type"})
                 [["id", "type", "risk"]].to_dict("records"))
    st.iframe(ui.alarm_banner(crit_rows), height="content")

    st.iframe(ui.health_table(M, HEALTH_COLS, warn=WARN, crit=CRIT), height="content")

    # failures per month | explain a machine: equal heights on wide screens (the chart
    # frame is stretched by theme_css), stacked on narrow ones
    with st.container(key="dash_row"):
        left, right = st.columns(2)
        with left:
            labels, counts_m = D.monthly_failures(FAIL)
            st.iframe(ui.month_chart(labels, counts_m, note="Component failures, last 12 months of data"),
                      height="content")
        with right:
            mid = st.selectbox("Explain a machine", RANKED, format_func=_machine_label, key="dash_explain")
            _explain(mid)


def machines():
    tbl, form = st.columns([1.6, 1])
    with tbl:
        st.iframe(
            ui.data_table(
                M.sort_values("machine_id")[["machine_id", "model", "age_years", "errors_7d", "failures_12mo", "risk", "status"]]
                 .assign(age_years=lambda d: d["age_years"].map("{:.0f}".format),
                         errors_7d=lambda d: d["errors_7d"].map("{:.0f}".format),
                         risk=lambda d: d["risk"].map("{:.0f}".format)),
                "Asset register",
                ["ID", "Model", "Age (years)", "Errors 7 d", "Failures 12 mo", "Risk %", "Status"],
                right=[2, 3, 4, 5], visible_rows=15),
            height="content")
    with form:
        st.markdown("#### Add machine")
        _show_flash()
        with st.form("add_machine", clear_on_submit=True):
            a, b = st.columns(2)
            mid = a.text_input("Machine ID", value=next_machine_id())
            model = b.selectbox("Model", D.MACHINE_MODELS)
            age = st.number_input("Age (years)", min_value=0, max_value=60, step=1)
            st.caption("A new machine has no history yet: its components count as old as the "
                       "machine and its sensor readings as the fleet median until records arrive.")
            if st.form_submit_button("Add machine"):
                if not mid.strip():
                    st.error("Machine ID is required.")
                elif mid in M["machine_id"].values:
                    st.error(f"Machine ID '{mid}' already exists — choose a different ID.")
                else:
                    D.add_machine(mid.strip(), model, age)
                    st.session_state.machines = D.load_machines()
                    _flash(f"{mid} added to the register.")


_SHOW = 500  # rows rendered in a log table; filter to see older ones


def _log_table(d: pd.DataFrame, title: str, headers: list[str], foot: str, **kw) -> str:
    meta = f"{len(d):,} records" + (f" · latest {_SHOW} shown" if len(d) > _SHOW else "")
    return ui.data_table(d.head(_SHOW), title, headers, footnote=foot, visible_rows=15, meta=meta, **kw)


def _date_input(col, label: str = "Date"):
    return col.date_input(label, value=AS_OF, min_value=FIRST_DAY, max_value=AS_OF,
                          help=f"The demo data ends on {AS_OF:%d %b %Y}, so that is 'today' here.")


def maintenance_log():
    tbl, form = st.columns([1.6, 1])
    with tbl:
        f1, f2, f3 = st.columns(3)
        mach = f1.selectbox("Machine", ["All machines"] + M["machine_id"].tolist())
        comp = f2.selectbox("Component", ["All components"] + rf.COMPONENTS)
        typ = f3.selectbox("Type", ["All types"] + D.MAINT_TYPES)
        d = MAINT
        if mach != "All machines":
            d = d[d["machine_id"] == mach]
        if comp != "All components":
            d = d[d["component"] == comp]
        if typ != "All types":
            d = d[d["type"] == typ]
        n_corr = int((d["type"] == "Corrective").sum())
        st.iframe(_log_table(d, "Component replacements", ["Machine", "Date", "Component", "Type", "Note"],
                             f"{len(d):,} replacements · {n_corr:,} corrective (same day as a failure), "
                             f"{len(d) - n_corr:,} scheduled", chip_cols=[2, 3]),
                  height="content")
    with form:
        st.markdown("#### Log a replacement")
        _show_flash()
        with st.form("log_maint", clear_on_submit=True):
            a, b = st.columns(2)
            mid = a.selectbox("Machine", M["machine_id"].tolist())
            when = _date_input(b)
            comp2 = a.selectbox("Component", rf.COMPONENTS)
            typ2 = b.selectbox("Type", D.MAINT_TYPES)
            note = st.text_input("Note (optional)", placeholder="Replaced during planned stop")
            if st.form_submit_button("Log replacement"):
                D.add_maintenance(mid, when, comp2, typ2, note)
                _reload("maintenance")
                _flash(f"{comp2} replacement logged for {mid}.")


def failure_log():
    tbl, form = st.columns([1.6, 1])
    with tbl:
        f1, f2 = st.columns(2)
        mach = f1.selectbox("Machine", ["All machines"] + M["machine_id"].tolist(), key="fm")
        comp = f2.selectbox("Component", ["All components"] + rf.COMPONENTS, key="fc")
        d = FAIL
        if mach != "All machines":
            d = d[d["machine_id"] == mach]
        if comp != "All components":
            d = d[d["component"] == comp]
        by = d["component"].value_counts().reindex(rf.COMPONENTS, fill_value=0)
        st.iframe(_log_table(d, "Failure records", ["Machine", "Date", "Component", "Note"],
                             f"{len(d):,} failures · " + " · ".join(f"{c} {n}" for c, n in by.items()),
                             alarm_cols=[2]),
                  height="content")
    with form:
        st.markdown("#### Log failure")
        _show_flash()
        with st.form("log_fail", clear_on_submit=True):
            a, b = st.columns(2)
            mid = a.selectbox("Machine", M["machine_id"].tolist())
            when = _date_input(b)
            comp2 = a.selectbox("Component", rf.COMPONENTS)
            note = b.text_input("Note (optional)", placeholder="Seized bearing")
            replaced = st.checkbox("Component replaced (also logs a corrective replacement)", value=True)
            if st.form_submit_button("Log failure"):
                D.add_failure(mid, when, comp2, note, replaced=replaced)
                _reload("failures", "maintenance")
                _flash(f"{comp2} failure logged for {mid}.")


def _metrics(*parts: str) -> dict | None:
    p = RESULTS.joinpath(*parts, "metrics.json")
    return json.loads(p.read_text()) if p.exists() else None


def _figure(*parts: str, caption: str = ""):
    p = RESULTS.joinpath(*parts)
    if p.exists():
        st.image(str(p), caption=caption or None, width="stretch")


def _pct(x) -> str:
    return "–" if x is None else f"{x:.0%}"


def model_page():
    rm = _metrics("risk_model")
    st.iframe(ui.notice(
        "Public sample data",
        "The machines, replacements, errors, sensor readings and <b>failure events</b> come from the "
        "<b>Microsoft Azure Predictive Maintenance</b> sample dataset. Microsoft generated it as a "
        "teaching example; the failures are records in that dataset, not designed by this project. "
        "It shows how the approach works; it is not a claim about any real factory."),
        height="content")

    # ── 1. plain words ───────────────────────────────────────────────────────
    st.markdown("#### What the risk score means")
    st.markdown(
        f"Each machine's score is the model's estimate of the **chance that one of its components "
        f"fails in the next 7 days**. A score of 30 means roughly a 30% chance. On average a machine "
        f"has about a {BASE:.0f}% chance in any given week. "
        f"**Normal** is below {WARN}%, **Warning** {WARN}–{CRIT - 1}%, **Critical** {CRIT}% or more. "
        f"Scores update as soon as a replacement or failure is logged.")
    if rm:
        ev, al = rm["test_failure_events"]["critical"], rm["test_alerts"]["critical"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Failures caught", f"{ev['caught']} of {ev['events']}",
                  help="Test months (Nov–Dec 2015): failures with a Critical alert on at least one "
                       "of the 7 days before")
        c2.metric("Caught 2+ days ahead", f"{ev['caught_2plus_days_ahead']} of {ev['events']}")
        c3.metric("Critical alerts correct", _pct(al["precision"]),
                  help=f"{al['true_alarms']} of {al['flagged']} Critical machine-days were followed by a "
                       f"failure within 7 days; {al['false_alarms']} were false alarms")
        c4.metric("Model trained", rm["final_model"]["trained_on"],
                  help=f"On {rm['final_model']['rows']:,} machine-days, "
                       f"{rm['final_model']['first_day']} to {rm['final_model']['last_day']}")
        st.caption(f"Measured on the last two months of data, which the model had not seen when it was tested. "
                   f"Critical alerts flagged {al['flagged']} machine-days, of which {al['false_alarms']} "
                   f"were false alarms.")

        with st.expander("How it was validated"):
            sp, sel = rm["split"], rm["selected"]
            st.markdown(
                f"- **One row per machine per day** ({rm['dataset']['rows']:,} rows). The label says whether "
                f"that machine had a component failure in the next 7 days ({rm['dataset']['positive_rate']:.0%} of rows do).\n"
                f"- **Split by time**, never at random: train {sp['train']['first_day']} → {sp['train']['last_day']}, "
                f"validation {sp['val']['first_day']} → {sp['val']['last_day']}, test {sp['test']['first_day']} → "
                f"{sp['test']['last_day']}. The 7 days before each boundary are dropped from the earlier block, "
                f"because their labels look into the next one.\n"
                f"- **Model choice** on validation PR-AUC only: **{sel['model']}** "
                f"({', '.join(f'{k}={v}' for k, v in sel['params'].items())}). The test months were used once, "
                f"by models refit on train + validation.\n"
                f"- **Features** only use records up to the day being scored: component ages, the day's and the "
                f"7-day mean sensor readings, error codes in the last 7 days, machine age and model.")
            t = pd.DataFrame(rm["results"])
            for col in ("val_pr_auc", "pr_auc", "roc_auc"):
                t[col] = t[col].map("{:.3f}".format)
            t["brier"] = t["brier"].map("{:.4f}".format)
            for col in ("precision_critical", "recall_critical"):
                t[col] = t[col].map(lambda v: "–" if v is None or pd.isna(v) else f"{v:.2f}")
            st.iframe(ui.data_table(
                t[["model", "val_pr_auc", "pr_auc", "roc_auc", "brier", "precision_critical", "recall_critical"]],
                "Model comparison", ["Model", "PR-AUC (validation)", "PR-AUC (test)", "ROC-AUC (test)",
                                     "Brier (test)", f"Precision ≥ {CRIT}%", f"Recall ≥ {CRIT}%"],
                right=[1, 2, 3, 4, 5, 6],
                footnote=f"Test set: {sp['test']['rows']:,} machine-days. PR-AUC of always predicting the "
                         f"failure rate = {sp['test']['positive_rate']:.3f}. Precision and recall here count "
                         f"machine-days; the tiles above count failure events."),
                height="content")
            tc, cal = rm["threshold_choice"], rm["calibration"]
            vw, vc = tc["validation"]["warning"], tc["validation"]["critical"]
            st.markdown(
                f"**Thresholds** (set on validation, then checked on test):\n"
                f"- **Critical ≥ {CRIT}%**: a failure in the next week is more likely than not. On validation, "
                f"{_pct(vc['precision'])} of Critical days were followed by a failure "
                f"(on test: {_pct(al['precision'])}).\n"
                f"- **Warning ≥ {WARN}%**: about 1.5× the average weekly risk; the highest cut-off that still "
                f"flags {_pct(vw['recall'])} of the days before a failure on validation. It casts a wider net "
                f"(precision {_pct(vw['precision'])}), meant for the watch list, not for work orders.\n\n"
                f"**Calibration.** A score of 30 should mean about 30%. On validation the average gap between "
                f"predicted and observed rates was {cal['validation_ece']:.3f} (limit {cal['max_ece']}), so no "
                f"calibration step was added: isotonic calibration would only move the test Brier score from "
                f"{cal['test_brier_raw_train_only']} to {cal['test_brier_isotonic_train_only']}, and the raw "
                f"forest keeps the SHAP explanations exact.")
            rel = pd.DataFrame(cal["test_reliability"])
            rel["range"] = rel["bin"].map(lambda b: f"{b * 10}–{b * 10 + 10}%")
            st.iframe(ui.data_table(
                rel.assign(predicted=rel["predicted"].map("{:.0%}".format),
                           observed=rel["observed"].map("{:.0%}".format))[["range", "n", "predicted", "observed"]],
                "Predicted vs observed (test)", ["Score range", "Machine-days", "Average score", "Failed within 7 days"],
                right=[1, 2, 3]), height="content")
            a, b = st.columns(2)
            with a:
                _figure("risk_model", "pr_curves.png")
            with b:
                _figure("risk_model", "reliability.png")
            st.caption("Full walkthrough: notebooks/01_risk_model_azure.ipynb · "
                       "code: training/train_risk_model.py")
    else:
        st.info("Run training/train_risk_model.py to produce the validation results.")

    # ── 2. global SHAP ───────────────────────────────────────────────────────
    st.markdown("#### What matters most across the fleet")
    imp = D.global_importance(CONTRIB)
    top3 = [D.FEATURE_LABELS[f].lower() for f in imp.index[:3]]
    st.markdown(f"Averaged over all {len(M)} machines today, the factors that move the score most are "
                f"**{top3[0]}**, **{top3[1]}** and **{top3[2]}**.")
    st.iframe(ui.importance_bars([(D.FEATURE_LABELS[f], v) for f, v in imp.items()],
                                 "Average impact on the risk score",
                                 f"mean |SHAP| over {len(M)} machines, percentage points"), height="content")
    with st.expander("How to read this"):
        st.markdown(
            "Each bar is the average size of a factor's push on the score, up or down, across the fleet "
            "on the latest day (SHAP values from `shap.TreeExplainer`). Component ages matter because "
            "components in this dataset tend to fail after long service; the sensor and error columns matter "
            "because the readings drift in the day or two before a failure.\n\n"
            "SHAP explains **the model**, not the physics: a large bar means the model relies on that input, "
            "not that changing it would prevent a failure.")
        _figure("risk_model", "shap_summary.png",
                caption="SHAP values on the test months (each dot is one machine-day)")

    # ── 3. one machine ───────────────────────────────────────────────────────
    st.markdown("#### Explain one machine")
    pick, _ = st.columns([1, 1.4])
    mid = pick.selectbox("Machine", RANKED, format_func=_machine_label, key="model_explain")
    _explain(mid)

    # ── 4. what-if ───────────────────────────────────────────────────────────
    st.markdown("#### What if…")
    st.markdown("Pick a machine and simulate an action to see how its score would change. "
                "Nothing is saved.")
    left, right = st.columns([1, 1.4])
    with left:
        wid = st.selectbox("Machine", RANKED, format_func=_machine_label, key="wi_machine")
        row = M[M["machine_id"] == wid].iloc[0]
        comps = st.multiselect(
            "Replace components today", rf.COMPONENTS, key="wi_comps",
            format_func=lambda c: f"{c} ({row[f'days_since_{c}']:.0f} days old)")
        full = st.toggle("Full service: replace all four components", key="wi_full")
        sensors = st.toggle("Sensor readings back to fleet normal", key="wi_sensors",
                            help="As if the fault behind unusual readings were fixed")
        errors = st.toggle("No error codes in the last 7 days", key="wi_errors")
    w = D.what_if(M, wid, replace=rf.COMPONENTS if full else comps, sensors_normal=sensors, clear_errors=errors)
    r = D.score_risk(w)
    w_base, w_contrib = D.explain_risk(w)
    with right:
        a, b = st.columns(2)
        a.metric("Risk now", f"{r['now']:.0f}%", help=ui.status_of(r["now"], WARN, CRIT)[0])
        b.metric("Risk after", f"{r['after']:.0f}%", delta=f"{r['after'] - r['now']:+.0f} points",
                 delta_color="inverse")
        st.iframe(ui.explain_panel(wid, r["after"], w_base, D.explain_factors(w, w_contrib, "after", D.fleet_medians(M)),
                                   warn=WARN, crit=CRIT, title=f"After: why {wid} would score {r['after']:.0f}"),
                  height="content")
    st.caption("The what-if changes only the model's inputs. It shows how the estimate responds; it cannot "
               "promise the machine would behave that way.")

    # ── 5. benchmarks ────────────────────────────────────────────────────────
    with st.expander("Also tested on public benchmarks"):
        benchmarks()


def benchmarks():
    st.iframe(ui.notice(
        "Simulated benchmark datasets",
        "Both benchmarks below use public datasets that were <b>generated by simulators</b> "
        "(AI4I 2020 by documented failure rules, C-MAPSS by NASA's engine simulation), not recorded "
        "from real factories or aircraft. They show that the modelling workflow holds up on standard "
        "predictive-maintenance problems. They are not a claim about real-world accuracy."),
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


_SUBTITLES = {
    "Dashboard": "Fleet health overview",
    "Machines": "Asset register",
    "Maintenance Log": "Component replacements",
    "Failure Log": "Breakdown history",
    "Model": "How the risk score works, and how far to trust it",
}
st.markdown(
    f"<div style='display:flex;align-items:baseline;gap:14px;margin-bottom:2px'>"
    f"<span style=\"font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:34px;"
    f"letter-spacing:.02em;text-transform:uppercase\">{screen}</span>"
    f"<span style='font-size:14px;color:#6b6f73'>{_SUBTITLES[screen]}</span></div>",
    unsafe_allow_html=True)
{"Dashboard": dashboard, "Machines": machines,
 "Maintenance Log": maintenance_log, "Failure Log": failure_log, "Model": model_page}[screen]()

st.iframe(ui.status_strip([
    ("Data", "Azure PdM sample (Microsoft)"), ("Model", D.model_name()),
    ("Records", f"{len(M) + len(MAINT) + len(FAIL):,}"),
]), height="content")
