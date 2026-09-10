"""Industry-styled predictive-maintenance dashboard on Streamlit.

Run:  streamlit run app.py
Layout: display surfaces are rendered as HTML embeds (ui.py); every input is a
native Streamlit widget, so the Python model stays in charge.
"""

from __future__ import annotations

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
WARN, CRIT = 40, 70

_SCREENS = ["Dashboard", "Machines", "Maintenance Log", "Failure Log"]
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
    # the bold markdown count is pushed flush right by theme_css
    screen = st.radio("Monitor", _SCREENS, format_func=lambda s: f"{s} **{_counts[s]}**",
                      label_visibility="collapsed")
    _last = pd.to_datetime(MAINT["date"]).max().strftime("%d %b %Y") if len(MAINT) else "—"
    st.markdown(
        "<div class='sb-info'>"
        "<div class='row'><span class='k'>Model</span><span class='v'>Random Forest Regressor</span></div>"
        f"<div class='row'><span class='k'>Latest record</span><span class='v'>{_last}</span></div>"
        f"<div class='row'><span class='k'>Total records</span><span class='v'>{len(M)+len(MAINT)+len(FAIL)}</span></div>"
        "</div>",
        unsafe_allow_html=True)

# ── scoring ──────────────────────────────────────────────────────────────────
M = M.copy()
M["risk"] = D.score_risk(M)
M["status"] = [ui.status_of(r, WARN, CRIT)[0] for r in M["risk"]]


# ── screens ──────────────────────────────────────────────────────────────────
def dashboard():
    counts = M["status"].value_counts()
    st.iframe(
        ui.metric_row(len(M), int(counts.get("Normal", 0)),
                      int(counts.get("Warning", 0)), int(counts.get("Critical", 0))),
        height=ui.H_METRICS)

    crit_rows = (M[M["status"] == "Critical"].sort_values("risk", ascending=False)
                 .rename(columns={"machine_id": "id", "machine_type": "type"})
                 [["id", "type", "risk"]].to_dict("records"))
    st.iframe(ui.alarm_banner(crit_rows), height="content")

    left, right = st.columns([1.3, 1])
    with left:
        st.iframe(ui.risk_chart(M, warn=WARN, crit=CRIT), height=60 + 16 * len(M) + 30)
    with right:
        labels, counts_m = D.monthly_failures(FAIL)
        st.iframe(ui.month_chart(labels, counts_m), height=290)

    st.iframe(ui.health_table(M, warn=WARN, crit=CRIT),
                    height=ui.table_height(len(M)))


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
            height=ui.table_height(len(M)))
    with form:
        st.markdown("#### Add machine")
        with st.form("add_machine", clear_on_submit=True):
            a, b = st.columns(2)
            mid = a.text_input("Machine ID", value=next_machine_id())
            mtype = b.text_input("Type", placeholder="Centrifugal Pump")
            loc = a.text_input("Location", placeholder="Line A")
            inst = b.date_input("Install date")
            hours = st.number_input("Operating hours at commissioning", min_value=0, step=100)
            if st.form_submit_button("Add machine"):
                if not mid.strip():
                    st.error("Machine ID is required.")
                elif mid in M["machine_id"].values:
                    st.error(f"Machine ID '{mid}' already exists — choose a different ID.")
                else:
                    D.add_machine(mid, mtype or "Unspecified", loc or "—", inst, hours)
                    st.session_state.machines = D.load_machines()
                    st.success(f"{mid} added to the register.")
                    st.rerun()


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
            height=ui.table_height(len(d), footnote=True))
    with form:
        st.markdown("#### Log maintenance")
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
                    st.success(f"Record logged for {mid}.")
                    st.rerun()


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
            height=ui.table_height(len(d), footnote=True))
    with form:
        st.markdown("#### Log failure")
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
                    st.success(f"Failure logged for {mid}.")
                    st.rerun()


_SUBTITLES = {
    "Dashboard": "Fleet health overview",
    "Machines": "Asset register",
    "Maintenance Log": "Service history",
    "Failure Log": "Breakdown history",
}
st.markdown(
    f"<div style='display:flex;align-items:baseline;gap:14px;margin-bottom:2px'>"
    f"<span style=\"font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:34px;"
    f"letter-spacing:.02em;text-transform:uppercase\">{screen}</span>"
    f"<span style='font-size:14px;color:#6b6f73'>{_SUBTITLES[screen]}</span></div>",
    unsafe_allow_html=True)
{"Dashboard": dashboard, "Machines": machines,
 "Maintenance Log": maintenance_log, "Failure Log": failure_log}[screen]()

st.iframe(ui.status_strip([
    ("Historian", "OK"), ("Model", "Random Forest Risk Regressor"),
    ("Records", str(len(M) + len(MAINT) + len(FAIL))),
]), height=ui.H_STRIP)
