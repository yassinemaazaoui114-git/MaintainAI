"""Streamlit AppTest: every page renders, every form writes to the temporary database,
and the Model page's what-if responds without writing anything."""

from datetime import date

import pytest
from streamlit.testing.v1 import AppTest

from conftest import ROOT

SCREENS = ["Dashboard", "Machines", "Maintenance Log", "Failure Log", "Model"]
AS_OF = date(2015, 12, 31)  # the latest record in the demo data


def open_app(screen: str = "Dashboard") -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    if screen != "Dashboard":
        at.sidebar.radio[0].set_value(screen).run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def widget(widgets, label, index=-1):
    """Widgets by label; the forms sit after the page filters, so default to the last match."""
    found = [w for w in widgets if w.label == label]
    assert found, f"no widget labelled {label!r}"
    return found[index]


def submit(at: AppTest, label: str) -> AppTest:
    widget(at.button, label).click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("screen", SCREENS)
def test_page_loads(temp_db, screen):
    at = open_app(screen)
    assert at.sidebar.radio[0].value == screen


def test_dashboard_explains_a_machine(temp_db):
    at = open_app()
    pick = widget(at.selectbox, "Explain a machine")
    assert len(pick.options) == 100
    assert pick.options[0].startswith(pick.value + " ")  # highest risk first (options are labels)
    pick.set_value(pick.options[-1].split(" · ")[0]).run()  # the lowest-risk machine
    assert not at.exception


def test_forms_default_to_the_data_date(temp_db):
    at = open_app("Maintenance Log")
    d = widget(at.date_input, "Date")
    assert d.value == AS_OF


def test_add_machine_writes_to_temp_db(temp_db, query):
    at = open_app("Machines")
    widget(at.text_input, "Machine ID").set_value("MCH-T01")
    widget(at.selectbox, "Model").set_value("model2")
    widget(at.number_input, "Age (years)").set_value(3)
    submit(at, "Add machine")
    assert [s.value for s in at.success] == ["MCH-T01 added to the register."]
    assert query("SELECT model, age_years FROM Machines WHERE machine_id = 'MCH-T01'") == [("model2", 3)]
    # a machine without any history is still scored on the dashboard
    at.sidebar.radio[0].set_value("Dashboard").run()
    assert not at.exception
    assert len(widget(at.selectbox, "Explain a machine").options) == 101


def test_add_machine_rejects_duplicate_id(temp_db, query):
    at = open_app("Machines")
    widget(at.text_input, "Machine ID").set_value("MCH-001")
    submit(at, "Add machine")
    assert at.error and "already exists" in at.error[0].value
    assert query("SELECT COUNT(*) FROM Machines") == [(100,)]


def test_log_replacement_writes_to_temp_db(temp_db, query):
    at = open_app("Maintenance Log")
    widget(at.selectbox, "Machine").set_value("MCH-007")
    widget(at.selectbox, "Component").set_value("comp3")
    widget(at.selectbox, "Type").set_value("Corrective")
    widget(at.text_input, "Note (optional)").set_value("Test: replaced coupling")
    submit(at, "Log replacement")
    assert at.success
    assert query("SELECT machine_id, date, component, type FROM Maintenance_Log WHERE note = ?",
                 ("Test: replaced coupling",)) == [("MCH-007", "2015-12-31", "comp3", "Corrective")]


def test_log_failure_also_logs_the_replacement(temp_db, query):
    at = open_app("Failure Log")
    before = query("SELECT COUNT(*) FROM Maintenance_Log")[0][0]
    widget(at.selectbox, "Machine").set_value("MCH-012")
    widget(at.selectbox, "Component").set_value("comp2")
    widget(at.text_input, "Note (optional)").set_value("Test: seized bearing")
    submit(at, "Log failure")
    assert at.success
    assert query("SELECT machine_id, date, component FROM Failure_Log WHERE note = ?",
                 ("Test: seized bearing",)) == [("MCH-012", "2015-12-31", "comp2")]
    assert query("SELECT COUNT(*) FROM Maintenance_Log")[0][0] == before + 1
    assert query("SELECT component, type FROM Maintenance_Log ORDER BY id DESC LIMIT 1") == [("comp2", "Corrective")]


def test_logging_a_replacement_updates_the_risk_score(temp_db):
    at = open_app()
    before = widget(at.selectbox, "Explain a machine").options[0]  # "MCH-xxx · risk 72%"
    riskiest = before.split(" · ")[0]
    at.sidebar.radio[0].set_value("Maintenance Log").run()
    for comp in ("comp1", "comp2", "comp3", "comp4"):
        widget(at.selectbox, "Machine").set_value(riskiest)
        widget(at.selectbox, "Component").set_value(comp)
        submit(at, "Log replacement")
    at.sidebar.radio[0].set_value("Dashboard").run()
    after = next(o for o in widget(at.selectbox, "Explain a machine").options if o.startswith(riskiest))
    assert after != before


def test_what_if_changes_risk_without_writing(temp_db, query):
    at = open_app("Model")
    counts = query("SELECT (SELECT COUNT(*) FROM Maintenance_Log), (SELECT COUNT(*) FROM Failure_Log)")
    machine = widget(at.selectbox, "Machine")  # what-if picker, riskiest machine by default
    assert machine.options[0].startswith(machine.value + " ")
    risk = {m.label: m.value for m in at.metric}
    assert risk["Risk now"] == risk["Risk after"]
    at.toggle(key="wi_full").set_value(True).run()
    at.toggle(key="wi_sensors").set_value(True).run()
    assert not at.exception
    risk = {m.label: m.value for m in at.metric}
    assert float(risk["Risk after"].rstrip("%")) < float(risk["Risk now"].rstrip("%"))
    assert query("SELECT (SELECT COUNT(*) FROM Maintenance_Log), (SELECT COUNT(*) FROM Failure_Log)") == counts
