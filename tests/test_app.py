"""Streamlit AppTest: every page renders, and every form writes to the temporary database."""

import pytest
from streamlit.testing.v1 import AppTest

from conftest import ROOT

SCREENS = ["Dashboard", "Machines", "Maintenance Log", "Failure Log", "Benchmarks"]


def open_app(screen: str = "Dashboard") -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
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
    pick = widget(at.selectbox, "Explain a machine's score")
    assert len(pick.options) == 50
    pick.set_value(pick.options[-1]).run()  # the lowest-risk machine
    assert not at.exception


def test_add_machine_writes_to_temp_db(temp_db, query):
    at = open_app("Machines")
    widget(at.text_input, "Machine ID").set_value("MCH-T01")
    widget(at.text_input, "Type").set_value("Test Rig")
    widget(at.number_input, "Operating hours (meter reading)").set_value(1200)
    submit(at, "Add machine")
    assert [s.value for s in at.success] == ["MCH-T01 added to the register."]
    assert query("SELECT type, operating_hours FROM Machines WHERE machine_id = 'MCH-T01'") == [("Test Rig", 1200)]


def test_add_machine_rejects_duplicate_id(temp_db, query):
    at = open_app("Machines")
    widget(at.text_input, "Machine ID").set_value("MCH-001")
    submit(at, "Add machine")
    assert at.error and "already exists" in at.error[0].value
    assert query("SELECT COUNT(*) FROM Machines") == [(50,)]


def test_log_maintenance_writes_to_temp_db(temp_db, query):
    at = open_app("Maintenance Log")
    widget(at.selectbox, "Machine").set_value("MCH-007")
    widget(at.selectbox, "Type").set_value("Corrective")
    widget(at.number_input, "Duration (h)").set_value(2.5)
    widget(at.text_area, "Reason").set_value("Test: replaced coupling")
    submit(at, "Log maintenance")
    assert at.success
    assert query("SELECT machine_id, type, duration_hours FROM Maintenance_Log WHERE reason = ?",
                 ("Test: replaced coupling",)) == [("MCH-007", "Corrective", 2.5)]


def test_log_maintenance_requires_reason(temp_db, query):
    at = open_app("Maintenance Log")
    submit(at, "Log maintenance")
    assert at.error and "reason" in at.error[0].value
    assert query("SELECT COUNT(*) FROM Maintenance_Log") == [(290,)]


def test_log_failure_writes_to_temp_db(temp_db, query):
    at = open_app("Failure Log")
    widget(at.selectbox, "Machine").set_value("MCH-012")
    widget(at.selectbox, "Failure type").set_value("Electrical")
    widget(at.number_input, "Downtime (h)").set_value(6.0)
    widget(at.number_input, "Repair time (h)").set_value(3.5)
    widget(at.text_input, "Cause").set_value("Test: blown fuse")
    submit(at, "Log failure")
    assert at.success
    assert query("SELECT machine_id, failure_type, downtime_hours, repair_hours FROM Failure_Log WHERE cause = ?",
                 ("Test: blown fuse",)) == [("MCH-012", "Electrical", 6.0, 3.5)]
