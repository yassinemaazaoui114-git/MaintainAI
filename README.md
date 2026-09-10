# Smart Predictive Maintenance — Industry-styled hybrid app

```
pip install -r requirements.txt
streamlit run app.py
```

## What this is

The same working application (Machines, Maintenance, Failures, Dashboard, AI Risk Score),
redesigned visually via Claude Design, then wired to the real data and trained model.

## Files

| File | Role |
| --- | --- |
| `app.py` | Screens, sidebar nav, native Streamlit widgets (forms/filters/sliders) |
| `ui.py` | HTML renderers — every visual surface of the design (unmodified from Claude Design) |
| `theme_css.py` | Restyles Streamlit's own chrome (sidebar, inputs, buttons) |
| `data.py` | **Real** loaders (factory_maintenance.db) + real model (risk_model.pkl) |
| `db_utils.py` | Shared DB connection/query helpers (same tested layer used elsewhere) |
| `factory_maintenance.db` | The real SQLite database |
| `risk_model.pkl` | The trained Random Forest risk model |

## What changed from Claude Design's original hand-off

- `data.py` was rewritten from scratch: loads real machines/maintenance/failures from
  `factory_maintenance.db` instead of sample data, and `score_risk()` calls the actual
  trained model instead of a placeholder formula.
- **Fixed a persistence bug**: the original forms only wrote to `st.session_state`,
  so logged records vanished on refresh. Forms now write directly to the database.
- Fixed a fake model label ("RUL-GBM v4.2" → "Random Forest Risk Regressor") and
  placeholder branding ("Northfield Works" → "Smart Predictive Maintenance").
- Migrated `st.components.v1.html` → `st.iframe`, since the former is deprecated and
  the stated removal date has already passed — every visual element in this design
  depends on this call, so this matters.
- Threshold sliders now default to 40/70 (Warning/Critical), matching the project's
  confirmed working assumption, not Claude Design's placeholder 45/75.
- `failures` / `downtime` per machine are computed live from the Failure_Log (via the
  `Machine_Risk_Features` view), never stored redundantly on the machine itself.
- Retired the old `pages/`-folder multipage structure — this app is a single file
  with its own sidebar radio nav, so keeping both would show two conflicting
  navigation systems.

## Known minor inconsistency (not a bug)

The Failure Log's "Add" form offers a simplified type list (Mechanical, Electrical,
Thermal, Hydraulic, Control) — different from the older, longer list already in the
existing 59 failure records. Neither list is used as an AI feature, so this is purely
cosmetic — old and new records will just show different phrasing for failure type.

## How the hybrid works

Read-only display surfaces (metric row, alarm banner, risk bars, month chart, tables)
are `st.iframe(...)` embeds — each is a sandboxed iframe, so it can't call back into
Python. All *inputs* (forms, filters, threshold sliders) are native Streamlit widgets,
so they round-trip through Python and write to the database normally.

Every screen and form submission was tested with Streamlit's `AppTest` harness before
delivery, including verifying the actual database writes — not just that pages load.
