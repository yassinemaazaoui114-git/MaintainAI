# Smart Predictive Maintenance System

An AI-powered system for tracking factory machines and predicting which ones need
maintenance before they fail. It logs machine, maintenance, and failure data, computes
a **Risk Score (0–100)** for every machine using a trained machine-learning model, and
presents everything on a dashboard so a maintenance team can prioritize inspections
instead of waiting for breakdowns.

Built as a graduation project. Tech stack: **Python · Pandas · scikit-learn · Streamlit · SQLite**.

---

## Features

The system is organized into five parts:

| Module | What it does |
| --- | --- |
| **Dashboard** | Fleet overview: machine counts by status, critical-machine alerts, a Risk Score chart, a failures-over-time chart, and a sortable health table. |
| **Machines** | View all machines and add new ones (ID, type, location, install date, operating hours). |
| **Maintenance** | View and filter the maintenance log; record new maintenance events. |
| **Failures** | View and filter the failure log; record new failures with downtime and repair time. |
| **AI Risk Score** | A trained Random Forest model scores each machine 0–100 from its operating data and buckets it into Normal / Warning / Critical. |

---

## Quick start

```
pip install -r requirements.txt
streamlit run app.py
```

This opens the app in your browser (usually http://localhost:8501). Use the sidebar to
move between the Dashboard, Machines, Maintenance, and Failure Log.

**Browser note:** use **Microsoft Edge or Chrome**. In the Brave browser, text fields
don't always repaint visually as you type (the data still saves correctly, but you can't
see it while typing) — a Brave rendering quirk, not a bug in the app.

---

## How the Risk Score works

Each machine is scored from eight features: operating hours, machine age, days since last
maintenance, failure count, total downtime, maintenance frequency, and recent temperature
and vibration readings. A `RandomForestRegressor` (scikit-learn) turns these into a single
0–100 score, which is then bucketed:

- **Normal** — below the Warning threshold (40)
- **Warning** — between the two thresholds (40–70)
- **Critical** — at or above the Critical threshold (70)

The thresholds are fixed constants (`WARN` / `CRIT` in `app.py`).

> **Important — for honest reporting.** The model is trained on a **synthetic dataset**
> with a **designed target formula**, because real labelled failure data isn't available
> for this project. The strong evaluation results (MAE ≈ 3.4, R² ≈ 0.96 on held-out data)
> demonstrate that the **pipeline and methodology are sound** — they do **not** prove
> real-world predictive accuracy. A production system would need a large set of genuine
> historical failures to validate that. See `risk_score_model.ipynb` for the full
> methodology, feature engineering, and evaluation.

---

## Project structure

| File | Role |
| --- | --- |
| `app.py` | Main application — screens, sidebar navigation, all input forms and filters. |
| `ui.py` | Renders the dashboard's visual surfaces (metric cards, charts, tables, alert banner). |
| `theme_css.py` | Applies the industrial visual theme to the interface. |
| `data.py` | Loads data from the database and runs the trained model to score machines. |
| `db_utils.py` | Shared database connection and query helpers. |
| `factory_maintenance.db` | SQLite database (machines, maintenance log, failure log, sensor readings). |
| `risk_model.pkl` | The trained Random Forest model. |
| `requirements.txt` | Python dependencies (pinned versions). |

The related notebook `risk_score_model.ipynb` (kept alongside this project) documents how
the model was built, trained, and evaluated.

---

## How it's built

Display surfaces (metric cards, charts, tables) are rendered as embedded HTML for a
polished, consistent look. All inputs — the forms and filters — are
native Streamlit widgets, so everything the user enters is processed in Python and saved
to the database. Records added through the forms persist in `factory_maintenance.db`.

---

## Notes

- **Data is synthetic.** Machine records, sensor readings, and history are generated for
  demonstration; the database contains no real or personal information.
- **Deployment:** designed to run locally. No cloud hosting or external services required.
- **Scale:** ships with 50 machines and ~2 years of history — enough to demonstrate the
  system end to end.