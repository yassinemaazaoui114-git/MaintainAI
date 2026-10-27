# MaintainAI — living spec

Personal learning / portfolio project: a small CMMS (machines, maintenance, failures) with a
predictive-maintenance risk score, plus benchmarks on public predictive-maintenance datasets.
This file is the source of truth for scope, decisions and open questions. Update it with every change.

_Last updated: 2026-10-27_

---

## 1. Scope

| Area | What it does | Status |
|---|---|---|
| Machines | Asset register + add-machine form | Done |
| Maintenance | Maintenance log + log form | Done |
| Failures | Failure log + log form | Done |
| Risk score | Random Forest, 0–100, Normal / Warning / Critical | Rebuilt 2026-10-27 |
| Explanations | Per-machine SHAP factors in plain words | Added 2026-10-27 |
| Dashboard | Fleet counts, alarm, risk chart, failures/month, health table | Done |
| Benchmarks | AI4I 2020 (failure classification), NASA C-MAPSS FD001 (RUL) | Added 2026-10-27 |
| Tests | Streamlit AppTest + model bundle test | Added 2026-10-27 |
| Deployment | Streamlit Community Cloud from GitHub | Prepared, not deployed |

Not planned: spare parts/inventory, OEE, mobile app, live IoT ingestion, user accounts.

## 2. Confirmed vs. assumed

**Confirmed (by the owner)**
- Personal learning project for a CV. No academic scope limits, no report.
- Stack: Python, pandas, scikit-learn, Streamlit, SQLite.
- `factory_maintenance.db` is fully synthetic (50 machines, 290 maintenance, 153 failures,
  4,915 weekly sensor readings, Sep 2024 – Sep 2026). No personal data.
- The risk model is trained on a designed proxy formula, not on observed failures, and must be described that way.
- Benchmarks use AI4I 2020 and C-MAPSS FD001, both simulated public datasets, and are labelled as such.
- Hosting target: Streamlit Community Cloud; the owner deploys it.

**Assumed (working assumptions — change freely)**
- Thresholds: Normal < 40, Warning 40–69, Critical ≥ 70 (stored in `risk_model.pkl`).
- Regression (continuous 0–100 score) rather than classification for the in-app model.
- "Now" for the app = the date of the latest log or sensor record (see D2).
- Operating hours accrue evenly since install (needed to back-estimate past hour-meter readings).
- Proxy formula weights and healthy/bad ranges (see `risk_features.PROXY_TERMS`) are my engineering judgement.
- On Streamlit Cloud, records added through the forms are temporary (the container's disk resets on restart). Fine for a demo.

## 3. Decisions

| # | Decision | Why |
|---|---|---|
| D1 | One feature module (`risk_features.py`) used by training, notebook and app | Training and scoring can't drift apart |
| D2 | Features are computed in Python "as of" the latest record date, not via the DB view with `julianday('now')` | The demo data is frozen; measuring against the wall clock made every machine drift towards Critical over time. The `Machine_Risk_Features` view stays in the DB but the app no longer uses it |
| D3 | Failures, downtime and maintenance frequency use a trailing 12-month window | Recent history is what matters for risk, and a fixed window means older machines aren't penalised just for having a longer log |
| D4 | Training data = weekly snapshots from one year after the first record (2,650 rows) | Counts over a 12-month window are never truncated |
| D5 | Main evaluation holds out whole machines (GroupShuffleSplit + 5-fold GroupKFold) | Weekly snapshots of one machine are near-duplicates; a random row split leaks them (the old 3.37 MAE was on such a split) |
| D6 | Keep Random Forest even though linear regression scores better on the proxy | The proxy is nearly linear by construction; a real failure process wouldn't be. RF also gives exact TreeExplainer SHAP values |
| D7 | RF settings: 300 trees, `min_samples_leaf=5`, `max_features=0.5` | Best 5-fold group-CV MAE in a small grid (3.89 vs 4.16) |
| D8 | Final model is refit on all snapshots; reported metrics come from the held-out evaluation | Standard practice; the app benefits from all data |
| D9 | All HTML embeds size their own iframe (`height="content"` + fit script) | Python-side height estimates clipped the risk chart and health table |
| D10 | Risk status is judged on the rounded, displayed score | A displayed "40" was labelled Normal |
| D11 | Saving a maintenance or failure record also reloads machine features | Otherwise the risk score didn't update until a page refresh |
| D12 | Live features include every machine; one installed after the "as of" date counts as brand new | Adding a machine with today's install date crashed the dashboard (NaN features) |
| D13 | Sidebar nav radio has a fixed `key` | Its labels contain record counts; when a count changed, Streamlit treated it as a new widget and jumped back to Dashboard after every save |
| D14 | Form success messages survive `st.rerun()` via session state | They were drawn and then wiped by the rerun, so users never saw a confirmation |
| D15 | Explanations: SHAP TreeExplainer; the health table shows the top 2 risk-raising factors (≥ 1 point), with a per-machine breakdown panel below | Plain-words "why" without changing the existing layout |
| D16 | Benchmarks: model choice by training-CV only; test set touched once. AI4I: class weights + F1-tuned threshold; failure-mode columns, UDI and Product ID excluded. C-MAPSS: official split, RUL cap 125, 10-cycle rolling features | Avoid leakage and test-set selection bias |
| D17 | Benchmark raw data is downloaded by the notebooks into git-ignored `data/`; only metrics JSON and PNGs are committed | Keep the repo small; respect dataset hosting |
| D18 | `.streamlit/config.toml`: light theme base, viewer toolbar | The custom styling assumes a light base; hides the Deploy/dev menu for visitors |
| D19 | `requirements.txt` = app runtime only (exact pins); `requirements-dev.txt` adds tests, notebooks, Playwright | Smaller, faster Community Cloud install; matplotlib is not needed to run the app |

## 4. Open questions

- Should the in-app risk model be retargeted to an observed outcome (e.g. "failure in the next 30 days")
  using the failure log itself, instead of the proxy formula? It would be more honest, but 153 failures over 50 machines is thin.
- Synthetic features are highly collinear (hours vs age r = 0.99, vibration vs hours 0.93). Regenerate a more varied
  fleet so SHAP explanations line up with the formula's weights?
- Model selection on C-MAPSS/AI4I used fixed, untuned gradient-boosting settings; tune them for a fairer comparison?
- Persistent storage for the hosted demo (e.g. a hosted Postgres or SQLite in object storage), or keep it ephemeral?

## 5. Results snapshot

Risk model (proxy target, 2,650 snapshots, 50 machines) — see `results/risk_model/metrics.json`:

| Evaluation | MAE | RMSE | R² |
|---|---|---|---|
| RF, 10 held-out machines | 4.64 | 5.85 | 0.944 |
| RF, 5-fold group CV (mean ± sd) | 3.89 ± 0.37 | — | 0.937 ± 0.016 |
| RF, random row split (leaky, for comparison) | 3.47 | — | 0.959 |
| Linear regression, held-out machines | 3.53 | 4.49 | 0.967 |
| Mean baseline, held-out machines | 21.89 | 27.34 | −0.22 |
| Noise ceiling (true formula) | 3.32 | — | 0.971 |
| Old model (random row split, retired) | 3.37 | 4.28 | 0.959 |

## 6. Change log

| Date | What | Why |
|---|---|---|
| 2026-09-08 → 2026-09-10 | Original build: database, Streamlit app, first risk model, Industry-style redesign (history in the retired spec, not committed) | Initial project |
| 2026-10-27 | Reframed as a personal portfolio project; old spec and handoff ignored by git; this SPEC.md created | Scope change by the owner |
| 2026-10-27 | Set up `.venv`, confirmed the app runs on Python 3.12 / Streamlit 1.63 / scikit-learn 1.9 | Task 1 |
| 2026-10-27 | Confirmed the sidebar radio dot is hidden in Chromium (marker `display:none`, native input visually hidden) | Leftover cosmetic item; already fixed by the existing CSS |
| 2026-10-27 | Fixed clipped risk chart and health table: every embed now sizes its own iframe | Found while checking task 1 |
| 2026-10-27 | Rebuilt the risk model: `risk_features.py` (features + documented proxy formula), `training/train_risk_model.py`, `notebooks/01_risk_model_synthetic.ipynb`, new `risk_model.pkl` | Original notebook and formula were lost |
| 2026-10-27 | App features now come from `risk_features` as of the latest record (D2, D3) | Stop time drift; one feature definition |
| 2026-10-27 | Found and documented strong collinearity in the synthetic data; SHAP credits vibration/hours more than the formula's weights | Honest reading of feature importance and explanations |
| 2026-10-27 | Added `notebooks/02_benchmark_ai4i2020.ipynb` and `03_benchmark_nasa_cmapss.ipynb`, results in `results/benchmarks/`, and a Benchmarks page | Task 3: real-world-style benchmarks on public (simulated) datasets |
| 2026-10-27 | Added SHAP explanations: "Main risk drivers" column + "Explain a machine's score" panel | Task 4 |
| 2026-10-27 | Added `tests/` (18 tests: AppTest pages + forms on a temp DB copy, model bundle, features, SHAP additivity) | Task 5. The tests caught three real bugs (D12–D14) |
| 2026-10-27 | Added LICENSE (MIT), pinned requirements, requirements-dev.txt, .gitignore, Playwright screenshot script + docs/screenshots, rewritten README | Task 6 |
| 2026-10-27 | Deployment prep: relative paths (`Path(__file__)`), `MAINTAINAI_DB` override, `.streamlit/config.toml`, checked in a clean venv with only requirements.txt | Task 7: Streamlit Community Cloud |
