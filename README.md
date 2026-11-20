# Apex · Formula 1 podium prediction platform

Apex forecasts Formula 1 race outcomes. For every upcoming Grand Prix it estimates each driver's probability of
winning, finishing in the top two, on the podium, in the top five and in the top ten, explains why, and lets you
run "what if" scenarios through a Monte Carlo race simulator. It downloads its own data, retrains itself after
every race, and keeps a permanent record of every prediction it has made so its accuracy can be checked.

Everything runs on free, keyless data sources and open-source Python.

![stack](https://img.shields.io/badge/python-3.11%2B-blue) ![ui](https://img.shields.io/badge/UI-Streamlit-red) ![db](https://img.shields.io/badge/storage-SQLite-lightgrey)

---

## Contents

1. [Quick start](#quick-start)
2. [What the dashboard shows](#what-the-dashboard-shows)
3. [Data sources and what they can and cannot do](#data-sources)
4. [How the predictive engine works](#how-the-predictive-engine-works)
5. [Preventing data leakage](#preventing-data-leakage)
6. [Monte Carlo simulator](#monte-carlo-simulator)
7. [Explainability](#explainability)
8. [Validation and results](#validation-and-results)
9. [Automatic updates](#automatic-updates)
10. [Project structure](#project-structure)
11. [Known limitations](#known-limitations)
12. [Methodology notes (modelling decisions)](#methodology-notes)

---

## Quick start

You need Python 3.11 or newer (Anaconda is fine).

```bash
git clone https://github.com/RogueTiger3/project.git apex
cd apex
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The repository already contains the database (`data/apex.db`) and the trained models (`models/`), so the
dashboard works straight away. Your browser opens at http://localhost:8501.

To pull the latest data yourself instead of waiting for the scheduled job:

```bash
python -m apex.pipeline update            # download new data, retrain if needed, refresh predictions
python -m apex.pipeline update --retrain  # force a full retrain and backtest (about 5 minutes)
python -m pytest -q                       # run the test suite (offline, about 30 seconds)
```

The **Run update now** button on the Data Management page does the same as the first command.

### Hosting it online for free (optional)

1. Sign in at https://share.streamlit.io with your GitHub account.
2. Create an app from this repository, branch `main`, main file `streamlit_app.py`.

Every time the scheduled GitHub Action commits new data, Streamlit Community Cloud redeploys automatically.

---

## What the dashboard shows

| Page | Contents |
|---|---|
| **Home** | Next Grand Prix with live countdown, latest podium prediction, most likely exact podiums, championship tables, last race result (and how Apex's prediction did), model and data status |
| **Race Predictions** | Any upcoming race, four prediction stages, win / top-2 / podium / top-5 / top-10 probabilities, finishing-band chart, data-depth indicator, per-driver SHAP explanation, what drives the race's predictions, how the forecast moved across the weekend |
| **Driver Analytics** | Career numbers, results and weighted form, season table, rolling podium rate, qualifying positions and gap to pole, circuit record, team-mate head-to-heads, next-race prediction |
| **Constructor Analytics** | Points trend, driver comparison, mechanical reliability vs field, retirement reasons, race pace (2023+), qualifying trend |
| **Race Simulator** | 1,000–50,000 simulated races, full position-probability heatmap, most frequent podiums, scenarios (weather, safety car, reliability, grid drops, team pace), with estimated and hypothetical inputs clearly labelled |
| **Circuit Analytics** | Winners, overtaking difficulty, pole-to-win rate, retirements, safety-car rate, pit stops, weather, tyre degradation, driver and team suitability |
| **Model Performance** | Model vs benchmark comparison, season-by-season backtest, calibration, ROC, confusion matrices, SHAP feature importance, live prediction history and model version history |
| **Data Management** | Last update, record counts, source status, coverage by season, update log, failed requests, model registry, manual refresh |

---

<a id="data-sources"></a>
## Data sources and what they can and cannot do

I assessed every source in the brief before building. Findings:

| Source | Used for | Coverage | Access | Notes |
|---|---|---|---|---|
| **[Jolpica F1](https://github.com/jolpica/jolpica-f1)** (Ergast successor) | Calendar, circuits, results, qualifying (Q1–Q3 times), sprints, pit stops, standings | 1950+ (Apex stores 2014+, pit stops 2018+) | Free, no key | Limit ~4 requests/s and ~500/hour. A full backfill takes about an hour, so raw responses are cached and complete seasons are never re-downloaded. |
| **[OpenF1](https://openf1.org/)** | Lap times, stints and tyre compounds, trackside weather, race-control messages (safety cars, red flags), team colours | 2023+ | Free for historical data, no key | Live data *during* a session is now for paying users only, so Apex reads each session once it has finished. Limit ~30 requests/minute. There is no data before 2023, so race-pace, tyre and safety-car features are missing for older seasons (handled explicitly, see below). |
| **[FastF1](https://docs.fastf1.dev/)** | Not used | – | **Requires an F1 account** | FastF1's timing data comes from Formula 1's live-timing service, which now rejects unauthenticated requests (tested: HTTP 401, "No F1 account credentials available"). Because the brief rules out logins and keys, Apex uses OpenF1 for the same lap, sector, stint and tyre information. |
| **[Open-Meteo](https://open-meteo.com/)** | Race-window forecast for upcoming races; historical race-window weather (ERA5 archive) | Forecast: 16 days. Archive: 1940+ | Free, no key | Daily quota per IP address. Shared cloud IPs can hit it (it happened while building this), in which case Apex skips weather for that run and fills it in on later runs (at most 400 archive requests per run). |

Every stored row carries `source` and `retrieved_at` columns, every network request is written to `fetch_log`,
and the Data Management page shows coverage per season so gaps are visible rather than hidden.

**Nothing is invented.** If a value is not available it stays missing. Models handle missing values explicitly
(LightGBM natively; logistic regression and random forest via median imputation plus "was missing" indicator
columns), so a 2018 race simply has no lap-time features rather than made-up ones.

---

## How the predictive engine works

```
Jolpica ─┐                      ┌─ Stage 1 model ─┐
OpenF1 ──┼─► SQLite ─► features ├─ Stage 2 model ─┼─► calibrated podium ─► Plackett-Luce ─► win / P2 / podium /
Open-Meteo┘   (cached,   (leak-  ├─ Stage 3 model ─┤   probability          Monte Carlo      top-5 / top-10,
              upserts)   free)   └─ Stage 4 model ─┘   (+ SHAP)             (+ scenarios)    podium combos
```

### 1. Features (`apex/features.py`)

About 50 features per driver per race, in these groups:

* **Driver form**: exponentially weighted finishing position (half-life 4 races), last-5 average, weighted podium,
  win and points rates, career podium rate, starts, championship points and position before the race.
* **Qualifying pace**: weighted qualifying position and gap to pole (%) from Q1–Q3 times.
* **Team-mate comparison**: share of qualifying head-to-heads won, average finishing gap to team-mate.
* **Racecraft and reliability**: places gained from the grid, driver DNF rate, team mechanical DNF rate.
* **Constructor form**: weighted team finishing position, points, podium rate, qualifying, a development
  trend (last 3 races vs the 6 before) and team championship position. Renamed teams (e.g. Toro Rosso →
  AlphaTauri → RB) keep their history through a lineage map.
* **Race pace (2023+)**: median clean-lap gap to the fastest car, lap-time consistency, tyre degradation relative
  to the field, all from OpenF1 laps with safety-car, pit and traffic laps removed.
* **Circuit**: driver and team average finish at the venue, driver podium rate at the venue (shrunk towards
  their career rate), and circuit character learned from history: grid-to-finish rank correlation (overtaking
  difficulty), pole-to-win rate, DNF rate, safety-car rate, pit stops, wet-race rate. Circuit statistics are
  shrunk towards the all-circuit average when a venue has few races.
* **Weather skill**: each driver's finishing performance in wet races relative to dry ones, shrunk towards zero.
* **Weekend information** (only in later stages): practice best-lap and long-run gaps, practice degradation,
  qualifying position and gap, team-mate qualifying delta, sprint result, starting grid and grid penalty,
  race-window weather.

Recent races matter more than old ones because every form feature is an exponentially weighted average.

### 2. Four prediction stages

| Stage | Available information |
|---|---|
| 1 · Pre-weekend | History only: form, team, circuit, championship |
| 2 · Post-practice | + practice pace, long runs and degradation (OpenF1, 2023+) |
| 3 · Post-qualifying | + qualifying position and gap, sprint result if the sprint ran before qualifying |
| 4 · Pre-race | + starting grid, grid penalties, sprint result, race-window weather |

Each stage has its own model, trained only on the features that stage would have.

### 3. Models (`apex/models.py`)

For each stage Apex trains and compares:

* **Logistic regression**: the baseline ML model.
* **Random forest**: non-linear relationships and interactions.
* **LightGBM gradient boosting**: usually the strongest single tabular model, handles missing values natively.
* **Ensemble**: average log-odds of the two models with the best validation score.

Hyper-parameters are tuned on a small grid using a chronological hold-out (the last training season), and the
same hold-out fits **Platt scaling** so that a predicted 30% means roughly 30%. Finally, each race's podium
probabilities are rescaled (a per-race shift in log-odds) so they add up to exactly three: there are always
three podium places.

**Model selection is automatic and based on genuine out-of-sample testing**: the algorithm with the lowest mean
log loss across all backtest seasons is deployed for that stage. More complex models are not assumed to be better,
and at the pre-weekend stage the plain logistic regression has won.

### 4. From podium probabilities to everything else

See [Monte Carlo simulator](#monte-carlo-simulator). The calibrated podium probabilities fix each driver's
strength; win, P2, P3, top-5, top-10 and exact podium combinations come from simulating the race thousands of
times, so they are always mutually consistent.

---

## Preventing data leakage

* **Every historical aggregate is shifted by one race** before it is used (`_prior_ewm`, `_prior_mean`), so a row
  never contains its own result.
* **Team-level history is aggregated per race first.** The leakage test caught a subtle bug here during
  development: grouping driver rows by team let one car's result leak into its team-mate's "team history" for
  the same race. It is fixed, and the test now guards against it.
* **Stages only see what exists at that point of the weekend** (e.g. a sprint result is used at the
  post-qualifying stage only if the sprint ran before qualifying).
* **Weather**: actual race weather is only used at the pre-race stage, where a same-day forecast is a fair
  stand-in. Earlier stages use each circuit's historical rain rate instead.
* **Chronological validation only**: never random train/test splits.
* **Automated test** (`tests/test_features.py::test_no_leakage_from_current_or_future_races`): rebuilds the
  features for a race after deleting that race's result and every later race, and checks every feature is
  identical.

---

## Monte Carlo simulator

`apex/simulate.py` simulates races with a **Plackett-Luce** model: each driver has a log-strength θ, Gumbel
noise is added and the field is sorted (equivalent to drawing the winner in proportion to strength, then second
from the rest, and so on). Before that, each driver may retire with a probability based on their team's
mechanical DNF rate and their own DNF rate.

* **Linked to the ML model**: θ is solved iteratively so the simulated podium rate of every driver matches the
  model's calibrated podium probability (within simulation noise). The tests check this.
* **Internally consistent**: all probabilities come from the same simulated races, so P(win) ≤ P(top 2) ≤
  P(podium) ≤ P(top 5) ≤ P(top 10), and each position's probabilities sum to 100% across drivers.
* **Scenario parameters are estimated from data** by maximum likelihood on the out-of-sample backtest:
  how much more random wet races and safety-car races are, how much a driver's wet-weather record matters in the
  rain, how DNF rates change in the wet, and how many units of strength 1% of race pace is worth. When there is
  not enough history to estimate one of these (for example fewer than five wet races with weather data), the
  simulator says so and leaves that effect neutral rather than guessing.
* **Hypothetical inputs are labelled**: a team pace change or DNF multiplier you choose is an assumption. A grid
  change is also your assumption, but its effect is estimated: the stage model is re-run with the new grid slot.

---

## Explainability

`apex/explain.py` computes **SHAP values** for each live prediction with `shap.PermutationExplainer` on the
calibrated log-odds, which works for every model type including the ensemble. Each value is how much one input
moved that driver's podium log-odds away from an average driver; they add up exactly to the prediction.

The written explanation is generated from those numbers: SHAP values are summed by feature group (recent form,
qualifying, constructor form, circuit suitability, reliability, …) and the strongest feature in each group is
quoted with the driver's actual value. Summing by group matters because correlated inputs (team points and team
qualifying record, say) can split credit with opposite signs; individual bars are still shown in the chart.
No language model is involved.

Global importance (mean |SHAP| on the latest season) and the share of importance by group ("how much does
qualifying matter compared with recent form?") are on the Model Performance page.

---

## Validation and results

**Expanding-window backtest**: for each season from 2018 onwards, models are trained only on earlier seasons
(2014 to the previous year) and then predict every race of that season. Metrics: accuracy, precision, recall,
F1 (50% threshold), Brier score, log loss, ROC-AUC, calibration curves, confusion matrices, and **podium hit
rate** (share of each race's three highest-probability drivers who actually made the podium).

**Benchmarks** are fitted the same way (lookup tables of historical podium rate by rank):
qualifying position, recent finishing positions (last 5), and constructor championship position.

The Model Performance page shows the current figures for every stage and season. The snapshot from the first
full training run (2018–2026 backtest, 193 races) is in the pull request description. Headline:

* **Pre-race stage**: the ensemble beats the qualifying-position benchmark, but **only by about 5% in Brier
  score**, and its podium hit rate is about the same (~66–67%). Qualifying already contains most of the
  predictable signal; the model's main gain is better-calibrated probabilities further down the order.
* **Pre-weekend stage** is where the model adds the most over simple benchmarks, because there is no qualifying
  result to lean on yet.

These are reported as they are. The app flags it on the performance page whenever the best model is less than
3% better than the strongest benchmark.

**Prediction history**: every live prediction (date, stage, model version, inputs, every driver's probabilities
and explanation) is inserted into `prediction_runs` / `predictions` and never overwritten, even after
retraining. Once the race result is in, the Model Performance page scores it.

---

## Automatic updates

`.github/workflows/update.yml` runs every six hours (and on demand from the Actions tab, with an optional
"force retrain" switch):

1. Runs the test suite.
2. `python -m apex.pipeline update`:
   * **Detects new races** by comparing the Jolpica calendar and results with the database.
   * **Provisional results**: a result is `provisional` until 72 hours after the race (stewards' decisions and
     penalties normally settle by then). Provisional races are re-downloaded on each run and are not used for
     training until they become `final`.
   * Downloads qualifying, sprint, pit stops, OpenF1 session data for every finished session, and weather.
     Complete past seasons are skipped; upserts on natural keys mean nothing is duplicated.
   * **Retrains only when a race result has newly become final** (or no model exists): full backtest, model
     selection, simulator parameter estimation, SHAP importance, then a new model version is saved and registered.
     Previous model files are kept (last three on disk, all in git history) and the registry keeps every version.
   * Predicts the next race at every stage whose inputs exist, storing each new prediction.
   * Logs the run (`update_log`), including failures. An API failure is recorded and the run carries on with
     cached data; a crash leaves the previous data and models untouched.
3. Commits `data/apex.db` and `models/` **only if the data actually changed**, so quiet runs add no commits.

A second workflow runs the tests on every pull request.

> GitHub pauses scheduled workflows in repositories with no activity for 60 days. The bot's own commits after
> each race normally count as activity; if it is ever paused, re-enable it from the Actions tab.

---

## Project structure

```
apex/
  config.py          paths and constants
  db.py              SQLite schema and helpers (portable SQL, PostgreSQL-ready)
  sources/
    http.py          cached, rate-limited, logged HTTP client with retries and stale-cache fallback
    jolpica.py       Jolpica F1 client and parsers
    openf1.py        OpenF1 client: lap/stint/weather/race-control summaries
    weather.py       Open-Meteo forecast and archive
  ingest.py          incremental sync of all sources into the database
  features.py        leakage-safe feature engineering and stage definitions
  models.py          models, calibration, backtesting, benchmarks, selection, persistence
  simulate.py        Plackett-Luce Monte Carlo simulator and scenario parameter fitting
  explain.py         SHAP values and generated explanations
  predict.py         live predictions and prediction history
  pipeline.py        command-line entry point for updates and retraining
app/
  ui.py              shared styling, cached queries, chart theme
  views/*.py         the eight dashboard pages
streamlit_app.py     dashboard entry point
tests/               pytest suite on a synthetic database (offline)
data/apex.db         the database (committed so the app works immediately)
models/              trained models (joblib) and active.json pointer
.github/workflows/   scheduled update and CI test workflows
```

### Moving to PostgreSQL later

All queries go through `apex/db.py` and use plain SQL. Switching means changing `connect()` to a PostgreSQL
driver and replacing `INSERT OR REPLACE` in `upsert()` with `INSERT … ON CONFLICT … DO UPDATE`.

---

## Known limitations

* **Small samples.** About 22 races a season and 3 podium places each: around 600 podium finishes across the
  backtest. Probabilities are useful, but no model will be right most of the time on exact podiums.
* **Regulation resets.** Big rule changes (2014, 2017, 2022, 2026) reshuffle the order. Form features adapt within
  a few races, but early-season predictions after a reset are the least reliable.
* **Lap data starts in 2023.** Race pace, consistency, degradation, practice pace and safety-car rates exist only
  from 2023, so models learn less from them than from results-based features. The post-practice stage in
  particular has few seasons to learn from.
* **Weather history depends on Open-Meteo's quota.** Until the archive backfill completes, wet races are only
  identified from OpenF1 (2023+), and the simulator's rain parameters stay neutral if there are too few.
* **Starting grid.** The official grid is published with the result, so before the race the pre-race stage
  uses qualifying order (which already reflects most penalties). Late penalties are not captured automatically;
  use the simulator's grid-change control.
* **Rookies and new teams** (e.g. Cadillac in 2026) have little history; their probabilities lean on team form.
  The "Data depth" column flags this.
* **SHAP with correlated features** can split credit between similar inputs in surprising ways; explanations are
  grouped by category for that reason.
* **Repository size.** The database is committed after each change, which grows git history by a few megabytes per
  race weekend. That is fine for years; if it ever matters, move the database to a release asset.
* **Streamlit Community Cloud** storage is temporary: a manual refresh there lasts until the app restarts. The
  scheduled GitHub Action is the durable update path.

---

## Methodology notes

* **Why podium as the primary target?** It balances signal and class balance: wins are dominated by one or two
  cars (very imbalanced), top-ten is much noisier. Podium probabilities then anchor the simulator for all other
  targets.
* **Why log loss for model selection?** It rewards honest probabilities, not just ranking. Accuracy is misleading
  here: predicting "no podium" for everyone is ~86% accurate.
* **Why calibrate, then normalise per race?** Platt scaling fixes systematic over/under-confidence learned on a
  held-out season; normalising to three podium places per race uses a hard constraint the classifier doesn't know.
* **Why Plackett-Luce?** It is the standard probabilistic model for rankings, has a single strength parameter per
  driver, and sampling it is equivalent to "Gumbel noise + sort", which makes 50,000 simulated races fast in NumPy.
* **Why exponential weighting?** A five-year-old result should count for something but much less than last
  week's. Half-lives (4–10 races depending on the feature) were chosen so form reacts within a few races while
  slower-moving traits (reliability, team-mate record) are smoothed more.
* **Why shrinkage?** Circuit-specific records rest on a handful of races. Shrinking towards each driver's overall
  record (or the all-circuit average) stops one lucky result dominating.
