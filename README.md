# WattWise NILM — REFIT data cleaning and washing-machine disaggregation

Flock Energy data-science take-home. A reproducible Python pipeline that

1. **cleans raw REFIT data for House 1** (timestamps, DST, duplicates, impossible values,
   1-minute resampling, short-gap filling, outage flagging) and checks the result against the
   official cleaned release;
2. **analyses washing-machine use in 9 households** from the official cleaned data (data quality,
   a documented cycle-detection rule, hour/weekday patterns, cross-house comparison);
3. **estimates washing-machine power from the household aggregate** at 1-minute resolution with
   LightGBM, compared against three baselines on households the model never saw;
4. turns the results into a **utility-facing recommendation**.

| Deliverable | Content |
|---|---|
| [DATA.md](DATA.md) | raw-data coverage and quality findings, cleaning steps, gap/outlier decisions, remaining concerns |
| [RESULTS.md](RESULTS.md) | washing-machine EDA, plots, model results, validation, failure analysis |
| [Recommendation.md](Recommendation.md) | one-page utility summary + Section 4 answers (UK→India, 15/30-min data) |
| [COMPLIANCE.md](COMPLIANCE.md) | requirement-by-requirement check of the assignment against the evidence |
| `src/wattwise/`, `run_pipeline.py`, `tests/` | source code, entry point, 41 automated tests |

## Headline results

* **Data defects found and fixed**: raw timestamps are UK wall-clock time (proved by the DST
  fall-back jump); Part1/Part2 overlap for 95 days and disagree; 509k exact duplicate rows;
  bit-pattern glitches (8192, 32767, 65535 W). The official cleaned files are **still on BST
  wall-clock time on 25–60 % of summer days** in every house used — we detect and repair this
  before any hour-of-day analysis (DATA.md §5).
* **3,419 washing-machine cycles** in 9 homes. Water heating is **~80 % of washing energy**; 16 %
  of washes start in the 16:00–19:00 peak; energy per wash differs 2.6× between homes.
* **Model (LightGBM hurdle: on/off classifier × power regressor)** on two unseen test homes:
  on/off F1 0.57, 75 % of washes found, daily energy MAE 0.25 kWh vs 0.34 for always-off and 0.44
  for a usage-profile baseline. Always-off has the best per-minute MAE (15.2 W vs 16.5 W) — a
  reminder that MAE alone is misleading for a sparse appliance. Accuracy is strongly
  household-dependent (energy error −35 % and +92 % in the two test homes).

## Setup (Windows, VS Code terminal)

Requires Python 3.10+ (tested with 3.13.0), ~8 GB RAM, ~2 GB free disk for outputs.

```powershell
# from the project root (the folder containing this README)
py -3.13 -m venv .venv                 # or: python -m venv .venv
.\.venv\Scripts\Activate.ps1           # Git Bash: source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

### Data

Download and extract (7-Zip) the two archives into the project root so that the folders sit
next to this README — the code finds them via `config.yaml → paths`:

| Folder | Source |
|---|---|
| `REFIT_RAW_081116/` (40 CSVs, 7.8 GB) | <https://pureportal.strath.ac.uk/files/62090177/REFIT_RAW_081116.7z> |
| `CLEAN_REFIT_081116/` (20 CSVs, 6.5 GB) | <https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z> |

The official readme is included at [references/CLEAN_READ_ME_081116.txt](references/CLEAN_READ_ME_081116.txt).
The source folders are only ever read.

## Run

```powershell
python run_pipeline.py                     # everything (~12 min on a 16-thread laptop)
python run_pipeline.py --stage raw         # Task 1 only (~40 s)
python run_pipeline.py --stage prepare eda # Task 2 (needs raw files for the time-base audit)
python run_pipeline.py --stage model       # Task 3 (needs 'prepare' first)
python -m pytest                           # 41 tests on synthetic fixtures (~5 s)
```

| Stage | Reads | Writes |
|---|---|---|
| `raw` | raw House 1 parts | `outputs/data/house1_raw_cleaned_1min.parquet`, `outputs/tables/task1_*`, `outputs/figures/task1_*` |
| `prepare` | cleaned + raw files of the 9 houses | `outputs/data/clean_house{h}_1min.parquet`, `outputs/tables/timebase_*` |
| `eda` | prepared caches (+ all 19 cleaned files for the screen) | `outputs/tables/task2_*`, `outputs/figures/task2_*` |
| `model` | prepared caches | `outputs/tables/task3_*`, `outputs/figures/task3_*`, test predictions |

Logs go to `outputs/logs/pipeline.log`. All thresholds, houses and splits live in
[config.yaml](config.yaml). Results are deterministic (fixed seeds).

## Code layout

```
config.yaml              every path, threshold, house list and split
run_pipeline.py          entry point (adds src/ to the path; no install needed)
src/wattwise/
  config.py              config loading, relative paths, logging
  io.py                  schema-validated readers for raw/cleaned CSVs
  raw_cleaning.py        Task 1 pipeline steps (each returns counts for the log)
  stage_raw.py           Task 1 outputs, figures, comparison with official data
  timebase.py            audit/repair of the official files' UTC vs BST time base
  clean_data.py          official cleaned data -> 1-min frames, frozen-sensor flag
  stage_prepare.py       builds the 1-min caches
  cycles.py              washing-machine cycle detection rule
  eda.py, stage_eda.py   Task 2 tables and figures
  features.py            aggregate-only features (centred or causal)
  models.py              baselines, LightGBM regression and hurdle model
  metrics.py             power, energy, on/off and cycle-level metrics
  stage_model.py         Task 3 training, selection, evaluation, ablations, figures
  plotting.py            shared figure style
tests/                   unit + end-to-end tests on synthetic data
references/              official dataset readme
outputs/                 generated (figures/tables are committed; data/logs are git-ignored)
```

## Key decisions and assumptions

**Preprocessing (Task 1, details in DATA.md)**
* `Unix` is the clock of record; it is UK local time and is converted to UTC with explicit DST
  handling (the unrecoverable interleaved hour on 2014-10-26 is dropped, 986 rows).
* Part1/Part2 overlap: Part2 wins (NaN-aware and ~3× fewer physical violations).
* Duplicates: exact duplicates dropped; conflicting same-timestamp rows → last non-null value per channel.
* Impossible values → NaN (not 0): aggregate 0 W or > 20 kW, plug monitors > 4 kW, negatives.
  Genuine peaks (10–12 kW showers, 2.3 kW heaters) are kept.
* 1-minute mean; gaps ≤ 5 min forward-filled (data are logged on change); longer gaps stay NaN and
  are listed as outages.

**EDA (Task 2)** — official cleaned data, time base repaired; frozen non-zero WM values (≥ 60 min
exactly constant) excluded; cycle = ≥ 20 W with ≤ 10-min pauses, ≥ 15 min, ≥ 50 Wh, peak ≥ 300 W,
≤ 6 h. Thresholds were derived from the data and checked by a sensitivity analysis (±4 %).

**Validation (Task 3)** — split by household: train 2, 5, 7, 9, 15, 16 · validation 17 · test 8, 18.
No house appears in two splits, so overlapping windows cannot leak; the threshold, early stopping
and model choice use validation only; the test houses were evaluated once. Selection metric:
validation daily-energy MAE (the closest to a customer energy breakdown).

**Assumption: offline disaggregation.** Features use a centred window (minutes before and after t),
appropriate for analysing stored meter data. A causal ablation reports the cost of real-time use
(F1 0.57 → 0.48).

## Trade-offs

* **Gradient boosting on hand-built window features rather than a neural seq2point model**:
  trains in minutes on a laptop with 8 GB RAM, is easy to inspect (feature importance), and the
  brief rewards a simple, well-validated model. A CNN would likely learn the cycle shape better.
* **Hurdle model over plain regression**: regression found more washes (cycle recall 0.88) but
  predicted ~3 cycles per real one; the hurdle model trades some recall for far fewer false alarms
  and the best energy error.
* **Training-row sub-sampling** (all minutes near activity + 25 % of the rest, weighted ×4) keeps
  memory bounded without biasing the target distribution.
* **Repairing the official time base** costs ~40 s of raw-file reading but avoids hour-of-day
  results that are off by one hour on many summer days.
* **Two test houses** keep the test honest and cheap but cannot give a confidence interval;
  leave-one-house-out cross-validation is the next step.

## Possible improvements

1. Per-household calibration (biggest measured gain: a model that has seen the house halves the
   on-power error).
2. Leave-one-house-out CV over more houses (3, 10, 13 screened as usable) for confidence intervals;
   test removing hour-of-day features, which encode training-house habits.
3. A seq2point CNN under the same split and metrics; event-level post-processing that enforces the
   heat → tumble → spin template.
4. An experiment on 15/30-minute resampled data to quantify the resolution trade-off discussed in
   Recommendation.md.

## AI tools used

This project was developed with **Claude Code (Anthropic, Claude Opus 5.5)** used as an AI
pair-programmer inside VS Code: it inspected the dataset, proposed and implemented the pipeline,
tests and documentation, ran every stage, and investigated anomalies (DST handling, Part1/Part2
overlap, the official files' time base). Every number in the documents comes from executed code
in this repository; the design decisions are explained so they can be reviewed independently.

## References

* Murray, D., Stankovic, L., Stankovic, V. (2017). *An electrical load measurements dataset of
  United Kingdom households from a two-year longitudinal study.* Scientific Data 4, 160122.
  <https://doi.org/10.1038/sdata.2016.122> — dataset description, sampling, collection issues.
* REFIT cleaned-data readme (University of Strathclyde), included in `references/` — channel
  mapping, Part1/Part2 conventions, official cleaning steps.
* Kelly, J., Knottenbelt, W. (2015). *Neural NILM: Deep Neural Networks Applied to Energy
  Disaggregation.* BuildSys '15. <https://arxiv.org/abs/1507.06594> — **adopted**: defining
  appliance activations with a power threshold plus minimum on/off durations (values re-derived
  for REFIT).
* Zhang, C., Zhong, M., Wang, Z., Goddard, N., Sutton, C. (2018). *Sequence-to-point learning with
  neural networks for non-intrusive load monitoring.* AAAI-18. <https://arxiv.org/abs/1612.09106> —
  **adopted**: predicting the appliance's power at a point from a window of aggregate centred on it;
  the REFIT washing-machine house set (as listed in the public implementation
  <https://github.com/MingjunZhong/transferNILM>). We do not reproduce their network or their results.
* Precioso, D., Gómez-Ullate, D. (2020). *NILM as a regression versus classification problem: the
  importance of thresholding.* <https://arxiv.org/abs/2010.16050> — **adopted**: combining an on/off
  classifier with power regression, and reporting both regression and classification metrics.

Published results in these papers are not comparable to ours (different splits, resolution and
pre-processing) and are not claimed here.

Dataset licence: CC-BY 4.0 (REFIT, University of Strathclyde).
