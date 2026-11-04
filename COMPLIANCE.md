# COMPLIANCE.md — Assignment requirements vs evidence

Status key: ✅ met and verified by execution · ⚠️ met with a stated limitation.
"Verified" means the code ran end-to-end (`python run_pipeline.py`, exit 0, ~12 min) and the
output was inspected; 41 automated tests pass (`python -m pytest`).

## Dataset

| Requirement | Where | Evidence | Status |
|---|---|---|---|
| Use REFIT raw + cleaned data; consult documentation | `REFIT_RAW_081116/`, `CLEAN_REFIT_081116/`, `references/CLEAN_READ_ME_081116.txt` | channel mapping taken from the readme; readme's claims (DST correction, 4 kW spike removal, Part1/2 conventions) checked against the data | ✅ |
| State households/files, date coverage, sampling interval, units, limitations | DATA.md §1, RESULTS.md §1–2 | tables of files, periods, 8 s change-logged sampling, W | ✅ |

## 1. Raw data inspection and cleaning (House 1)

| Requirement | Where | Evidence | Status |
|---|---|---|---|
| Raw data, House 1, resampled to 1 min | `raw_cleaning.py`, `stage_raw.py` | `outputs/data/house1_raw_cleaned_1min.parquet`, 920,031 minutes | ✅ |
| Timestamp parsing & ordering | `inspect_timestamps`, `local_to_utc` | 0 parse failures; 1 backward jump identified as DST fall-back; tests `test_local_to_utc_*` | ✅ |
| Duplicate timestamps | `sort_and_deduplicate` | 508,772 exact duplicates dropped; 87,000 conflicting timestamps merged by stated rule; test `test_sort_and_deduplicate_rules` | ✅ |
| Missing readings & irregular intervals | `task1_timestamp_checks.csv`, `task1_gaps_by_channel.csv` | intervals, gap counts, per-channel NaN counts reported | ✅ |
| Impossible / extreme values | `apply_validity_rules`, `count_isolated_spikes` | glitch values nulled per channel; genuine 10–12 kW and 2.3 kW peaks kept; ambiguous spikes counted, not removed | ✅ |
| Summarise findings | DATA.md §2 | 5 findings incl. Part1/Part2 overlap disagreement | ✅ |
| Documented pipeline: sort, de-dup rule, short gaps, flag long outages | DATA.md §3, `raw_cleaning.py` docstrings | ≤ 5-min gaps filled (893 min), 381 outages flagged (`task1_outages.csv`), never filled | ✅ |
| Before/after plots: aggregate + ≥ 2 appliances | `task1_before_after_day.png`, `task1_daily_max_before_after.png` | Aggregate, Washing Machine, Fridge, Dishwasher | ✅ |
| (extra) independent validation | `task1_vs_official_clean_monthly.csv` | r ≥ 0.99 vs official data in GMT months; exposed official time-base defect | ✅ |

## 2. Washing-machine EDA (official cleaned data)

| Requirement | Where | Evidence | Status |
|---|---|---|---|
| ≥ 4 households at 1 min with a WM channel; state homes & periods | RESULTS.md §1 | 9 homes with periods; all 19 WM homes screened (`task2_wm_channel_screen.csv`) | ✅ |
| Availability & quality: date range, missing intervals, zero/off time, number of cycles | `task2_availability.csv`, RESULTS.md §2 | per-house table | ✅ |
| Simple, reasonable cycle rule in plain language; threshold stated | `cycles.py`, RESULTS.md §3 | 20 W / 10-min merge / ≥ 15 min / ≥ 50 Wh / peak ≥ 300 W / ≤ 6 h; derived from data; sensitivity ±4 %; 7 tests | ✅ |
| Usage by hour of day and day of week | `task2_usage_patterns.png`, `task2_hour_of_day_pct.csv`, `task2_weekday_pct.csv` | local time after time-base repair | ✅ |
| Compare ≥ 1 cycle characteristic across homes | `task2_cycle_characteristics.csv`, `task2_cross_house.png` | duration, energy, peak, heating share, cycles/week | ✅ |
| Plot (i) 24-h aggregate + WM trace | `task2_24h_trace.png` + interpretation | ✅ |
| Plot (ii) detected cycle(s) | `task2_example_cycles.png` + interpretation | ✅ |
| Plot (iii) cross-house comparison | `task2_cross_house.png` + interpretation | ✅ |

## 3. Disaggregation model

| Requirement | Where | Evidence | Status |
|---|---|---|---|
| Official cleaned data, 1 min, WM power from aggregate; homes & periods stated | `stage_model.py`, RESULTS.md §6.1–6.2 | target = WM W; inputs = aggregate-only features (test `test_features_shape_and_no_target_columns`) | ✅ |
| Appropriate, explainable model | `models.py` | LightGBM hurdle (classifier × regressor), selected on validation | ✅ |
| ≥ 1 simple baseline and comparison | `models.py`, `task3_metrics.csv` | always-off, constant mean, weekday×hour usage profile | ✅ |
| Sensible split avoiding leakage, explained | RESULTS.md §6.2 | household hold-out (train 6 / val 1 / test 2); test `test_config_loads_and_resolves_relative_paths` asserts disjoint splits; causal-feature test | ✅ |
| Appropriate metrics, explained | `metrics.py` docstring, RESULTS.md §6.4 | MAE, RMSE, MAE-on, energy error, daily energy MAE, P/R/F1, cycle recall/precision; always-off caveat shown with numbers | ✅ |
| Actual-vs-predicted plot on evaluation data | `task3_actual_vs_predicted.png` | unseen House 8 | ✅ |
| 1–2 failure cases | `task3_failure_cases.png`, RESULTS.md §6.7 | phantom load (overlapping signatures, washer-dryer), missed night-time back-to-back washes | ✅ |
| (extra) ablations | `task3_metrics.csv` | causal-only features; seen-house temporal hold-out | ✅ |
| Test set size | — | only two test houses; no confidence interval | ⚠️ |

## 4. Practical implications (Recommendation.md)

| Requirement | Evidence | Status |
|---|---|---|
| One realistic opportunity + how to measure it | "cooler and later" insight card; randomised pilot with smart-plug sub-sample, pre-registered outcomes | ✅ |
| Limitations: overlapping signatures, missing data, household differences, UK→India | dedicated sections, each tied to measured results | ✅ |
| Paragraph on 15/30-minute data | final section (reasoned, explicitly not measured) | ⚠️ not quantified experimentally |

## Research guidance

| Requirement | Evidence | Status |
|---|---|---|
| Cite articles, say which ideas were adopted, implement own work | README "References": Murray 2017; Kelly & Knottenbelt 2015 (activation rule); Zhang 2018 (centred windows, house set); Precioso & Gómez-Ullate 2020 (classification + regression). No published numbers claimed. | ✅ |

## Deliverables

| File | Status |
|---|---|
| Source code running end-to-end (`run_pipeline.py`, `src/wattwise/`) | ✅ verified full run, exit 0 |
| README.md (build, setup/run, assumptions, preprocessing, validation, trade-offs, AI tools, improvements) | ✅ |
| DATA.md | ✅ |
| RESULTS.md | ✅ |
| Recommendation.md (one-page summary + Section 4 answers) | ✅ (dense; about one printed page plus the Section 4 answers) |
| Public GitHub repository | ⚠️ not created by the pipeline; the folder is ready to publish (`.gitignore` excludes the 14 GB of REFIT data and large intermediates) |

## Reproducibility checks performed

* Two full runs produced identical metrics, cycle counts and failure days (fixed seeds).
* All paths are relative to `config.yaml`; no machine-specific paths in the code.
* `requirements.txt` pins the versions that were executed (Python 3.13.0, Windows 11).
* Original REFIT folders are read-only inputs; nothing is written into them.
