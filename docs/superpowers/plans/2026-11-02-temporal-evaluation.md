# Chronological evaluation implementation plan

> Historical implementation plan: executed inline before this local analysis folder was connected to Git. On 2026-11-02 the user requested Git initialization and publication; current repository state supersedes the original no-Git workflow.

**Goal:** Measure unseen-future accuracy of the existing 50-tree LightGBM ensemble.

**Architecture / specification:** Reuse official features and existing hyperparameters. Train from 2023-06-01 01:00 KST through 2024-05-31 00:00 KST (exclusive cutoff 2024-05-31 01:00). Hold out 2024-06-01 01:00 through 2025-06-01 00:00 KST. The 24-hour embargo avoids using labels unavailable at the first day-ahead forecast. Freeze both models throughout the holdout year; no tuning or early stopping on this period.

**Tech Stack:** Existing Python, pandas, NumPy, LightGBM, standard-library unittest.

## Global constraints

- Preserve official data and previous training outputs.
- Score raw, unclipped held-out labels; retain the existing training-target clipping only during fitting.
- Compare round 10, round 17, their equal-weight ensemble, and training-only site/month/hour climatology with training-only fallbacks.
- Missing labels are absent observations, never zeros. Score sites on observed labels; headline team metrics require every assigned site to be observed at that hour. Also report coverage and partial-observation metrics separately.
- Report MAE, RMSE, mean capacity-normalized absolute error, WAPE, and signed bias, with separate daytime metrics.
- Apply historical SMP and hour-based k to complete team-hours for a historical score proxy, never an official competition score.
- Report monthly results and save predictions, models, parameters, and split evidence.

## Review focus

- Split boundaries and first forecast issuance: unit test exact endpoints and embargo.
- Missing member labels: unit test exclusion from complete-team metrics.
- Nighttime zeros and zero denominators: unit test finite metrics/null ratios.
- Outliers: unit test scoring without label clipping.
- Unseen climatology keys: unit test training-only fallback.

## Tasks

- [x] Write `test_temporal.py` with hand-calculated fixtures; confirm the missing implementation fails.
- [x] Implement `evaluate_temporal.py` with `split_data`, `metrics`, `aggregate_groups`, `climatology_predict`, and a CLI runner; run all unit tests.
- [x] Run the full chronological evaluation into a new `outputs/temporal_*` folder.
- [x] Independently recompute headline metrics and verify chronology, saved models and output keys; document the results in README and the output summary.

## Progress

Protocol selected before inspecting holdout errors. The user authorized temporal evaluation; period and metric choices are routine analytical decisions within that scope.

Execution completed in `outputs/temporal_20261102_200911_C859iM`. Five unit tests pass; independent raw-label, metric, completeness, chronology and saved-model checks pass. A fresh reviewer found no correctness issues. Headline caveats: 97.94% observed site coverage, 66.16% complete group-hour coverage, 48 unseen sites.

The independent check initially assumed 12 complete-group months; data show no complete group-hours in June 2024. The checker was corrected to compare monthly keys against actual completeness rather than require unsupported metrics. Production evaluation logic did not change.
