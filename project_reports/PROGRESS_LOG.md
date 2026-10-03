# Progress Log

## 2026-10-03 — Phase 1: Repository analysis
- Public GitHub front page and `requirements.txt` read; folder pages blocked by GitHub robots rules.
- User supplied the repo ZIP; all scripts and both notebooks' recorded outputs reviewed (not executed).
- Created `docs/REFERENCE_ANALYSIS.md`.
- Initialized `project_reports/` (PROJECT_PROGRESS, PROGRESS_LOG, ACHIEVEMENTS, MILESTONES, ISSUES_AND_FIXES) and evidence folders.
- Result: Phase 1 COMPLETED; overall progress 5%.

## 2026-10-03 — Phase 3 (partial): dataset verification
- Added scripts/verify_dataset.py (tested on synthetic data only before the user ran it).
- User ran it on the real dataset; output recorded in evidence/dataset/verify_dataset_output.txt and metrics/dataset_metrics.json.
- Corrected REFERENCE_ANALYSIS: 13,779 images/class (13,780 files incl. one .db each). An earlier edit this day wrongly said the notebook was off; fixed.
- Overall progress 8%.

## 2026-10-03 — Phase 2: Architecture
- Decisions recorded: 224x224, Colab/Kaggle training, EXP-000 included (user accepted defaults).
- Created docs/ARCHITECTURE.md, config.yaml, src/config.py, requirements.txt (version ranges, unverified install), .gitignore, folder skeleton.
- Validation: config loader run once, returned expected values. No other code exists yet.
- Overall progress 9%.

## 2026-10-03 — Phase 3: group-aware splitter written
- Added src/data/splitter.py (StratifiedGroupKFold, 2 stages, seed 42; patient_id grouping with smear_id fallback; hard asserts for group overlap/coverage).
- Tested only on synthetic data shaped like the real filenames (3,426 fake files): no group overlap, splits 72.8/13.8/13.4 %. NOT YET RUN on the real dataset.
- Overall progress unchanged at 9% until it is run on real data.
