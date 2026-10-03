# MalariaAI — Project Progress Dashboard

**Last updated:** 2026-10-03
**Overall completion:** 9%
**Current phase:** Phases 1-2 complete → Phase 3 remainder (exploration notebook + group-aware split)
**Current status:** IN PROGRESS

## Weighted milestones

| Milestone | Weight | Earned | Status | Note |
|---|---:|---:|---|---|
| Repository / Dataset | 10 | 8 | IN PROGRESS | Reference repo analysed; dataset verified (counts, corruption, duplicates, sizes). Remaining: exploration notebook figures, group-aware split manifest |
| Preprocessing | 10 | 1 | IN PROGRESS | Design decided and config loader verified; pipeline code not written |
| Custom CNN | 10 | 0 | NOT STARTED | |
| Training | 10 | 0 | NOT STARTED | |
| Transfer Learning | 10 | 0 | NOT STARTED | |
| Evaluation | 10 | 0 | NOT STARTED | |
| Explainability | 10 | 0 | NOT STARTED | |
| Backend / API | 10 | 0 | NOT STARTED | |
| Web Application | 10 | 0 | NOT STARTED | |
| Testing / Deployment / Documentation | 10 | 0 | NOT STARTED | |
| **Total** | **100** | **9** | | |

## Phases

| Phase | Status |
|---|---|
| 1 Repository analysis | COMPLETED (dataset-level facts still NOT VERIFIED) |
| 2 Architecture | COMPLETED (docs/ARCHITECTURE.md, config.yaml, requirements.txt, .gitignore, skeleton) |
| 3–22 | NOT STARTED |

## Achievements
- Dataset verified on the user's machine: 27,558 RGB PNG images, balanced 13,779/13,779, 0 corrupted, 0 exact duplicates.
- Leakage risk quantified: 1,408 smear groups, 936 contain both classes -> group-aware split required.
- Reference repo fully reviewed from the supplied ZIP; `docs/REFERENCE_ANALYSIS.md` written.

## Current problems
- Image sizes vary widely (46-394 px); resizing choice affects both models.
- Reference repo has no license (see ISSUES_AND_FIXES.md, ISS-002).

## Next milestone
Phase 2 — finalize architecture, config schema and folder structure.

## Next action
Parse patient IDs, build the group-aware split manifests, add the exploration notebook.
