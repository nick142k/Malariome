# Issues and Fixes

## ISS-001 — GitHub folder pages not accessible to automated fetch
- **Date:** 2026-10-03
- **Problem:** `src/`, `models/`, `utils/` listings could not be fetched.
- **Cause:** GitHub robots restrictions on tree pages.
- **Resolution:** User uploaded the repository ZIP.
- **Files changed:** none
- **Status:** RESOLVED
- **Lesson:** Use a ZIP/clone for repository analysis.

## ISS-002 — Reference repository has no license
- **Date:** 2026-10-03
- **Problem:** No LICENSE file found in the repo root.
- **Cause:** Not provided by the original author.
- **Resolution:** No reference code will be copied; repo credited as reference in README/docs. Consider asking the author about licensing if reuse is ever wanted.
- **Files changed:** none
- **Status:** OPEN (mitigated)
- **Lesson:** Check licensing before reusing code.

## ISS-003 — Wrong correction of dataset count in analysis doc
- **Date:** 2026-10-03
- **Problem:** After the user counted 13,780 files per class I wrote that the notebook's 13,779 was unexplained/off.
- **Cause:** Counted files, not images; each class folder contains one non-image .db file.
- **Resolution:** verify_dataset.py separates images from non-image files; doc corrected.
- **Files changed:** docs/REFERENCE_ANALYSIS.md
- **Status:** RESOLVED
- **Lesson:** Count by file type, not by directory listing.

## ISS-004 — Random split would leak across smear images
- **Date:** 2026-10-03
- **Problem:** 936 of 1,408 smear groups contain cells from both classes; the reference repo splits per class at random.
- **Cause:** Many cell crops come from the same smear image.
- **Resolution:** Group-aware stratified split implemented in src/data/splitter.py; verified on synthetic data only.
- **Files changed:** src/data/splitter.py
- **Status:** IN PROGRESS (awaiting run on real data)
- **Lesson:** Split by source image/patient, not by crop.
