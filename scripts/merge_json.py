"""Merge experiment records from a downloaded training_metrics.json (e.g. from Kaggle) into the local one.
Existing local experiments are never overwritten; only new experiment IDs are added.

Usage (project root):  python scripts\\merge_json.py "C:\\path\\to\\downloaded\\training_metrics.json"
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "project_reports" / "metrics" / "training_metrics.json"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    if not src.is_file():
        sys.exit(f"File not found: {src}")
    new = json.loads(src.read_text(encoding="utf-8"))
    old = json.loads(DST.read_text(encoding="utf-8")) if DST.exists() else {}
    added, kept = [], []
    for key, value in new.items():
        if key in old:
            kept.append(key)
        else:
            old[key] = value
            added.append(key)
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(old, indent=2), encoding="utf-8")
    print("added:", added, "| already present locally (kept as is):", kept)
    print("local file now has:", sorted(old))


if __name__ == "__main__":
    main()
