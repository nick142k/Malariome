"""Plot training curves from a history CSV written by train.py.

Usage (project root, venv active):   python scripts/plot_history.py EXP-001
Reads  reports/metrics/<exp>_history.csv
Writes reports/figures/<exp>_curves.png and project_reports/evidence/training/<exp>_curves.png
"""
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main(exp_id):
    tag = exp_id.lower().replace("-", "")
    csv = ROOT / "reports" / "metrics" / f"{tag}_history.csv"
    if not csv.exists():
        sys.exit(f"History file not found: {csv}")
    df = pd.read_csv(csv)
    x = df["epoch"] + 1
    best = int(df["val_auc"].idxmax()) + 1
    panels = [("Loss", ["loss", "val_loss"]), ("Accuracy", ["accuracy", "val_accuracy"]), ("ROC-AUC", ["auc", "val_auc"]),
              ("PR-AUC", ["pr_auc", "val_pr_auc"]), ("Validation precision / recall", ["val_precision", "val_recall"]),
              ("Learning rate", ["learning_rate"])]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (title, cols) in zip(axes.ravel(), panels):
        for c in cols:
            if c in df:
                ax.plot(x, df[c], marker="o", label=c)
        ax.axvline(best, color="gray", linestyle="--", linewidth=1)
        ax.set_title(title); ax.set_xlabel("epoch"); ax.legend()
        if title == "Learning rate":
            ax.set_yscale("log")
    fig.suptitle(f"{exp_id} — training curves (dashed line = best epoch {best} by val_auc)")
    plt.tight_layout()
    out = ROOT / "reports" / "figures"; out.mkdir(parents=True, exist_ok=True)
    ev = ROOT / "project_reports" / "evidence" / "training"; ev.mkdir(parents=True, exist_ok=True)
    png = out / f"{tag}_curves.png"
    plt.savefig(png, dpi=130)
    shutil.copy(png, ev / png.name)
    print("Saved:", png, "and", ev / png.name)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "EXP-001")
