"""Figures and comparison tables for evaluate.py (matplotlib 'Agg' backend; no display needed)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

NOTE = "Predictions are from a patient-level split; thresholds were chosen on validation data only."


def plot_roc(curves, path, title):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set(xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (sensitivity)", title=title)
    ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_pr(curves, path, title):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, (y, p) in curves.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=name)
    ax.set(xlabel="Recall (sensitivity)", ylabel="Precision", title=title, ylim=(0, 1.02))
    ax.legend(loc="lower left"); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_confusion(m, path, title):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=13)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Uninfected", "Parasitized"], yticklabels=["Uninfected", "Parasitized"],
           xlabel="Predicted", ylabel="Actual", title=title)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_threshold_curves(tables, marks, path):
    fig, axes = plt.subplots(1, len(tables), figsize=(6 * len(tables), 4.5), squeeze=False)
    for ax, (name, t) in zip(axes[0], tables.items()):
        for col in ("sensitivity", "specificity", "precision", "f1"):
            ax.plot(t["threshold"], t[col], label=col)
        ax.axvline(marks[name], color="gray", linestyle="--", linewidth=1)
        ax.axhline(0.95, color="lightgray", linestyle=":", linewidth=1)
        ax.set(title=f"{name}\n(dashed = selected threshold {marks[name]:.2f})", xlabel="threshold", ylim=(0, 1.02))
        ax.legend(loc="lower center", ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fmt(v, ci=None, digits=3):
    s = "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}f}"
    if ci and not np.isnan(ci["lo"]):
        s += f" ({ci['lo']:.{digits}f}-{ci['hi']:.{digits}f})"
    return s


def comparison_markdown(rows, model_names, title):
    """rows: list of (metric label, {model: cell text}). Returns a GitHub-flavoured markdown table."""
    out = [f"### {title}", "", "| Metric | " + " | ".join(model_names) + " |", "|---|" + "---:|" * len(model_names)]
    for label, cells in rows:
        out.append(f"| {label} | " + " | ".join(cells.get(m, "n/a") for m in model_names) + " |")
    return "\n".join(out) + "\n\n" + NOTE + "\n"
