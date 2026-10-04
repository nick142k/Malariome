"""Evaluate models on the validation or test split.

  python evaluate.py --split val
      score both models on VALIDATION, pick thresholds
      (rule in src/evaluation/metrics.py)

  python evaluate.py --split val --benchmark
      also time the models (parameters, size, ms/image)

  python evaluate.py --split test --final
      THE one final test evaluation, with thresholds frozen from validation

Test-set rules enforced here:
--final is required;
thresholds must already exist from validation;
if test predictions already exist the run is refused unless --allow-repeat;
every test run is appended to project_reports/TEST_ACCESS_LOG.md.
"""

import argparse
import datetime
import json
import shutil
import sys

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation import metrics as M
from src.evaluation import reports as R


DISPLAY = {
    "custom_cnn": "Custom CNN",
    "transfer_model": "Transfer Model (MobileNetV2)",
}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}

    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]

    if isinstance(o, np.integer):
        return int(o)

    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)

    if isinstance(o, np.bool_):
        return bool(o)

    return o


def load_json(path):
    return (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {}
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "--split",
        choices=["val", "test"],
        required=True,
    )

    ap.add_argument(
        "--models",
        nargs="+",
        default=list(DISPLAY),
        choices=list(DISPLAY),
    )

    ap.add_argument(
        "--final",
        action="store_true",
        help="required for the test split",
    )

    ap.add_argument(
        "--allow-repeat",
        action="store_true",
        help="re-analyse an already evaluated test split (logged)",
    )

    ap.add_argument(
        "--refresh",
        action="store_true",
        help="recompute predictions instead of using the cached CSV",
    )

    ap.add_argument(
        "--benchmark",
        action="store_true",
    )

    ap.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
    )

    a = ap.parse_args()

    cfg = load_config()
    P = cfg["abs_paths"]

    pred_dir = P["reports_dir"] / "predictions"
    met_dir = P["reports_dir"] / "metrics"
    fig_dir = P["reports_dir"] / "figures"

    ev_dir = (
        P["project_reports_dir"]
        / "evidence"
        / "evaluation"
    )

    pm_path = (
        P["project_reports_dir"]
        / "metrics"
        / "performance_metrics.json"
    )

    sel_path = met_dir / "selected_thresholds.json"

    for d in (
        pred_dir,
        met_dir,
        fig_dir,
        ev_dir,
        pm_path.parent,
    ):
        d.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Test split safety checks
    # ---------------------------------------------------------

    if a.split == "test":
        if not a.final:
            sys.exit(
                "The test split is locked. Choose models and thresholds "
                "on validation, then run once with --final."
            )

        if not sel_path.exists():
            sys.exit(
                "No selected thresholds yet. "
                "Run `python evaluate.py --split val` first."
            )

        done = [
            m
            for m in a.models
            if (pred_dir / f"{m}_test.csv").exists()
        ]

        if done and not a.allow_repeat:
            sys.exit(
                f"Test predictions already exist for {done}: "
                "the test set has been evaluated. "
                "Re-running is not a clean single evaluation; "
                "use --allow-repeat only if you must (it is logged)."
            )

    # ---------------------------------------------------------
    # Benchmark models
    # ---------------------------------------------------------

    if a.benchmark:
        from src.evaluation.batch_inference import benchmark_model

        perf = load_json(pm_path)

        for m in a.models:
            b = benchmark_model(m, cfg)

            perf.setdefault(m, {})[b["device"]] = b

            print(f"benchmark {m}: {b}")

        pm_path.write_text(
            json.dumps(perf, indent=2),
            encoding="utf-8",
        )

    # ---------------------------------------------------------
    # Generate/load predictions
    # ---------------------------------------------------------

    preds = {}

    for m in a.models:
        f = pred_dir / f"{m}_{a.split}.csv"

        if f.exists() and not a.refresh:
            preds[m] = pd.read_csv(f)

        else:
            from src.evaluation.batch_inference import predict_split

            print(
                f"Predicting {a.split} split with {m} ..."
            )

            preds[m] = predict_split(
                m,
                a.split,
                cfg,
            )

            preds[m].to_csv(
                f,
                index=False,
            )

    # ---------------------------------------------------------
    # Load/select thresholds and calculate metrics
    # ---------------------------------------------------------

    selected = load_json(sel_path)
    results = {}

    for m, df in preds.items():
        y = df["label"].to_numpy(int)
        p = df["prob"].to_numpy(float)
        pid = df["patient_id"].to_numpy()

        if a.split == "val":
            selected[m] = {
                **M.select_threshold(y, p),
                "selected_on": "validation",
                "date": datetime.date.today().isoformat(),
            }

        elif m not in selected:
            sys.exit(
                f"No validation-selected threshold for {m}."
            )

        thr = selected[m]["threshold"]

        res = {
            "images": len(df),
            "patients": int(len(np.unique(pid))),
            "parasitized": int(y.sum()),
            **M.ranking_metrics(y, p),
            "threshold_selected": thr,
            "at_selected_threshold": M.metrics_at(
                y,
                p,
                thr,
            ),
            "at_0.5": M.metrics_at(
                y,
                p,
                0.5,
            ),
            "ci95_patient_bootstrap": M.bootstrap_by_patient(
                y,
                p,
                pid,
                thr,
                a.bootstrap,
            ),
        }

        if a.split == "val":
            res["note"] = (
                "threshold was selected on this same data, "
                "so results at the selected threshold are optimistic"
            )

        results[m] = res

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------

    metrics_path = (
        met_dir / f"{a.split}_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            jsonable(
                {
                    "split": a.split,
                    "models": results,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Plot ROC / PR curves
    # ---------------------------------------------------------

    names = {
        m: DISPLAY[m]
        for m in results
    }

    curves = {
        names[m]: (
            preds[m]["label"].to_numpy(int),
            preds[m]["prob"].to_numpy(float),
        )
        for m in results
    }

    outputs = [metrics_path]

    roc_path = (
        fig_dir / f"{a.split}_roc.png"
    )

    pr_path = (
        fig_dir / f"{a.split}_pr.png"
    )

    R.plot_roc(
        curves,
        roc_path,
        f"ROC curve ({a.split} split)",
    )

    R.plot_pr(
        curves,
        pr_path,
        f"Precision-recall curve ({a.split} split)",
    )

    outputs += [
        roc_path,
        pr_path,
    ]

    # ---------------------------------------------------------
    # Confusion matrices
    # ---------------------------------------------------------

    for m, res in results.items():
        f = (
            fig_dir
            / f"{a.split}_confusion_{m}.png"
        )

        R.plot_confusion(
            res["at_selected_threshold"],
            f,
            (
                f"{names[m]} - {a.split}\n"
                f"threshold {res['threshold_selected']:.2f}"
            ),
        )

        outputs.append(f)

    # ---------------------------------------------------------
    # Validation-only threshold analysis
    # ---------------------------------------------------------

    if a.split == "val":
        sel_path.write_text(
            json.dumps(
                jsonable(selected),
                indent=2,
            ),
            encoding="utf-8",
        )

        tables = {}

        for m, df in preds.items():
            t = M.threshold_table(
                df["label"].to_numpy(int),
                df["prob"].to_numpy(float),
                M.REPORT_THRESHOLDS,
            )

            t.insert(
                0,
                "model",
                m,
            )

            tables[names[m]] = t

        threshold_csv = (
            met_dir / "threshold_analysis.csv"
        )

        old = (
            pd.read_csv(threshold_csv)
            if threshold_csv.exists()
            else None
        )

        new = pd.concat(
            tables.values(),
            ignore_index=True,
        )

        if old is not None:
            new = pd.concat(
                [
                    old[
                        ~old["model"].isin(
                            new["model"]
                        )
                    ],
                    new,
                ],
                ignore_index=True,
            )

        new.to_csv(
            threshold_csv,
            index=False,
        )

        threshold_fig = (
            fig_dir / "val_threshold_curves.png"
        )

        R.plot_threshold_curves(
            {
                names[m]: M.threshold_table(
                    preds[m]["label"].to_numpy(int),
                    preds[m]["prob"].to_numpy(float),
                    M.FINE_GRID,
                )
                for m in results
            },
            {
                names[m]: selected[m]["threshold"]
                for m in results
            },
            threshold_fig,
        )

        outputs += [
            sel_path,
            threshold_csv,
            threshold_fig,
        ]

    # ---------------------------------------------------------
    # Load performance information
    # ---------------------------------------------------------

    perf = load_json(pm_path)

    def cell(m, key, ci=True, digits=3):
        r = results[m]

        if key in ("roc_auc", "pr_auc"):
            v = r[key]
        else:
            v = r["at_selected_threshold"][key]

        return R.fmt(
            v,
            r["ci95_patient_bootstrap"].get(key)
            if ci
            else None,
            digits,
        )

    def perf_cell(m, key, unit=""):
        entries = perf.get(m, {})

        if not entries:
            return "NOT YET MEASURED"

        dev, e = next(iter(entries.items()))

        return (
            f"{e[key]:,}{unit}"
            + (
                f" ({dev})"
                if "ms" in key
                else ""
            )
        )

    # ---------------------------------------------------------
    # Comparison table
    # ---------------------------------------------------------

    rows = [
        (
            "Accuracy",
            {
                m: cell(m, "accuracy")
                for m in results
            },
        ),
        (
            "Precision",
            {
                m: cell(m, "precision")
                for m in results
            },
        ),
        (
            "Recall / Sensitivity",
            {
                m: cell(m, "sensitivity")
                for m in results
            },
        ),
        (
            "Specificity",
            {
                m: cell(m, "specificity")
                for m in results
            },
        ),
        (
            "F1",
            {
                m: cell(m, "f1")
                for m in results
            },
        ),
        (
            "ROC-AUC",
            {
                m: cell(m, "roc_auc")
                for m in results
            },
        ),
        (
            "PR-AUC (average precision)",
            {
                m: cell(m, "pr_auc")
                for m in results
            },
        ),
        (
            "Threshold used",
            {
                m: f"{results[m]['threshold_selected']:.2f}"
                for m in results
            },
        ),
        (
            "Parameters",
            {
                m: perf_cell(m, "parameters")
                for m in results
            },
        ),
        (
            "Model size (MB)",
            {
                m: perf_cell(m, "model_size_mb")
                for m in results
            },
        ),
        (
            "Inference, single image (ms)",
            {
                m: perf_cell(
                    m,
                    "single_image_ms_mean",
                )
                for m in results
            },
        ),
    ]

    # Convert model keys to display names
    rows = [
        (
            label,
            {
                names[m]: text
                for m, text in cells.items()
            },
        )
        for label, cells in rows
    ]

    title = (
        f"Model comparison - {a.split} split "
        "(95% patient-bootstrap CI in brackets)"
    )

    md = R.comparison_markdown(
        rows,
        [names[m] for m in results],
        title,
    )

    if a.split == "val":
        md += (
            "\nThresholds were selected on this same "
            "validation data, so threshold-dependent "
            "numbers here are optimistic.\n"
        )

    comparison_path = (
        met_dir / f"{a.split}_comparison.md"
    )

    comparison_path.write_text(
        md,
        encoding="utf-8",
    )

    outputs.append(comparison_path)

    # ---------------------------------------------------------
    # Copy evidence files
    # ---------------------------------------------------------

    for f in outputs:
        shutil.copy(
            f,
            ev_dir / f.name,
        )

    # ---------------------------------------------------------
    # Test-set access log
    # ---------------------------------------------------------

    if a.split == "test":
        log = (
            P["project_reports_dir"]
            / "TEST_ACCESS_LOG.md"
        )

        if not log.exists():
            log.write_text(
                "# Test-set access log\n\n"
                "Every evaluation on the held-out test "
                "split is recorded here.\n\n",
                encoding="utf-8",
            )

        threshold_text = ", ".join(
            f"{m}={selected[m]['threshold']:.2f}"
            for m in results
        )

        with open(
            log,
            "a",
            encoding="utf-8",
        ) as fh:
            fh.write(
                f"- "
                f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
                f"| models: {', '.join(results)} "
                f"| thresholds: {threshold_text} "
                f"| repeat: {bool(a.allow_repeat)} "
                f"| images: "
                f"{next(iter(results.values()))['images']}\n"
            )

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    print("\n" + md)

    for m, r in results.items():
        if (
            a.split == "val"
            and selected[m].get("fallback_max_f1")
        ):
            print(
                f"WARNING: {m} never reached "
                f"sensitivity {M.TARGET_SENSITIVITY}; "
                "the max-F1 fallback threshold was used."
            )

    print(
        "Saved to reports/metrics, reports/figures "
        "and project_reports/evidence/evaluation."
    )


if __name__ == "__main__":
    main()
