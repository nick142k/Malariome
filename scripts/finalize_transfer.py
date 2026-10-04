"""Finalize a transfer-learning run whose stage 2 was interrupted (e.g. Kaggle connection lost).

Uses ONLY what the run already saved: per-epoch history CSVs and best-epoch checkpoints for both stages.
Picks the stage with the higher best val_auc, copies its checkpoint to models/transfer_model.keras, writes the meta
file, appends an experiment-log entry that clearly says the run was interrupted, and updates training_metrics.json.
The test set is not touched.

Usage (project root):  python scripts/finalize_transfer.py --exp-id EXP-002 [--finetune-epochs-planned 10]
"""
import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data.preprocessing import Preprocessor  # noqa: E402
from src.training.staged_trainer import combine_stage_summaries  # noqa: E402
from src.training.trainer import _merge_json, format_experiment_entry, summarize_history  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-id", default="EXP-002")
    ap.add_argument("--finetune-epochs-planned", type=int, default=10)
    ap.add_argument("--unfreeze-layers", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--finetune-lr", type=float)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()

    cfg = load_config()
    t, tc, paths = cfg["training"], cfg.get("transfer", {}), cfg["abs_paths"]
    lr = a.lr or t["learning_rate"]
    ft_lr = a.finetune_lr or t["fine_tune_learning_rate"]
    batch = a.batch_size or t["batch_size"]
    seed = a.seed if a.seed is not None else t["random_seed"]
    unfreeze = a.unfreeze_layers or tc.get("unfreeze_layers", 30)
    tag = a.exp_id.lower().replace("-", "")

    hist = {s: paths["reports_dir"] / "metrics" / f"{tag}_stage{s}_history.csv" for s in (1, 2)}
    ckpt = {s: paths["checkpoints_dir"] / f"{tag}_stage{s}_best.keras" for s in (1, 2)}
    missing = [str(p) for p in [*hist.values(), *ckpt.values()] if not p.exists()]
    if missing:
        sys.exit("Missing saved files from the interrupted run:\n  " + "\n  ".join(missing))

    h1, h2 = (pd.read_csv(hist[s]).to_dict(orient="list") for s in (1, 2))
    s1, s2 = summarize_history(h1), summarize_history(h2)
    summary, chosen = combine_stage_summaries(s1, s2)
    print(f"Stage 1: {s1['epochs_run']} epochs, best val_auc {s1['val_auc']:.4f} (epoch {s1['best_epoch']})")
    print(f"Stage 2: {s2['epochs_run']} of {a.finetune_epochs_planned} planned epochs, best val_auc {s2['val_auc']:.4f} (epoch {s2['best_epoch']})")
    print(f"Chosen: {chosen} (global best epoch {summary['best_epoch']}, val_auc {summary['val_auc']:.4f})")

    final = paths["models_dir"] / "transfer_model.keras"
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ckpt[1 if chosen == "stage1" else 2], final)

    n_total = n_train = 0
    tf_v = keras_v = "unknown"
    try:                                              # parameter counts and versions need TensorFlow
        import keras
        import numpy as np
        import tensorflow as tf
        tf_v, keras_v = tf.__version__, keras.__version__
        m = keras.saving.load_model(str(final))
        n_total = int(m.count_params())
        n_train = int(sum(int(np.prod(w.shape)) for w in m.trainable_weights))
    except Exception as exc:
        print("Note: could not load the model to count parameters:", type(exc).__name__)

    pre = Preprocessor.from_config(cfg, "mobilenetv2")
    split_file = paths["splits_dir"] / "split_summary.json"
    dataset_txt = "Kaggle cell_images, patient-level split"
    if split_file.exists():
        ss = json.loads(split_file.read_text(encoding="utf-8"))["splits"]
        dataset_txt += f" (train {ss['train']['images']:,} / val {ss['val']['images']:,} / test {ss['test']['images']:,} images; seed {seed})"
    planned = s1["epochs_run"] + a.finetune_epochs_planned
    info = {"title": (f"MobileNetV2 two-stage, INTERRUPTED: stage 2 stopped after {s2['epochs_run']} of "
                      f"{a.finetune_epochs_planned} planned epochs (session connection lost); final = {chosen}"),
            "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"), "model": "mobilenetv2",
            "params_total": n_total, "params_trainable": n_train, "dataset": dataset_txt, "image_size": pre.image_size,
            "resize_mode": pre.resize_mode, "scaling": pre.scaling, "batch_size": batch, "epochs_planned": planned,
            "optimizer": "adam", "learning_rate": f"{lr} (head) / {ft_lr} (fine-tune, top {unfreeze} layers)",
            "augmentation": cfg["augmentation"],
            "validation_strategy": "fixed patient-level validation split; final stage chosen by best val_auc",
            "summary": summary, "train_seconds": float("nan"), "device": "GPU (Kaggle)",
            "tf_version": tf_v, "keras_version": keras_v, "seed": seed}
    entry = format_experiment_entry(a.exp_id, info)
    entry = entry.replace("nan min", "not recorded (run interrupted)")
    entry = entry.replace("(0 parameters, 0 trainable)", "(parameter counts not recorded)")
    entry = entry.replace("(to be written after reviewing the history CSV)",
                          f"Run interrupted during stage 2 after {s2['epochs_run']} of {a.finetune_epochs_planned} planned epochs; "
                          "the best checkpoints saved before the interruption were used. Stage 2 did not run to completion, "
                          "so early stopping and the full planned fine-tuning schedule were not applied. (Add further observations.)")
    log = paths["project_reports_dir"] / "EXPERIMENT_LOG.md"
    log.parent.mkdir(parents=True, exist_ok=True)
    if not log.exists():
        log.write_text("# Experiment Log\n\nEntries are appended; earlier experiments are never overwritten.\n\n", encoding="utf-8")
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(entry)
    (paths["project_reports_dir"] / "metrics").mkdir(parents=True, exist_ok=True)
    _merge_json(paths["project_reports_dir"] / "metrics" / "training_metrics.json", a.exp_id,
                {**info, "train_seconds": None, "stage1_summary": s1, "stage2_summary": s2, "chosen_stage": chosen,
                 "unfreeze_layers": unfreeze, "interrupted": True, "stage2_epochs_planned": a.finetune_epochs_planned})
    meta = {"model_name": "mobilenetv2", "model_version": f"0.1-{a.exp_id}", "experiment": a.exp_id, "input_size": pre.image_size,
            "class_names": cfg["classes"]["names"], "positive_class": cfg["classes"]["positive_class"],
            "preprocessing": pre.to_metadata(), "optimizer": "adam", "learning_rate": {"head": lr, "fine_tune": ft_lr},
            "chosen_stage": chosen, "best_epoch_global": summary["best_epoch"], "random_seed": seed, "threshold": None,
            "threshold_note": "not selected yet (validation data only)", "interrupted_run": True,
            "validation_metrics_at_best_epoch": {k: v for k, v in summary.items() if k.startswith("val_")},
            "tensorflow": tf_v, "keras": keras_v}
    (paths["models_dir"] / "transfer_model.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {final}\nLogged {a.exp_id} in {log}")


if __name__ == "__main__":
    main()
