"""Training orchestration. TensorFlow/Keras are imported lazily so the pure helpers
(summarize_history, format_experiment_entry) can be tested without TensorFlow.

Only the TRAIN and VALIDATION splits are used here. The test split is never touched.
Status: IMPLEMENTED - NOT YET VERIFIED for the TF parts (first check: `python train.py --smoke`).
"""
import datetime
import json
import time
from pathlib import Path

import numpy as np

from src.config import load_config
from src.data.augmentation import build_augmentation
from src.data.loader import make_tf_dataset
from src.data.preprocessing import Preprocessor
from src.models.model_loader import build_model
from src.training.callbacks import build_callbacks

METRIC_KEYS = ["loss", "accuracy", "precision", "recall", "auc", "pr_auc"]


def summarize_history(history: dict) -> dict:
    """Best epoch = highest val_auc. Returns the train/val metrics recorded at that epoch (pure function)."""
    val_auc = history["val_auc"]
    best = int(np.argmax(val_auc))
    out = {"epochs_run": len(val_auc), "best_epoch": best + 1}
    for k in METRIC_KEYS:
        if f"val_{k}" in history:
            out[f"val_{k}"] = float(history[f"val_{k}"][best])
        if k in history:
            out[f"train_{k}"] = float(history[k][best])
    return out


def format_experiment_entry(exp_id, info: dict) -> str:
    """Markdown entry for EXPERIMENT_LOG.md. Only values measured in the run are filled in."""
    s = info["summary"]
    lines = [
        f"## {exp_id} — {info['title']}", "",
        f"- **Date (UTC):** {info['date']}",
        f"- **Model:** {info['model']} ({info['params_total']:,} parameters, {info['params_trainable']:,} trainable)",
        f"- **Dataset:** {info['dataset']}",
        f"- **Image size:** {info['image_size']}x{info['image_size']} ({info['resize_mode']}, scaling={info['scaling']})",
        f"- **Batch size:** {info['batch_size']}",
        f"- **Epochs:** {s['epochs_run']} run of {info['epochs_planned']} planned (best epoch {s['best_epoch']})",
        f"- **Optimizer / LR:** {info['optimizer']} / {info['learning_rate']} (ReduceLROnPlateau on val_auc)",
        f"- **Augmentation (train only):** {json.dumps(info['augmentation'])}",
        f"- **Validation strategy:** {info['validation_strategy']}",
        "- **Validation metrics at best epoch (threshold 0.5):** " + ", ".join(
            f"{k.replace('val_', '')}={s[k]:.4f}" for k in [f"val_{m}" for m in METRIC_KEYS] if k in s),
        f"- **Training time:** {info['train_seconds'] / 60:.1f} min on {info['device']}",
        f"- **Environment:** TensorFlow {info['tf_version']}, Keras {info['keras_version']}, seed {info['seed']}",
        "- **Test set:** NOT YET EVALUATED",
        "- **Observations:** (to be written after reviewing the history CSV)",
        "- **Conclusion:** (to be written)", "",
    ]
    return "\n".join(lines) + "\n"


def compile_model(model, lr):
    import keras
    M = keras.metrics
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="binary_crossentropy",
                  metrics=[M.BinaryAccuracy(name="accuracy"), M.Precision(name="precision"), M.Recall(name="recall"),
                           M.AUC(name="auc"), M.AUC(name="pr_auc", curve="PR")])


def _merge_json(path: Path, key: str, value: dict):
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[key] = value
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def train(model_name="custom_cnn", exp_id="EXP-001", cfg=None, epochs=None, batch_size=None, lr=None, seed=None,
          patience=3, smoke=False, title=None):
    import keras
    import tensorflow as tf

    cfg = cfg or load_config()
    t = cfg["training"]
    epochs = epochs or t["epochs"]
    batch_size = batch_size or t["batch_size"]
    lr = lr or t["learning_rate"]
    seed = seed if seed is not None else t["random_seed"]
    keras.utils.set_random_seed(seed)

    gpus = tf.config.list_physical_devices("GPU")
    device = f"GPU ({len(gpus)})" if gpus else "CPU"
    print(f"Device: {device} | TensorFlow {tf.__version__} | Keras {keras.__version__}")

    paths = cfg["abs_paths"]
    run_dir = paths["reports_dir"] / ("smoke" if smoke else "metrics")
    ckpt_dir = paths["reports_dir"] / "smoke" if smoke else paths["checkpoints_dir"]
    for d in (run_dir, ckpt_dir, paths["models_dir"], paths["project_reports_dir"] / "metrics"):
        Path(d).mkdir(parents=True, exist_ok=True)

    pre = Preprocessor.from_config(cfg, model_name)
    aug = build_augmentation(cfg["augmentation"], seed)
    to_col = lambda x, y: (x, tf.expand_dims(y, -1))      # labels (N,) -> (N, 1) to match the sigmoid output
    train_ds = make_tf_dataset("train", pre, batch_size, training=True, cfg=cfg, augmentation=aug, seed=seed).map(to_col)
    val_ds = make_tf_dataset("val", pre, batch_size, training=False, cfg=cfg).map(to_col)
    if smoke:
        train_ds, val_ds, epochs = train_ds.take(2), val_ds.take(1), 1

    model = build_model(model_name, cfg)
    compile_model(model, lr)
    n_total = int(model.count_params())
    n_train = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    print(f"Parameters: {n_total:,} total, {n_train:,} trainable")

    tag = "smoke" if smoke else exp_id.lower().replace("-", "")
    callbacks = build_callbacks(ckpt_dir / f"{tag}_best.keras", run_dir / f"{tag}_history.csv", patience=patience,
                                reduce_lr_patience=t["reduce_lr_patience"], reduce_lr_factor=t["reduce_lr_factor"])
    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2)
    seconds = time.time() - t0
    summary = summarize_history(hist.history)
    print("Best epoch summary:", json.dumps(summary, indent=2))
    if smoke:
        print("\nSMOKE TEST PASSED: data loading, augmentation, model, metrics and callbacks all ran. Nothing was saved to models/ or the logs.")
        return summary

    final_path = paths["models_dir"] / f"{model_name}.keras"
    model.save(str(final_path))          # weights restored to the best val_auc epoch by EarlyStopping
    split_file = paths["splits_dir"] / "split_summary.json"
    dataset_txt = "Kaggle cell_images, patient-level split"
    if split_file.exists():
        ss = json.loads(split_file.read_text(encoding="utf-8"))["splits"]
        dataset_txt += f" (train {ss['train']['images']:,} / val {ss['val']['images']:,} / test {ss['test']['images']:,} images; seed {seed})"
    info = {"title": title or f"{model_name} (CPU/GPU run)", "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "model": model_name, "params_total": n_total, "params_trainable": n_train, "dataset": dataset_txt,
            "image_size": pre.image_size, "resize_mode": pre.resize_mode, "scaling": pre.scaling, "batch_size": batch_size,
            "epochs_planned": epochs, "optimizer": "adam", "learning_rate": lr, "augmentation": cfg["augmentation"],
            "validation_strategy": "fixed patient-level validation split; best epoch chosen by val_auc",
            "summary": summary, "train_seconds": seconds, "device": device, "tf_version": tf.__version__,
            "keras_version": keras.__version__, "seed": seed}
    log = paths["project_reports_dir"] / "EXPERIMENT_LOG.md"
    if not log.exists():
        log.write_text("# Experiment Log\n\nEntries are appended; earlier experiments are never overwritten.\n\n", encoding="utf-8")
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(format_experiment_entry(exp_id, info))
    _merge_json(paths["project_reports_dir"] / "metrics" / "training_metrics.json", exp_id,
                {k: v for k, v in info.items() if k not in ("augmentation",)} | {"augmentation": cfg["augmentation"]})
    meta = {"model_name": model_name, "model_version": f"0.1-{exp_id}", "experiment": exp_id, "input_size": pre.image_size,
            "class_names": cfg["classes"]["names"], "positive_class": cfg["classes"]["positive_class"],
            "preprocessing": pre.to_metadata(), "optimizer": "adam", "learning_rate": lr, "epochs_run": summary["epochs_run"],
            "best_epoch": summary["best_epoch"], "random_seed": seed, "threshold": None,
            "threshold_note": "not selected yet (Phase 12, validation data only)",
            "validation_metrics_at_best_epoch": {k: v for k, v in summary.items() if k.startswith("val_")},
            "tensorflow": tf.__version__, "keras": keras.__version__}
    (paths["models_dir"] / f"{model_name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved model: {final_path}\nLogged {exp_id} in {log}")
    return summary
