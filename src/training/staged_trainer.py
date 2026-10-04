"""Two-stage transfer-learning training (head only, then fine-tuning). Uses only the train and validation splits.

The final model is whichever stage reached the higher best val_auc. Results go to models/transfer_model.keras,
EXPERIMENT_LOG.md and training_metrics.json. Status: IMPLEMENTED - NOT YET VERIFIED (TF parts; first check: --smoke).
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
from src.training.callbacks import build_callbacks
from src.training.trainer import _merge_json, compile_model, format_experiment_entry, summarize_history


def combine_stage_summaries(s1, s2):
    """Pick the stage with the higher best val_auc (ties go to stage 2). Epoch numbers are global across stages."""
    if s2 is not None and s2["val_auc"] >= s1["val_auc"]:
        chosen, src, offset = "stage2", s2, s1["epochs_run"]
    else:
        chosen, src, offset = "stage1", s1, 0
    out = dict(src)
    out["best_epoch"] = src["best_epoch"] + offset
    out["epochs_run"] = s1["epochs_run"] + (s2["epochs_run"] if s2 else 0)
    return out, chosen


def _counts(model):
    return int(model.count_params()), int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))


def train_transfer(exp_id="EXP-002", cfg=None, stage1_epochs=None, finetune_epochs=None, unfreeze_layers=None, lr=None,
                   finetune_lr=None, batch_size=None, seed=None, patience=3, smoke=False, pretrained=True, title=None):
    import keras
    import tensorflow as tf
    from src.models.transfer_learning import build_mobilenetv2, unfreeze_top_layers

    cfg = cfg or load_config()
    t, tc = cfg["training"], cfg.get("transfer", {})
    stage1_epochs = stage1_epochs or tc.get("stage1_epochs", 6)
    finetune_epochs = finetune_epochs or tc.get("finetune_epochs", 10)
    unfreeze_layers = unfreeze_layers or tc.get("unfreeze_layers", 30)
    lr = lr or t["learning_rate"]
    finetune_lr = finetune_lr or t["fine_tune_learning_rate"]
    batch_size = batch_size or t["batch_size"]
    seed = seed if seed is not None else t["random_seed"]
    if not pretrained and not smoke:
        raise ValueError("Real runs must use ImageNet weights; --no-pretrained is only for offline smoke tests.")
    keras.utils.set_random_seed(seed)

    pre = Preprocessor.from_config(cfg, "mobilenetv2")
    if pre.scaling != "minus1_1":
        raise ValueError("config.yaml needs data.scaling_by_model.mobilenetv2: minus1_1 (MobileNetV2 expects [-1, 1] input).")

    gpus = tf.config.list_physical_devices("GPU")
    device = f"GPU ({len(gpus)})" if gpus else "CPU"
    print(f"Device: {device} | TensorFlow {tf.__version__} | Keras {keras.__version__} | pretrained={pretrained}")

    paths = cfg["abs_paths"]
    run_dir = paths["reports_dir"] / ("smoke" if smoke else "metrics")
    ckpt_dir = paths["reports_dir"] / "smoke" if smoke else paths["checkpoints_dir"]
    for d in (run_dir, ckpt_dir, paths["models_dir"], paths["project_reports_dir"] / "metrics"):
        Path(d).mkdir(parents=True, exist_ok=True)

    aug = build_augmentation(cfg["augmentation"], seed)
    to_col = lambda x, y: (x, tf.expand_dims(y, -1))
    train_ds = make_tf_dataset("train", pre, batch_size, training=True, cfg=cfg, augmentation=aug, seed=seed).map(to_col)
    val_ds = make_tf_dataset("val", pre, batch_size, training=False, cfg=cfg).map(to_col)
    if smoke:
        train_ds, val_ds, stage1_epochs, finetune_epochs = train_ds.take(2), val_ds.take(1), 1, 1

    dropout = cfg.get("model", {}).get("mobilenetv2", {}).get("dropout", 0.3)
    model, base = build_mobilenetv2(pre.image_size, dropout, "imagenet" if pretrained else None)
    tag = "smoke" if smoke else exp_id.lower().replace("-", "")
    rl = dict(reduce_lr_patience=t["reduce_lr_patience"], reduce_lr_factor=t["reduce_lr_factor"])
    t0 = time.time()

    # ---- stage 1: head only ----
    compile_model(model, lr)
    n_total, n_train1 = _counts(model)
    print(f"Stage 1 (frozen base): {n_total:,} parameters, {n_train1:,} trainable, lr={lr}")
    ck1 = ckpt_dir / f"{tag}_stage1_best.keras"
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=stage1_epochs, verbose=2,
                   callbacks=build_callbacks(ck1, run_dir / f"{tag}_stage1_history.csv", patience=patience, **rl))
    s1 = summarize_history(h1.history)
    print("Stage 1 best:", json.dumps(s1))

    # ---- stage 2: fine-tune top layers at a much lower learning rate ----
    unfreeze_top_layers(base, unfreeze_layers)
    compile_model(model, finetune_lr)
    _, n_train2 = _counts(model)
    print(f"Stage 2 (top {unfreeze_layers} base layers unfrozen, BatchNorm frozen): {n_train2:,} trainable, lr={finetune_lr}")
    ck2 = ckpt_dir / f"{tag}_stage2_best.keras"
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=finetune_epochs, verbose=2,
                   callbacks=build_callbacks(ck2, run_dir / f"{tag}_stage2_history.csv", patience=patience, **rl))
    s2 = summarize_history(h2.history)
    print("Stage 2 best:", json.dumps(s2))

    summary, chosen = combine_stage_summaries(s1, s2)
    seconds = time.time() - t0
    if chosen == "stage1":
        model = keras.saving.load_model(str(ck1))
    n_train = n_train1 if chosen == "stage1" else n_train2
    print(f"Chosen: {chosen} | best val_auc {summary['val_auc']:.4f} (global epoch {summary['best_epoch']})")
    if smoke:
        print("\nSMOKE TEST PASSED: both stages ran. Nothing was saved to models/ or the logs.")
        return summary

    final_path = paths["models_dir"] / "transfer_model.keras"
    model.save(str(final_path))
    split_file = paths["splits_dir"] / "split_summary.json"
    dataset_txt = "Kaggle cell_images, patient-level split"
    if split_file.exists():
        ss = json.loads(split_file.read_text(encoding="utf-8"))["splits"]
        dataset_txt += f" (train {ss['train']['images']:,} / val {ss['val']['images']:,} / test {ss['test']['images']:,} images; seed {seed})"
    info = {"title": title or f"MobileNetV2 two-stage ({stage1_epochs} head epochs @ {lr}, then top {unfreeze_layers} layers "
                              f"up to {finetune_epochs} epochs @ {finetune_lr}); final = {chosen}",
            "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"), "model": "mobilenetv2",
            "params_total": n_total, "params_trainable": n_train, "dataset": dataset_txt, "image_size": pre.image_size,
            "resize_mode": pre.resize_mode, "scaling": pre.scaling, "batch_size": batch_size,
            "epochs_planned": stage1_epochs + finetune_epochs, "optimizer": "adam",
            "learning_rate": f"{lr} (head) / {finetune_lr} (fine-tune)", "augmentation": cfg["augmentation"],
            "validation_strategy": "fixed patient-level validation split; final stage chosen by best val_auc",
            "summary": summary, "train_seconds": seconds, "device": device, "tf_version": tf.__version__,
            "keras_version": keras.__version__, "seed": seed}
    log = paths["project_reports_dir"] / "EXPERIMENT_LOG.md"
    if not log.exists():
        log.write_text("# Experiment Log\n\nEntries are appended; earlier experiments are never overwritten.\n\n", encoding="utf-8")
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(format_experiment_entry(exp_id, info))
    _merge_json(paths["project_reports_dir"] / "metrics" / "training_metrics.json", exp_id,
                {**info, "stage1_summary": s1, "stage2_summary": s2, "chosen_stage": chosen, "unfreeze_layers": unfreeze_layers})
    meta = {"model_name": "mobilenetv2", "model_version": f"0.1-{exp_id}", "experiment": exp_id, "input_size": pre.image_size,
            "class_names": cfg["classes"]["names"], "positive_class": cfg["classes"]["positive_class"],
            "preprocessing": pre.to_metadata(), "optimizer": "adam", "learning_rate": {"head": lr, "fine_tune": finetune_lr},
            "chosen_stage": chosen, "best_epoch_global": summary["best_epoch"], "random_seed": seed, "threshold": None,
            "threshold_note": "not selected yet (validation data only)",
            "validation_metrics_at_best_epoch": {k: v for k, v in summary.items() if k.startswith("val_")},
            "tensorflow": tf.__version__, "keras": keras.__version__}
    (paths["models_dir"] / "transfer_model.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved model: {final_path}\nLogged {exp_id} in {log}")
    return summary
