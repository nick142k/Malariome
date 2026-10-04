"""Model inference over a whole split + latency benchmark. Needs TensorFlow (imported lazily).
Preprocessing is rebuilt from the model's own meta file, so it always matches what the model was trained with.
Status: IMPLEMENTED - NOT YET VERIFIED (first run: `python evaluate.py --split val`)."""
import json

import numpy as np

from src.data.preprocessing import Preprocessor


def load_model_and_pre(name, cfg):
    import keras
    d = cfg["abs_paths"]["models_dir"]
    meta = json.loads((d / f"{name}.meta.json").read_text(encoding="utf-8"))
    return keras.saving.load_model(str(d / f"{name}.keras")), Preprocessor.from_metadata(meta["preprocessing"]), meta


def predict_split(name, split, cfg, batch_size=64):
    from src.data.loader import load_split, make_tf_dataset
    model, pre, _ = load_model_and_pre(name, cfg)
    _, _, df = load_split(split, cfg)
    ds = make_tf_dataset(split, pre, batch_size, training=False, cfg=cfg)      # same order as the split CSV
    prob = model.predict(ds, verbose=0).ravel()
    if len(prob) != len(df):
        raise RuntimeError(f"{len(prob)} predictions for {len(df)} images")
    out = df[["path", "label", "patient_id"]].copy()
    out["prob"] = prob
    return out


def benchmark_model(name, cfg, n_images=64):
    """Latency on validation images (preprocessing excluded). Single-image and batch-of-32 timings, model size, parameters."""
    import time

    import tensorflow as tf
    from src.data.loader import load_split
    model, pre, _ = load_model_and_pre(name, cfg)
    paths, _, _ = load_split("val", cfg)
    x = np.stack([pre(p) for p in paths[:n_images]]).astype("float32")
    for _ in range(3):
        model(x[:1], training=False)
    single = []
    for i in range(30):
        s = time.perf_counter()
        np.asarray(model(x[i % len(x): i % len(x) + 1], training=False))
        single.append(time.perf_counter() - s)
    model.predict(x[:32], verbose=0)
    batch = []
    for _ in range(5):
        s = time.perf_counter()
        model.predict(x[:32], verbose=0)
        batch.append(time.perf_counter() - s)
    size_mb = (cfg["abs_paths"]["models_dir"] / f"{name}.keras").stat().st_size / 1e6
    return {"device": "GPU" if tf.config.list_physical_devices("GPU") else "CPU", "parameters": int(model.count_params()),
            "model_size_mb": round(size_mb, 2), "single_image_ms_mean": round(float(np.mean(single)) * 1000, 2),
            "single_image_ms_median": round(float(np.median(single)) * 1000, 2),
            "batch32_ms_per_image": round(float(np.mean(batch)) / 32 * 1000, 2), "timing_runs_single": 30,
            "preprocessing_included": False, "tensorflow": tf.__version__}
