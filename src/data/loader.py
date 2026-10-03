"""Split manifests -> arrays / tf.data pipelines.

load_split() is pure pandas (tested). make_tf_dataset() needs TensorFlow and is
IMPLEMENTED - NOT YET VERIFIED (first run: Phase 6 on Colab/Kaggle).
Order inside the pipeline: decode+pad+resize (0..255) -> [train only] augmentation -> model-specific scaling.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config

SPLITS = ("train", "val", "test")


def load_split(split, cfg=None):
    """Return (absolute image paths, float32 labels, dataframe) for 'train' | 'val' | 'test'."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    cfg = cfg or load_config()
    df = pd.read_csv(Path(cfg["abs_paths"]["splits_dir"]) / f"{split}.csv")
    data_dir = Path(cfg["abs_paths"]["data_dir"])
    return [str(data_dir / p) for p in df["path"]], df["label"].to_numpy(np.float32), df


def make_tf_dataset(split, preprocessor, batch_size, training=False, cfg=None, augmentation=None, seed=42):
    import tensorflow as tf
    paths, labels, _ = load_split(split, cfg)
    size = preprocessor.image_size
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def _load(path, label):
        img = tf.py_function(lambda p: preprocessor.resize_image(p.numpy().decode("utf-8")), [path], tf.float32)
        img.set_shape((size, size, 3))
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    if training and augmentation is not None:
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    def _scale(x, y):                                   # mirrors Preprocessor.scale
        if preprocessor.scaling == "unit":
            return x / 255.0, y
        if preprocessor.scaling == "minus1_1":
            return x / 127.5 - 1.0, y
        return x, y

    return ds.map(_scale, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
