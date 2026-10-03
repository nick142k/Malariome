"""Training-only augmentation (Keras 3 preprocessing layers). Operates on 0..255 pixels, i.e. BEFORE
model-specific scaling. Never attach this to validation or test data.

Black fill: rotations/shifts/zooms fill new area with black (constant 0) to match the dataset's black
background; Keras' default 'reflect' would mirror cell content and create fake structures.
Status: IMPLEMENTED - NOT YET VERIFIED (needs TensorFlow; first exercised in Phase 6).
"""


def build_augmentation(aug_cfg: dict, seed: int = 42):
    import keras
    L = keras.layers
    fill = {"fill_mode": aug_cfg.get("fill_mode", "constant"), "fill_value": 0.0}
    layers = []
    if aug_cfg.get("horizontal_flip") and aug_cfg.get("vertical_flip"):
        layers.append(L.RandomFlip("horizontal_and_vertical", seed=seed))
    elif aug_cfg.get("horizontal_flip"):
        layers.append(L.RandomFlip("horizontal", seed=seed))
    elif aug_cfg.get("vertical_flip"):
        layers.append(L.RandomFlip("vertical", seed=seed))
    if aug_cfg.get("rotation_degrees", 0) > 0:
        layers.append(L.RandomRotation(aug_cfg["rotation_degrees"] / 360.0, seed=seed, **fill))
    if aug_cfg.get("zoom", 0) > 0:
        layers.append(L.RandomZoom(aug_cfg["zoom"], seed=seed, **fill))
    if aug_cfg.get("translation", 0) > 0:
        layers.append(L.RandomTranslation(aug_cfg["translation"], aug_cfg["translation"], seed=seed, **fill))
    if aug_cfg.get("brightness", 0) > 0:
        layers.append(L.RandomBrightness(aug_cfg["brightness"], value_range=(0, 255), seed=seed))
    if aug_cfg.get("contrast", 0) > 0:
        layers.append(L.RandomContrast(aug_cfg["contrast"], seed=seed))
    if aug_cfg.get("hue", 0) > 0 and hasattr(L, "RandomHue"):
        layers.append(L.RandomHue(aug_cfg["hue"], value_range=(0, 255), seed=seed))
    if aug_cfg.get("saturation", 0) > 0 and hasattr(L, "RandomSaturation"):
        layers.append(L.RandomSaturation(aug_cfg["saturation"], value_range=(0, 255), seed=seed))
    return keras.Sequential(layers, name="train_augmentation")
