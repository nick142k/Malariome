"""Custom CNN trained from scratch (no pretrained weights).

Conv -> BN -> ReLU -> MaxPool -> Dropout  (x2), then Conv -> BN -> ReLU -> GlobalAveragePooling -> Dense -> Dropout -> Sigmoid.
Input: float images already scaled to [0, 1] by Preprocessor (scaling='unit'); no rescaling layer inside the model.
Output: single sigmoid = P(Parasitized). The last conv block's layers are named conv3 / relu3 (used by Grad-CAM later).
Status: IMPLEMENTED - NOT YET VERIFIED (needs TensorFlow/Keras 3; first run via `python train.py --dry-run`).
"""


def build_custom_cnn(image_size=224, filters=(32, 64, 128), dense_units=64, dropouts=(0.25, 0.25, 0.4)):
    import keras
    L = keras.layers
    if len(dropouts) != len(filters):
        raise ValueError("dropouts needs one value per conv block (the last one is used after the dense layer)")
    inp = keras.Input(shape=(image_size, image_size, 3), name="image")
    x = inp
    for i, f in enumerate(filters):
        x = L.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i + 1}")(x)
        x = L.BatchNormalization(name=f"bn{i + 1}")(x)
        x = L.ReLU(name=f"relu{i + 1}")(x)
        if i < len(filters) - 1:
            x = L.MaxPooling2D(2, name=f"pool{i + 1}")(x)
            x = L.Dropout(dropouts[i], name=f"drop{i + 1}")(x)
    x = L.GlobalAveragePooling2D(name="gap")(x)
    x = L.Dense(dense_units, activation="relu", name="dense")(x)
    x = L.Dropout(dropouts[-1], name="drop_head")(x)
    out = L.Dense(1, activation="sigmoid", name="prob_parasitized")(x)
    return keras.Model(inp, out, name="custom_cnn")
