"""MobileNetV2 transfer model (ImageNet weights as initialisation; still trained on the malaria data).

Stage 1: base frozen, only the new head (GAP -> Dropout -> Sigmoid) is trained.
Stage 2: the top N base layers are unfrozen (BatchNorm layers stay frozen) and fine-tuned at a much lower learning rate.
Input must be scaled to [-1, 1] (Preprocessor scaling 'minus1_1'), which is what MobileNetV2 was trained with.
Status: IMPLEMENTED - NOT YET VERIFIED (needs TensorFlow + internet for the ImageNet weights).
"""


def build_mobilenetv2(image_size=224, dropout=0.3, weights="imagenet"):
    """Return (model, base). weights=None gives random init (offline smoke tests only)."""
    import keras
    L = keras.layers
    base = keras.applications.MobileNetV2(input_shape=(image_size, image_size, 3), include_top=False, weights=weights)
    base.trainable = False
    inp = keras.Input(shape=(image_size, image_size, 3), name="image")
    x = base(inp, training=False)                 # keeps BatchNorm in inference mode, also during fine-tuning
    x = L.GlobalAveragePooling2D(name="gap")(x)
    x = L.Dropout(dropout, name="drop_head")(x)
    out = L.Dense(1, activation="sigmoid", name="prob_parasitized")(x)
    return keras.Model(inp, out, name="mobilenetv2_malaria"), base


def unfreeze_top_layers(base, n_layers):
    """Make the last n_layers of the base trainable, except BatchNorm layers. Recompile the model afterwards."""
    import keras
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers[-n_layers:]:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
