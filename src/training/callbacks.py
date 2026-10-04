"""Standard callbacks. Everything watches validation AUC with mode='max' (higher is better)."""


def build_callbacks(checkpoint_path, history_csv, patience=3, reduce_lr_patience=2, reduce_lr_factor=0.5):
    import keras
    C = keras.callbacks
    return [
        C.ModelCheckpoint(str(checkpoint_path), monitor="val_auc", mode="max", save_best_only=True, verbose=1),
        C.EarlyStopping(monitor="val_auc", mode="max", patience=patience, restore_best_weights=True, verbose=1),
        C.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=reduce_lr_factor, patience=reduce_lr_patience,
                            min_lr=1e-6, verbose=1),
        C.CSVLogger(str(history_csv)),
    ]
