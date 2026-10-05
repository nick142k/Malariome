"""
Train a malaria parasite species classifier.

Classes:
    Falciparum
    Malariae
    Ovale
    Vivax

Dataset:
    data/species/classification/

Expected structure:
    classification/
        Falciparum/
        Malariae/
        Ovale/
        Vivax/

The dataset is strongly imbalanced, so class weights are used.
"""

from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

DATA_DIR = Path("data/species/classification")
MODEL_DIR = Path("models")
REPORT_DIR = Path("project_reports/evidence/species")

MODEL_PATH = MODEL_DIR / "species_model.keras"
META_PATH = MODEL_DIR / "species_model.meta.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
VALIDATION_SPLIT = 0.20

LEARNING_RATE = 1e-4

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ============================================================
# CHECK DATASET
# ============================================================

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset directory not found:\n{DATA_DIR.resolve()}"
    )

class_names = sorted(
    [
        directory.name
        for directory in DATA_DIR.iterdir()
        if directory.is_dir()
    ]
)

expected_classes = ["Falciparum", "Malariae", "Ovale", "Vivax"]

if class_names != expected_classes:
    raise ValueError(
        f"Expected classes {expected_classes}, "
        f"but found {class_names}"
    )

print("\n========================================")
print("MALARIA SPECIES CLASSIFIER")
print("========================================")

print("\nDataset:")
print(DATA_DIR.resolve())

print("\nClasses:")
for class_name in class_names:
    count = sum(
        1
        for f in (DATA_DIR / class_name).rglob("*")
        if f.is_file()
        and f.suffix.lower()
        in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )
    print(f"  {class_name}: {count}")


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# CALCULATE CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

y_train = np.concatenate(
    [labels.numpy() for _, labels in train_ds],
    axis=0,
)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=y_train,
)

class_weights = {
    int(i): float(weight)
    for i, weight in enumerate(class_weights_array)
}

print("\nClass weights:")
for i, class_name in enumerate(class_names):
    print(f"  {class_name}: {class_weights[i]:.4f}")


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ],
    name="data_augmentation",
)


# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding MobileNetV2 species classifier...")

base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)

# Stage 1:
# Freeze pretrained feature extractor.
base_model.trainable = False


inputs = layers.Input(shape=IMG_SIZE + (3,), name="image")

x = data_augmentation(inputs)
x = preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

x = layers.Dense(
    128,
    activation="relu",
    name="species_features",
)(x)

x = layers.Dropout(0.25)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax",
    name="species_output",
)(x)

model = models.Model(
    inputs,
    outputs,
    name="malaria_species_mobilenetv2",
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy",
    ],
)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
]


# ============================================================
# TRAIN
# ============================================================

print("\n========================================")
print("STARTING TRAINING")
print("========================================")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "model": "MobileNetV2",
    "task": "malaria_species_classification",
    "classes": class_names,
    "image_size": list(IMG_SIZE),
    "batch_size": BATCH_SIZE,
    "validation_split": VALIDATION_SPLIT,
    "seed": SEED,
    "class_weights": class_weights,
    "learning_rate": LEARNING_RATE,
}

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)


# ============================================================
# VALIDATION EVALUATION
# ============================================================

print("\n========================================")
print("VALIDATION EVALUATION")
print("========================================")

val_loss, val_accuracy = model.evaluate(
    val_ds,
    verbose=1,
)

print(f"\nValidation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_accuracy:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

y_true = []
y_pred = []

for images, labels in val_ds:
    probabilities = model.predict(
        images,
        verbose=0,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predictions)


print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
)

print("\nConfusion matrix:")
print(cm)

np.savetxt(
    REPORT_DIR / "species_confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d",
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = REPORT_DIR / "species_training_history.json"

with open(history_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            key: [float(v) for v in values]
            for key, values in history.history.items()
        },
        f,
        indent=2,
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print(f"\nBest model:")
print(MODEL_PATH.resolve())

print("\nMetadata:")
print(META_PATH.resolve())

print("\nValidation report:")
print(REPORT_DIR.resolve())

print("\nSpecies classes:")
for i, class_name in enumerate(class_names):
    print(f"  {i}: {class_name}")

print("\nNext step:")
print("Evaluate per-species performance before integrating")
print("this model into the parasite detection pipeline.")