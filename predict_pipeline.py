"""
Integrated Malaria Prediction Pipeline

Pipeline:
    Cell image
        ↓
    Parasite detector (transfer_model)
        ↓
    If parasitized → species classifier
        ↓
    Falciparum / Malariae / Ovale / Vivax

Usage:
    python predict_pipeline.py "path\\to\\cell.png"
"""

from pathlib import Path
import json
import sys

import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PARASITE_MODEL = BASE_DIR / "models" / "transfer_model.keras"
SPECIES_MODEL = BASE_DIR / "models" / "species_model.keras"

IMAGE_SIZE = (224, 224)

SPECIES_CLASSES = [
    "Falciparum",
    "Malariae",
    "Ovale",
    "Vivax",
]

# This is the frozen validation threshold used by your
# existing parasite detector.
PARASITE_THRESHOLD = 0.42


# ============================================================
# MODEL LOADING
# ============================================================

def load_models():
    """Load both trained models."""

    if not PARASITE_MODEL.exists():
        raise FileNotFoundError(
            f"Parasite model not found: {PARASITE_MODEL}"
        )

    if not SPECIES_MODEL.exists():
        raise FileNotFoundError(
            f"Species model not found: {SPECIES_MODEL}"
        )

    parasite_model = tf.keras.models.load_model(PARASITE_MODEL)
    species_model = tf.keras.models.load_model(SPECIES_MODEL)

    return parasite_model, species_model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_image(image_path):
    """Load and preprocess an image for the models."""

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image = tf.keras.utils.img_to_array(image)

    image = image.astype(np.float32)

    image = np.expand_dims(image, axis=0)

    return image


# ============================================================
# PARASITE PREDICTION
# ============================================================

def predict_parasite(model, image):
    """
    Predict whether the cell is parasitized.

    The existing transfer model outputs the probability
    of the positive/parasitized class.
    """

    output = model.predict(image, verbose=0)

    probability = float(np.asarray(output).reshape(-1)[0])

    if probability >= PARASITE_THRESHOLD:
        prediction = "PARASITIZED"
    else:
        prediction = "UNINFECTED"

    return prediction, probability


# ============================================================
# SPECIES PREDICTION
# ============================================================

def predict_species(model, image):
    """Predict malaria species."""

    probabilities = model.predict(image, verbose=0)[0]

    class_index = int(np.argmax(probabilities))

    species = SPECIES_CLASSES[class_index]

    probability = float(probabilities[class_index])

    all_probabilities = {
        SPECIES_CLASSES[i]: float(probabilities[i])
        for i in range(len(SPECIES_CLASSES))
    }

    return species, probability, all_probabilities


# ============================================================
# FULL PIPELINE
# ============================================================

def run_pipeline(image_path):

    parasite_model, species_model = load_models()

    image = load_image(image_path)

    # --------------------------------------------------------
    # Stage 1: Parasite detection
    # --------------------------------------------------------

    parasite_status, parasite_probability = predict_parasite(
        parasite_model,
        image
    )

    result = {
        "image": str(Path(image_path).resolve()),
        "parasite_status": parasite_status,
        "parasite_probability": round(parasite_probability, 6),
        "parasite_threshold": PARASITE_THRESHOLD,
        "species": None,
        "species_probability": None,
        "species_probabilities": None,
    }

    # --------------------------------------------------------
    # Stage 2: Species identification
    # --------------------------------------------------------

    if parasite_status == "PARASITIZED":

        species, species_probability, all_probabilities = (
            predict_species(
                species_model,
                image
            )
        )

        result["species"] = species
        result["species_probability"] = round(
            species_probability,
            6
        )

        result["species_probabilities"] = {
            key: round(value, 6)
            for key, value in all_probabilities.items()
        }

    return result


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            '  python predict_pipeline.py "path\\to\\image.png"'
        )
        sys.exit(1)

    image_path = sys.argv[1]

    try:

        result = run_pipeline(image_path)

        print()
        print("=" * 55)
        print("MALARIA PREDICTION PIPELINE")
        print("=" * 55)

        print(f"Image: {result['image']}")

        print()
        print(
            f"Parasite status: "
            f"{result['parasite_status']}"
        )

        print(
            f"Parasite probability: "
            f"{result['parasite_probability']:.4f}"
        )

        if result["parasite_status"] == "PARASITIZED":

            print()
            print(
                f"Species: "
                f"{result['species']}"
            )

            print(
                f"Species probability: "
                f"{result['species_probability']:.4f}"
            )

            print()
            print("Species probabilities:")

            for species, probability in (
                result["species_probabilities"].items()
            ):
                print(
                    f"  {species:<12} "
                    f"{probability:.4f}"
                )

        else:

            print()
            print(
                "Species classification skipped "
                "(cell classified as uninfected)."
            )

        print()
        print("=" * 55)

        print()
        print(json.dumps(result, indent=2))

    except Exception as exc:

        print()
        print("ERROR:")
        print(str(exc))

        sys.exit(1)


if __name__ == "__main__":
    main()
