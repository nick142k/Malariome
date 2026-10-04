"""
MalariaAI inference adapter.

Pipeline:
    Image
      |
      v
Parasite detector (MobileNetV2 transfer_model)
      |
      +---- UNINFECTED -> stop
      |
      +---- PARASITIZED -> species classifier
                              |
                              +-- Falciparum
                              +-- Malariae
                              +-- Ovale
                              +-- Vivax

This module provides the Predictor contract expected by app.py:

    predictor.predict(image_bytes) -> dict
    predictor.explain(image_bytes) -> optional dict
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

PARASITE_MODEL_PATH = MODELS_DIR / "transfer_model.keras"
SPECIES_MODEL_PATH = MODELS_DIR / "species_model.keras"

PARASITE_META_PATH = MODELS_DIR / "transfer_model.meta.json"
SPECIES_META_PATH = MODELS_DIR / "species_model.meta.json"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

IMAGE_SIZE = (224, 224)

DEFAULT_PARASITE_THRESHOLD = 0.42

SPECIES_CLASSES = [
    "Falciparum",
    "Malariae",
    "Ovale",
    "Vivax",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode image bytes and convert to RGB float32 tensor.

    MobileNetV2 preprocessing is applied inside the model pipeline
    for the species model. For the parasite model we use the same
    224x224 RGB input convention.
    """

    if not image_bytes:
        raise ValueError("Image data is empty.")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)
    except Exception as exc:
        raise ValueError("Unable to decode uploaded image.") from exc

    array = np.asarray(image, dtype=np.float32)

    # Add batch dimension.
    return np.expand_dims(array, axis=0)


def _extract_threshold(meta: dict) -> float:
    """
    Extract the frozen parasite threshold from metadata.

    Falls back to 0.42, which is the validation-selected threshold
    already used by the existing predict.py.
    """

    possible_keys = [
        "threshold",
        "decision_threshold",
        "classification_threshold",
    ]

    for key in possible_keys:
        value = meta.get(key)

        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    return DEFAULT_PARASITE_THRESHOLD


# ---------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------

class MalariaPredictor:
    """
    Two-stage malaria inference pipeline.

    Stage 1:
        transfer_model.keras
        -> parasitized / uninfected

    Stage 2:
        species_model.keras
        -> species, only for parasitized images
    """

    model_name = "MalariaAI Two-Stage MobileNetV2"

    def __init__(self, cfg=None):
        self.cfg = cfg

        if not PARASITE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Parasite model not found: {PARASITE_MODEL_PATH}"
            )

        if not SPECIES_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Species model not found: {SPECIES_MODEL_PATH}"
            )

        print("[MalariaAI] Loading parasite detector...")
        self.parasite_model = tf.keras.models.load_model(
            PARASITE_MODEL_PATH,
            compile=False,
        )

        print("[MalariaAI] Loading species classifier...")
        self.species_model = tf.keras.models.load_model(
            SPECIES_MODEL_PATH,
            compile=False,
        )

        self.parasite_meta = _read_json(PARASITE_META_PATH)
        self.species_meta = _read_json(SPECIES_META_PATH)

        self.threshold = _extract_threshold(self.parasite_meta)

        self.species_classes = self._load_species_classes()

        print("[MalariaAI] Models loaded successfully.")
        print(f"[MalariaAI] Parasite threshold: {self.threshold}")
        print(f"[MalariaAI] Species classes: {self.species_classes}")

    # -----------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------

    def _load_species_classes(self):
        """
        Prefer the class order saved by training metadata.
        Otherwise use the known training order.
        """

        possible = (
            self.species_meta.get("classes")
            or self.species_meta.get("class_names")
            or self.species_meta.get("species_classes")
        )

        if isinstance(possible, list) and len(possible) == 4:
            return [str(x) for x in possible]

        return SPECIES_CLASSES.copy()

    # -----------------------------------------------------------------
    # Parasite prediction
    # -----------------------------------------------------------------

    def _predict_parasite(self, image):
        """
        Return parasite probability.

        The existing transfer_model predicts a binary malaria class.
        We treat its single sigmoid output as the probability of
        PARASITIZED.
        """

        prediction = self.parasite_model.predict(
            image,
            verbose=0,
        )

        prediction = np.asarray(prediction)

        # Handle common binary model output formats:
        #
        # (1, 1) -> sigmoid probability
        # (1,)   -> sigmoid probability
        # (1, 2) -> softmax probability of class 1
        if prediction.ndim == 2 and prediction.shape[1] == 2:
            malaria_probability = float(prediction[0, 1])

        elif prediction.size == 1:
            malaria_probability = float(prediction.reshape(-1)[0])

        else:
            raise ValueError(
                f"Unexpected parasite model output shape: {prediction.shape}"
            )

        return float(np.clip(malaria_probability, 0.0, 1.0))

    # -----------------------------------------------------------------
    # Species prediction
    # -----------------------------------------------------------------

    def _predict_species(self, image):
        """
        Predict one of:

            Falciparum
            Malariae
            Ovale
            Vivax
        """

        prediction = self.species_model.predict(
            image,
            verbose=0,
        )

        prediction = np.asarray(prediction)

        if prediction.ndim != 2 or prediction.shape[1] != 4:
            raise ValueError(
                f"Unexpected species model output shape: {prediction.shape}"
            )

        probabilities = prediction[0]

        index = int(np.argmax(probabilities))
        probability = float(probabilities[index])

        return (
            self.species_classes[index],
            probability,
            probabilities.tolist(),
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def predict(self, image_bytes: bytes) -> dict:
        """
        Main Predictor contract used by Flask.

        Returns:
            prediction
            malaria_probability
            threshold
            model
            model_version
            species
            species_probability
        """

        image = _load_image(image_bytes)

        # -------------------------------------------------------------
        # Stage 1: parasite detection
        # -------------------------------------------------------------

        malaria_probability = self._predict_parasite(image)

        is_parasitized = malaria_probability >= self.threshold

        if is_parasitized:
            prediction = "PARASITIZED"
        else:
            prediction = "UNINFECTED"

        result = {
            "prediction": prediction,
            "malaria_probability": round(malaria_probability, 6),
            "threshold": self.threshold,
            "model": "MobileNetV2 transfer_model",
            "model_version": self.parasite_meta.get(
                "model_version",
                "transfer_model",
            ),
        }

        # -------------------------------------------------------------
        # Stage 2: species classification
        #
        # IMPORTANT:
        # Only classify species after parasite detection.
        # -------------------------------------------------------------

        if is_parasitized:

            species, species_probability, species_probabilities = (
                self._predict_species(image)
            )

            result["species"] = species
            result["species_probability"] = round(
                species_probability,
                6,
            )

            # Useful for API/debugging.
            result["species_probabilities"] = {
                name: round(float(prob), 6)
                for name, prob in zip(
                    self.species_classes,
                    species_probabilities,
                )
            }

        else:
            result["species"] = None
            result["species_probability"] = None

        return result

    # -----------------------------------------------------------------
    # Optional explainability
    # -----------------------------------------------------------------

    def explain(self, image_bytes: bytes):
        """
        Placeholder for Grad-CAM.

        The Flask application already treats explain() as optional.
        Returning None means the application can run normally without
        Grad-CAM integration.

        We intentionally do not fabricate a heatmap.
        """

        return None


# ---------------------------------------------------------------------
# Factory expected by app.py
# ---------------------------------------------------------------------

def build_predictor(cfg=None):
    """
    Factory used by:

        from src.inference.predictor import build_predictor
    """

    return MalariaPredictor(cfg)
