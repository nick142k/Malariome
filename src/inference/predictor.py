"""
MalariaAI inference adapter.

Two-stage pipeline:

    Image
      |
      v
    MobileNetV2 parasite detector
      |
      +---- UNINFECTED
      |
      +---- PARASITIZED
                |
                v
          Species classifier
                |
                +-- Falciparum
                +-- Malariae
                +-- Ovale
                +-- Vivax

Predictor contract:

    predictor.predict(image_bytes) -> dict
    predictor.explain(image_bytes) -> dict
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data.preprocessing import Preprocessor


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
    """Read JSON metadata safely."""

    if not path.exists():
        return {}

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return {}


def _load_image(
    image_bytes: bytes,
    preprocessor: Preprocessor,
) -> np.ndarray:
    """
    Preprocess an uploaded image using the exact
    preprocessing configuration used by training.

    Returns:
        NumPy array with shape:

            (1, 224, 224, 3)
    """

    array = preprocessor(
        image_bytes
    )

    return np.expand_dims(
        array,
        axis=0,
    )


def _extract_threshold(meta: dict) -> float:
    """Extract parasite decision threshold from metadata."""

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

            except (
                TypeError,
                ValueError,
            ):
                pass

    return DEFAULT_PARASITE_THRESHOLD


# ---------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------

class MalariaPredictor:
    """
    Two-stage malaria prediction system.

    Stage 1:
        transfer_model.keras
        -> PARASITIZED / UNINFECTED

    Stage 2:
        species_model.keras
        -> malaria species
    """

    model_name = (
        "MalariaAI Two-Stage MobileNetV2"
    )

    def __init__(
        self,
        cfg=None,
    ):

        self.cfg = cfg

        # -------------------------------------------------------------
        # Validate model files
        # -------------------------------------------------------------

        if not PARASITE_MODEL_PATH.exists():

            raise FileNotFoundError(
                "Parasite model not found: "
                f"{PARASITE_MODEL_PATH}"
            )

        if not SPECIES_MODEL_PATH.exists():

            raise FileNotFoundError(
                "Species model not found: "
                f"{SPECIES_MODEL_PATH}"
            )

        # -------------------------------------------------------------
        # Load parasite detector
        # -------------------------------------------------------------

        print(
            "[MalariaAI] Loading parasite detector..."
        )

        self.parasite_model = (
            tf.keras.models.load_model(
                PARASITE_MODEL_PATH,
                compile=False,
            )
        )

        # -------------------------------------------------------------
        # Load species classifier
        # -------------------------------------------------------------

        print(
            "[MalariaAI] Loading species classifier..."
        )

        self.species_model = (
            tf.keras.models.load_model(
                SPECIES_MODEL_PATH,
                compile=False,
            )
        )

        # -------------------------------------------------------------
        # Load metadata
        # -------------------------------------------------------------

        self.parasite_meta = _read_json(
            PARASITE_META_PATH
        )

        self.species_meta = _read_json(
            SPECIES_META_PATH
        )

        # -------------------------------------------------------------
        # Build preprocessing
        # -------------------------------------------------------------

        if "preprocessing" not in self.parasite_meta:

            raise ValueError(
                "Parasite model metadata does not contain "
                "preprocessing configuration."
            )

        self.parasite_preprocessor = (
            Preprocessor.from_metadata(
                self.parasite_meta["preprocessing"],
                allowed_extensions=(
                    "png",
                    "jpg",
                    "jpeg",
                ),
                max_bytes=5 * 1024 * 1024,
            )
        )

        # -------------------------------------------------------------
        # Threshold
        # -------------------------------------------------------------

        self.threshold = _extract_threshold(
            self.parasite_meta
        )

        # -------------------------------------------------------------
        # Species classes
        # -------------------------------------------------------------

        self.species_classes = (
            self._load_species_classes()
        )

        # -------------------------------------------------------------
        # Startup information
        # -------------------------------------------------------------

        print(
            "[MalariaAI] Models loaded successfully."
        )

        print(
            "[MalariaAI] Parasite threshold: "
            f"{self.threshold}"
        )

        print(
            "[MalariaAI] Parasite preprocessing: "
            f"{self.parasite_meta['preprocessing']}"
        )

        print(
            "[MalariaAI] Species classes: "
            f"{self.species_classes}"
        )

    # -----------------------------------------------------------------
    # Species metadata
    # -----------------------------------------------------------------

    def _load_species_classes(self):

        possible = (
            self.species_meta.get("classes")
            or self.species_meta.get("class_names")
            or self.species_meta.get("species_classes")
        )

        if (
            isinstance(possible, list)
            and len(possible) == 4
        ):

            return [
                str(x)
                for x in possible
            ]

        return SPECIES_CLASSES.copy()

    # -----------------------------------------------------------------
    # Parasite prediction
    # -----------------------------------------------------------------

    def _predict_parasite(
        self,
        image,
    ):

        prediction = (
            self.parasite_model.predict(
                image,
                verbose=0,
            )
        )

        prediction = np.asarray(
            prediction
        )

        # -------------------------------------------------------------
        # Softmax output
        # -------------------------------------------------------------

        if (
            prediction.ndim == 2
            and prediction.shape[1] == 2
        ):

            malaria_probability = float(
                prediction[0, 1]
            )

        # -------------------------------------------------------------
        # Sigmoid output
        # -------------------------------------------------------------

        elif prediction.size == 1:

            malaria_probability = float(
                prediction.reshape(-1)[0]
            )

        else:

            raise ValueError(
                "Unexpected parasite model "
                f"output shape: {prediction.shape}"
            )

        return float(
            np.clip(
                malaria_probability,
                0.0,
                1.0,
            )
        )

    # -----------------------------------------------------------------
    # Species prediction
    # -----------------------------------------------------------------

    def _predict_species(
        self,
        image,
    ):

        prediction = (
            self.species_model.predict(
                image,
                verbose=0,
            )
        )

        prediction = np.asarray(
            prediction
        )

        if (
            prediction.ndim != 2
            or prediction.shape[1] != 4
        ):

            raise ValueError(
                "Unexpected species model "
                f"output shape: {prediction.shape}"
            )

        probabilities = prediction[0]

        index = int(
            np.argmax(
                probabilities
            )
        )

        probability = float(
            probabilities[index]
        )

        return (
            self.species_classes[index],
            probability,
            probabilities.tolist(),
        )

    # -----------------------------------------------------------------
    # Public prediction API
    # -----------------------------------------------------------------

    def predict(
        self,
        image_bytes: bytes,
    ) -> dict:
        """
        Run the complete two-stage prediction pipeline.
        """

        image = _load_image(
            image_bytes,
            self.parasite_preprocessor,
        )

        # -------------------------------------------------------------
        # Stage 1: parasite detection
        # -------------------------------------------------------------

        malaria_probability = (
            self._predict_parasite(
                image
            )
        )

        is_parasitized = (
            malaria_probability
            >= self.threshold
        )

        prediction = (
            "PARASITIZED"
            if is_parasitized
            else "UNINFECTED"
        )

        result = {

            "prediction": prediction,

            "malaria_probability": round(
                malaria_probability,
                6,
            ),

            "threshold": self.threshold,

            "model": (
                "MobileNetV2 transfer_model"
            ),

            "model_version": (
                self.parasite_meta.get(
                    "model_version",
                    "transfer_model",
                )
            ),
        }

        # -------------------------------------------------------------
        # Stage 2: species classification
        # -------------------------------------------------------------

        if is_parasitized:

            (
                species,
                species_probability,
                species_probabilities,
            ) = self._predict_species(
                image
            )

            result["species"] = species

            result["species_probability"] = (
                round(
                    species_probability,
                    6,
                )
            )

            result[
                "species_probabilities"
            ] = {

                name: round(
                    float(prob),
                    6,
                )

                for name, prob in zip(
                    self.species_classes,
                    species_probabilities,
                )
            }

        else:

            result["species"] = None

            result[
                "species_probability"
            ] = None

        return result

    # -----------------------------------------------------------------
    # Grad-CAM
    # -----------------------------------------------------------------

    def explain(
        self,
        image_bytes: bytes,
    ):
        """
        Generate Grad-CAM explanation.

        Returns:

            {
                "original": base64 PNG,
                "heatmap": base64 PNG,
                "overlay": base64 PNG
            }
        """

        from src.explainability.gradcam import (
            make_gradcam_heatmap,
            save_gradcam,
        )

        # -------------------------------------------------------------
        # Preprocess exactly like prediction
        # -------------------------------------------------------------

        image = _load_image(
            image_bytes,
            self.parasite_preprocessor,
        )

        # -------------------------------------------------------------
        # Generate Grad-CAM
        #
        # IMPORTANT:
        #
        # make_gradcam_heatmap() signature is:
        #
        #     make_gradcam_heatmap(
        #         image_tensor,
        #         model,
        #         conv_layer_name,
        #     )
        #
        # Verified MobileNetV2 feature layer:
        #
        #     out_relu
        # -------------------------------------------------------------

        heatmap = make_gradcam_heatmap(
            image,
            self.parasite_model,
            "out_relu",
        )

        # -------------------------------------------------------------
        # Output directory
        # -------------------------------------------------------------

        output_dir = (
            PROJECT_ROOT
            / "reports"
            / "explainability"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_dir
            / "flask_gradcam.png"
        )

        # -------------------------------------------------------------
        # Create overlay
        # -------------------------------------------------------------

        result = save_gradcam(
            image_bytes,
            heatmap,
            output_path,
        )

        return result


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------

def build_predictor(
    cfg=None,
):
    """
    Factory used by app.py.
    """

    return MalariaPredictor(
        cfg
    )
