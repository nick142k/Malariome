"""Stage 2 of the cascade: species classifier (models/species_model.keras, trained by train_species.py).

That model has augmentation and MobileNetV2 `preprocess_input` BUILT IN, so it expects raw 0-255 pixels stretched to
224x224 (the way `image_dataset_from_directory` fed it during training). This is DIFFERENT from the detector, which expects
padded images scaled to [-1, 1]. Keeping the two preprocessing recipes in their own classes is what prevents mix-ups.
Only called by MalariaPredictor when Stage 1 says PARASITIZED.
Status: IMPLEMENTED - NOT YET VERIFIED with the real species model (logic tested with a fake model).
"""
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.preprocessing import load_image

DEFAULT_CLASSES = ("Falciparum", "Malariae", "Ovale", "Vivax")      # alphabetical = training label order


class SpeciesClassifier:
    def __init__(self, model_or_path, classes=DEFAULT_CLASSES, image_size=224):
        if isinstance(model_or_path, (str, Path)):
            import keras
            model_or_path = keras.saving.load_model(str(model_or_path))
        self.model, self.classes, self.size = model_or_path, list(classes), int(image_size)
        self._lock = threading.Lock()

    def _resize(self, rgb):
        try:                                                         # same op as training (bilinear stretch, no padding)
            import tensorflow as tf
            return tf.image.resize(np.asarray(rgb, dtype=np.float32), (self.size, self.size)).numpy().astype("float32")
        except ImportError:
            return np.asarray(rgb.resize((self.size, self.size), Image.BILINEAR), dtype=np.float32)

    def predict(self, source):
        """source: path, bytes, file-like or PIL image. Raises ImageValidationError for bad images."""
        x = self._resize(load_image(source))                         # raw 0..255, NO scaling (the model scales itself)
        with self._lock:
            p = np.asarray(self.model(x[None], training=False), dtype=float).ravel()
        i = int(np.argmax(p))
        return {"species": f"P. {self.classes[i].lower()}", "species_probability": float(p[i])}