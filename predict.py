import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path("models/transfer_model.keras")
THRESHOLD = 0.42
IMAGE_SIZE = (224, 224)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict(image_path):
    model = load_model()
    x = preprocess(image_path)

    probability = float(model.predict(x, verbose=0).reshape(-1)[0])

    if probability >= THRESHOLD:
        label = "PARASITIZED"
    else:
        label = "UNINFECTED"

    return {
        "image": str(image_path),
        "prediction": label,
        "probability": round(probability, 6),
        "threshold": THRESHOLD,
        "model": "MobileNetV2 transfer_model",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Malaria cell image inference"
    )
    parser.add_argument("image", help="Path to cell image")
    args = parser.parse_args()

    result = predict(args.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
