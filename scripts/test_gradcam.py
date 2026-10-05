from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocessing import Preprocessor
from src.explainability.gradcam import (
    find_last_conv_layer,
    make_gradcam_heatmap,
    save_gradcam,
)


MODEL_PATH = ROOT / "models" / "transfer_model.keras"
OUTPUT_DIR = ROOT / "reports" / "explainability"


def main():
    print("=" * 60)
    print("Malariome Grad-CAM verification")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load model
    # ---------------------------------------------------------
    print("[1/6] Loading MobileNetV2...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print("[PASS] Model loaded")

    # ---------------------------------------------------------
    # 2. Find Grad-CAM layer
    # ---------------------------------------------------------
    print("[2/6] Finding convolutional feature layer...")

    conv_layer = find_last_conv_layer(model)

    print(f"[PASS] Grad-CAM layer: {conv_layer.name}")
    print(f"       Output shape: {conv_layer.output.shape}")

    # ---------------------------------------------------------
    # 3. Select an image
    # ---------------------------------------------------------
    print("[3/6] Searching for a test image...")

    image_candidates = list(
        (ROOT / "data" / "raw" / "cell_images" / "Parasitized").glob("*.png")
    )

    if not image_candidates:
        image_candidates = list(
            (ROOT / "data" / "raw" / "cell_images" / "Uninfected").glob("*.png")
        )

    if not image_candidates:
        raise FileNotFoundError(
            "No PNG images found in the dataset."
        )

    image_path = image_candidates[0]

    print(f"[PASS] Image: {image_path}")

    # ---------------------------------------------------------
    # 4. Use EXACT model preprocessing
    # ---------------------------------------------------------
    print("[4/6] Applying MobileNetV2 preprocessing...")

    meta_path = ROOT / "models" / "transfer_model.meta.json"

    import json

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    preprocessor = Preprocessor.from_metadata(
        meta["preprocessing"],
        allowed_extensions=("png", "jpg", "jpeg"),
        max_bytes=5 * 1024 * 1024,
    )

    image_tensor = preprocessor(image_path)

    # Model requires batch dimension.
    image_batch = np.expand_dims(
        image_tensor,
        axis=0,
    ).astype(np.float32)

    print(f"[PASS] Tensor shape: {image_batch.shape}")
    print(
        f"       Range: {image_batch.min():.4f} "
        f"to {image_batch.max():.4f}"
    )

    # ---------------------------------------------------------
    # 5. Prediction
    # ---------------------------------------------------------
    print("[5/6] Running prediction...")

    prediction = model.predict(
        image_batch,
        verbose=0,
    )

    if prediction.shape[-1] == 1:
        probability = float(prediction[0][0])
    else:
        probability = float(prediction[0][1])

    print(
        f"[PASS] Parasitized probability: "
        f"{probability:.4f}"
    )

    # ---------------------------------------------------------
    # 6. Grad-CAM
    # ---------------------------------------------------------
    print("[6/6] Generating Grad-CAM...")

    heatmap = make_gradcam_heatmap(
        image_batch,
        model,
        
    )

    if heatmap.ndim != 2:
        raise RuntimeError(
            f"Unexpected heatmap shape: {heatmap.shape}"
        )

    if not np.isfinite(heatmap).all():
        raise RuntimeError(
            "Grad-CAM heatmap contains NaN or infinite values."
        )

    if heatmap.min() < 0 or heatmap.max() > 1:
        raise RuntimeError(
            "Grad-CAM heatmap is outside [0, 1]."
        )

    print(
        f"[PASS] Heatmap shape: {heatmap.shape}"
    )

    original = Image.open(image_path).convert("RGB")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR /
        f"gradcam_{image_path.stem}.png"
    )

    save_gradcam(
        original,
        heatmap,
        output_path,
        alpha=0.40,
    )

    print(f"[PASS] Saved: {output_path}")

    print()
    print("=" * 60)
    print("Grad-CAM verification completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()