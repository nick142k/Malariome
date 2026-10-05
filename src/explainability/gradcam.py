"""Grad-CAM explainability for the final malaria detector."""

from __future__ import annotations

from pathlib import Path

from matplotlib.pyplot import colormaps
import numpy as np
import tensorflow as tf
from PIL import Image


def _find_nested_conv_layer(model):
    """
    Find the final 4-D feature layer inside a nested backbone.

    The saved malaria detector contains MobileNetV2 as a nested
    Functional model. Grad-CAM must obtain gradients through that
    nested model rather than treating its internal layer as a direct
    output of the outer model.
    """

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for nested_layer in reversed(layer.layers):
                try:
                    output_shape = nested_layer.output.shape

                    if len(output_shape) == 4:
                        return layer, nested_layer
                except (AttributeError, TypeError):
                    continue

    raise ValueError(
        "Could not find a suitable 4-D convolutional feature layer."
    )


def find_last_conv_layer(model):
    """
    Find the final convolutional feature layer.

    For the Malariome MobileNetV2 model this returns the internal
    MobileNetV2 feature layer, typically ``out_relu``.
    """

    _, conv_layer = _find_nested_conv_layer(model)
    return conv_layer


def _get_gradcam_backbone(model, conv_layer_name=None):
    """
    Return the nested backbone and the requested internal feature layer.
    """

    nested_models = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.Model)
    ]

    if not nested_models:
        raise ValueError(
            "Could not find a nested feature-extraction model."
        )

    # The Malariome model has MobileNetV2 as its feature extractor.
    backbone = nested_models[0]

    if conv_layer_name is None:
        # Prefer the final ReLU feature map.
        for preferred_name in (
            "out_relu",
            "Conv_1",
            "Conv_1_bn",
        ):
            try:
                return backbone, backbone.get_layer(preferred_name)
            except ValueError:
                pass

        # Generic fallback.
        for layer in reversed(backbone.layers):
            try:
                if len(layer.output.shape) == 4:
                    return backbone, layer
            except (AttributeError, TypeError):
                continue

        raise ValueError(
            "Could not find a suitable 4-D feature layer."
        )

    # Explicit layer name may refer to the nested backbone.
    try:
        return backbone, backbone.get_layer(conv_layer_name)
    except ValueError:
        pass

    # It may be useful to pass the outer nested model name.
    try:
        outer_layer = model.get_layer(conv_layer_name)

        if isinstance(outer_layer, tf.keras.Model):
            return outer_layer, outer_layer.layers[-1]

    except ValueError:
        pass

    raise ValueError(
        f"Could not find Grad-CAM layer '{conv_layer_name}'."
    )


def make_gradcam_heatmap(
    image_tensor: np.ndarray,
    model: tf.keras.Model,
    conv_layer_name: str | None = None,
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for the Parasitized class.

    image_tensor:
        Shape (1, H, W, 3), already preprocessed exactly as required
        by the model.

    Returns:
        Float32 heatmap in range [0, 1].
    """

    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
        raise ValueError(
            "Expected image tensor shape (1, H, W, 3), "
            f"got {image_tensor.shape}"
        )

    image_tensor = tf.convert_to_tensor(
        image_tensor,
        dtype=tf.float32,
    )

    backbone, conv_layer = _get_gradcam_backbone(
        model,
        conv_layer_name,
    )

    print(
        f"[Grad-CAM] Backbone: {backbone.name}"
    )
    print(
        f"[Grad-CAM] Feature layer: {conv_layer.name}"
    )
    print(
        f"[Grad-CAM] Feature shape: {conv_layer.output.shape}"
    )

    # ------------------------------------------------------------------
    # IMPORTANT:
    #
    # The outer saved model contains:
    #
    # image -> MobileNetV2 -> GAP -> Dropout -> Dense
    #
    # We need gradients through the actual backbone computation.
    #
    # Calling the backbone inside the GradientTape keeps the feature
    # activations and final prediction connected to the same graph.
    # ------------------------------------------------------------------

    with tf.GradientTape() as tape:
        # Run the image through the nested MobileNetV2.
        features = backbone(
            image_tensor,
            training=False,
        )

        # ``features`` should be the final MobileNetV2 feature map.
        # For this model it is (1, 7, 7, 1280).
        tape.watch(features)

        # Reproduce the classification head from the outer model.
        #
        # The outer model has:
        #   gap
        #   drop_head
        #   prob_parasitized
        #
        # We obtain these layers directly from the saved model.
        gap = model.get_layer("gap")
        drop_head = model.get_layer("drop_head")
        classifier = model.get_layer("prob_parasitized")

        pooled = gap(features)

        # Dropout is disabled during inference.
        dropped = drop_head(
            pooled,
            training=False,
        )

        predictions = classifier(dropped)

        if predictions.shape[-1] == 1:
            class_score = predictions[:, 0]
        else:
            class_score = predictions[:, 1]

    gradients = tape.gradient(
        class_score,
        features,
    )

    if gradients is None:
        raise RuntimeError(
            "Gradients could not be computed. "
            "The selected feature tensor is not connected "
            "to the classification score."
        )

    # ------------------------------------------------------------------
    # Global-average-pool the gradients.
    # ------------------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2),
    )

    features = features[0]
    pooled_gradients = pooled_gradients[0]

    # Weight each feature map by its average gradient.
    heatmap = tf.reduce_sum(
        features * pooled_gradients,
        axis=-1,
    )

    # Retain only positive contributions.
    heatmap = tf.maximum(
        heatmap,
        0,
    )

    # Normalize to [0, 1].
    max_value = tf.reduce_max(heatmap)

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        tf.zeros_like(heatmap),
    )

    return heatmap.numpy().astype(np.float32)


def resize_heatmap(
    heatmap: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    """Resize heatmap to (width, height)."""

    heatmap_uint8 = np.uint8(
        np.clip(heatmap, 0, 1) * 255
    )

    image = Image.fromarray(
        heatmap_uint8,
        mode="L",
    )

    image = image.resize(
        size,
        Image.Resampling.BILINEAR,
    )

    return np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0


def create_overlay(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.40,
) -> Image.Image:
    """Create a heatmap overlay on the original RGB image."""

    original_image = original_image.convert("RGB")

    heatmap_resized = resize_heatmap(
        heatmap,
        original_image.size,
    )

    # Use matplotlib only for the colormap.
    from matplotlib import colormaps
    colored = colormaps["jet"](heatmap_resized)


    colored = np.uint8(
        colored[:, :, :3] * 255
    )

    heatmap_image = Image.fromarray(
        colored,
        mode="RGB",
    )

    return Image.blend(
        original_image,
        heatmap_image,
        alpha=float(alpha),
    )


def save_gradcam(
    original_image,
    heatmap,
    output_path,
    alpha=0.45,
):
    """
    Create Grad-CAM heatmap + overlay.

    Saves the overlay to output_path and returns
    base64 PNG images for Flask.
    """

    import base64
    import io
    from pathlib import Path

    import numpy as np
    import tensorflow as tf
    from PIL import Image
    from matplotlib import colormaps

    # -------------------------------------------------------------
    # Load original image
    # -------------------------------------------------------------

    if isinstance(original_image, Image.Image):
    # Already a PIL Image
        original = original_image.convert("RGB")

    elif isinstance(original_image, (bytes, bytearray)):
        # Raw image bytes
        original = Image.open(
            io.BytesIO(original_image)
        ).convert("RGB")    
    else:
        # File path
        original = Image.open(
            original_image
        ).convert("RGB")


    original = original.resize(
        (224, 224),
        Image.Resampling.BILINEAR,
    )

    # -------------------------------------------------------------
    # Convert heatmap to numpy
    # -------------------------------------------------------------

    if tf.is_tensor(heatmap):
        heatmap = heatmap.numpy()

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32,
    )

    # Remove unnecessary dimensions
    heatmap = np.squeeze(heatmap)

    # Normalize
    heatmap = np.maximum(
        heatmap,
        0,
    )

    max_value = np.max(heatmap)

    if max_value > 0:
        heatmap = heatmap / max_value

    heatmap = np.clip(
        heatmap,
        0.0,
        1.0,
    )

    # -------------------------------------------------------------
    # Resize heatmap to image size
    # -------------------------------------------------------------

    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        (224, 224),
        method="bilinear",
    ).numpy()[..., 0]

    heatmap_resized = np.clip(
        heatmap_resized,
        0.0,
        1.0,
    )

    # -------------------------------------------------------------
    # Apply JET colormap
    # -------------------------------------------------------------

    cmap = colormaps["jet"]

    colored = cmap(
        heatmap_resized
    )[..., :3]

    colored = (
        colored * 255
    ).astype(np.uint8)

    heatmap_image = Image.fromarray(
        colored,
        mode="RGB",
    )

    # -------------------------------------------------------------
    # Create overlay
    # -------------------------------------------------------------

    overlay = Image.blend(
        original,
        heatmap_image,
        alpha=alpha,
    )

    # -------------------------------------------------------------
    # Make output directory
    # -------------------------------------------------------------

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Save overlay
    # -------------------------------------------------------------

    overlay.save(
        output_path,
        format="PNG",
    )

    print(
        f"[Grad-CAM] Saved overlay: {output_path}"
    )

    # -------------------------------------------------------------
    # PNG -> base64
    # -------------------------------------------------------------

    def png_base64(img):

        buffer = io.BytesIO()

        img.save(
            buffer,
            format="PNG",
        )

        return base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

    # -------------------------------------------------------------
    # Return Flask-compatible result
    # -------------------------------------------------------------

    return {
        "original": png_base64(original),
        "heatmap": png_base64(heatmap_image),
        "overlay": png_base64(overlay),
    }


