"""Shared image preprocessing: the SAME code path is used by training, evaluation and inference.

Pipeline:  load + validate -> RGB -> pad to square (black) -> resize -> float32 in 0..255
           -> model-specific scaling (done separately so augmentation can run on 0..255 pixels).

Why pad with black: the dataset's cell crops sit on a black (masked) background, so black padding
is invisible to the model and keeps the cell's aspect ratio. Everything here is pure Pillow/NumPy
(no TensorFlow), so the Flask app and the tests can use it without loading TF.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

SCALINGS = ("unit", "minus1_1", "none")          # [0,1] | [-1,1] (MobileNetV2) | raw 0..255
RESIZE_MODES = ("pad_square", "stretch")
_ALLOWED_FORMATS = {"PNG", "JPEG"}


class ImageValidationError(ValueError):
    """Raised for any image that cannot be safely preprocessed (message is safe to show to users)."""


def to_rgb(im: Image.Image) -> Image.Image:
    """Convert any Pillow image to RGB. Transparent pixels are composited onto black."""
    if im.mode == "RGB":
        return im
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return im.convert("RGB")


def pad_to_square(im: Image.Image) -> Image.Image:
    """Centre the image on a black square canvas whose side is the longer edge."""
    w, h = im.size
    if w == h:
        return im
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    return canvas


def load_image(source, allowed_extensions=None, max_bytes=None, min_side=16, max_pixels=25_000_000) -> Image.Image:
    """Load and validate an image from a path, bytes, file-like object or PIL image.

    Raises ImageValidationError (never anything else) for bad extension, missing/empty/oversized
    file, non-image or truncated data, unsupported format, or absurd dimensions.
    """
    try:
        if isinstance(source, Image.Image):
            im = source
            data = None
        else:
            if isinstance(source, (str, Path)):
                p = Path(source)
                if allowed_extensions is not None:
                    allowed = {e.lower().lstrip(".") for e in allowed_extensions}
                    if p.suffix.lower().lstrip(".") not in allowed:
                        raise ImageValidationError(f"Unsupported file extension '{p.suffix or '(none)'}'.")
                if not p.is_file():
                    raise ImageValidationError("File not found.")
                if max_bytes is not None and p.stat().st_size > max_bytes:
                    raise ImageValidationError("File is too large.")
                data = p.read_bytes()
            elif isinstance(source, (bytes, bytearray)):
                data = bytes(source)
            elif hasattr(source, "read"):
                data = source.read()
            else:
                raise ImageValidationError("Unsupported input type.")
            if not data:
                raise ImageValidationError("File is empty.")
            if max_bytes is not None and len(data) > max_bytes:
                raise ImageValidationError("File is too large.")
            im = Image.open(io.BytesIO(data))
            if im.format not in _ALLOWED_FORMATS:
                raise ImageValidationError("Unsupported image format (PNG or JPEG required).")
        w, h = im.size                       # known before decoding: reject bombs/tiny images first
        if min(w, h) < min_side:
            raise ImageValidationError(f"Image is too small (minimum {min_side} px per side).")
        if w * h > max_pixels:
            raise ImageValidationError("Image dimensions are too large.")
        if data is not None:
            im.load()                        # full decode: detects truncated/corrupted files
        return to_rgb(im)
    except ImageValidationError:
        raise
    except Exception as exc:                 # Pillow raises many types for broken files
        raise ImageValidationError(f"Cannot read image ({type(exc).__name__}).") from exc


@dataclass(frozen=True)
class Preprocessor:
    image_size: int = 224
    resize_mode: str = "pad_square"
    scaling: str = "unit"
    allowed_extensions: tuple | None = None   # set for web uploads only
    max_bytes: int | None = None               # set for web uploads only
    min_side: int = 16
    max_pixels: int = 25_000_000

    def __post_init__(self):
        if self.resize_mode not in RESIZE_MODES:
            raise ValueError(f"resize_mode must be one of {RESIZE_MODES}")
        if self.scaling not in SCALINGS:
            raise ValueError(f"scaling must be one of {SCALINGS}")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")

    def resize_image(self, source) -> np.ndarray:
        """Load -> RGB -> pad/stretch -> resize. Returns float32 (H, W, 3) with values 0..255."""
        im = load_image(source, self.allowed_extensions, self.max_bytes, self.min_side, self.max_pixels)
        if self.resize_mode == "pad_square":
            im = pad_to_square(im)
        im = im.resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32)

    def scale(self, x: np.ndarray) -> np.ndarray:
        if self.scaling == "unit":
            return x / 255.0
        if self.scaling == "minus1_1":
            return x / 127.5 - 1.0
        return x

    def __call__(self, source) -> np.ndarray:
        return self.scale(self.resize_image(source))

    def to_metadata(self) -> dict:
        """Stored in models/metadata.json so inference reproduces training preprocessing exactly."""
        return {"image_size": self.image_size, "channels": 3, "resize_mode": self.resize_mode,
                "padding": "black" if self.resize_mode == "pad_square" else None, "interpolation": "bilinear",
                "scaling": self.scaling, "color_mode": "RGB"}

    @classmethod
    def from_metadata(cls, meta: dict, **limits) -> "Preprocessor":
        return cls(image_size=meta["image_size"], resize_mode=meta["resize_mode"], scaling=meta["scaling"], **limits)

    @classmethod
    def from_config(cls, cfg: dict, model_name: str | None = None, for_upload: bool = False) -> "Preprocessor":
        d = cfg["data"]
        model_name = model_name or cfg["training"]["model_name"]
        kw = {}
        if for_upload:
            kw["allowed_extensions"] = tuple(cfg["web"]["allowed_extensions"])
            kw["max_bytes"] = int(cfg["web"]["upload_limit_mb"] * 1024 * 1024)
        return cls(image_size=d["image_size"], resize_mode=d["resize_mode"],
                   scaling=d.get("scaling_by_model", {}).get(model_name, "unit"),
                   min_side=d.get("min_side_px", 16), max_pixels=d.get("max_pixels", 25_000_000), **kw)
