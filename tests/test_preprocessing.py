"""Tests for src/data/preprocessing.py (no TensorFlow needed)."""
import io
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.preprocessing import (ImageValidationError, Preprocessor, load_image, pad_to_square, to_rgb)


def _png_bytes(w=60, h=40, color=(255, 0, 0), mode="RGB", fmt="PNG"):
    buf = io.BytesIO()
    Image.new(mode, (w, h), color).save(buf, format=fmt)
    return buf.getvalue()


def _save(tmp_path, name="cell.png", **kw):
    p = tmp_path / name
    p.write_bytes(_png_bytes(**kw))
    return p


# ---------------- geometry ----------------
def test_pad_to_square_is_centred_and_black():
    out = np.asarray(pad_to_square(Image.new("RGB", (60, 40), (255, 0, 0))))   # w=60, h=40
    assert out.shape == (60, 60, 3)
    assert out[:10].sum() == 0 and out[50:].sum() == 0                          # black bars top/bottom
    assert (out[10:50] == [255, 0, 0]).all()                                    # content untouched


def test_pad_to_square_leaves_square_images_alone():
    im = Image.new("RGB", (50, 50), (1, 2, 3))
    assert pad_to_square(im) is im


def test_stretch_mode_has_no_padding():
    x = Preprocessor(image_size=60, resize_mode="stretch").resize_image(Image.new("RGB", (60, 40), (255, 0, 0)))
    assert (x == [255, 0, 0]).all()


@pytest.mark.parametrize("size", [(46, 40), (130, 130), (394, 385), (300, 130), (130, 300), (17, 17)])
def test_output_shape_dtype_and_range_for_real_dataset_sizes(size):
    x = Preprocessor(image_size=224)(Image.new("RGB", size, (200, 100, 150)))
    assert x.shape == (224, 224, 3) and x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0 and np.isfinite(x).all()


# ---------------- scaling ----------------
def test_scaling_ranges():
    white, black = Image.new("RGB", (32, 32), (255, 255, 255)), Image.new("RGB", (32, 32), (0, 0, 0))
    for scaling, lo, hi in [("unit", 0.0, 1.0), ("minus1_1", -1.0, 1.0), ("none", 0.0, 255.0)]:
        p = Preprocessor(image_size=32, scaling=scaling)
        assert np.allclose(p(black), lo) and np.allclose(p(white), hi)


def test_invalid_settings_rejected():
    with pytest.raises(ValueError):
        Preprocessor(scaling="bogus")
    with pytest.raises(ValueError):
        Preprocessor(resize_mode="bogus")
    with pytest.raises(ValueError):
        Preprocessor(image_size=0)


# ---------------- train/inference consistency ----------------
def test_same_result_from_path_bytes_filelike_and_pil(tmp_path):
    p = _save(tmp_path, w=77, h=91, color=(10, 200, 30))
    pre = Preprocessor(image_size=64, scaling="minus1_1")
    ref = pre(p)
    assert np.array_equal(ref, pre(p.read_bytes()))
    assert np.array_equal(ref, pre(io.BytesIO(p.read_bytes())))
    assert np.array_equal(ref, pre(Image.open(p)))
    assert np.array_equal(ref, pre(p))                      # deterministic


def test_metadata_round_trip():
    pre = Preprocessor(image_size=128, resize_mode="stretch", scaling="none")
    again = Preprocessor.from_metadata(pre.to_metadata())
    assert (again.image_size, again.resize_mode, again.scaling) == (128, "stretch", "none")


def test_from_config_picks_scaling_per_model_and_upload_limits():
    cfg = {"data": {"image_size": 224, "resize_mode": "pad_square",
                    "scaling_by_model": {"custom_cnn": "unit", "mobilenetv2": "minus1_1"}},
           "training": {"model_name": "custom_cnn"},
           "web": {"allowed_extensions": ["png", "jpg"], "upload_limit_mb": 5}}
    assert Preprocessor.from_config(cfg).scaling == "unit"
    assert Preprocessor.from_config(cfg, "mobilenetv2").scaling == "minus1_1"
    up = Preprocessor.from_config(cfg, for_upload=True)
    assert up.max_bytes == 5 * 1024 * 1024 and up.allowed_extensions == ("png", "jpg")
    assert Preprocessor.from_config(cfg).max_bytes is None


# ---------------- colour modes ----------------
def test_grayscale_and_palette_become_rgb():
    g = np.asarray(to_rgb(Image.new("L", (20, 20), 99)))
    assert g.shape == (20, 20, 3) and (g == 99).all()
    assert to_rgb(Image.new("P", (20, 20))).mode == "RGB"


def test_transparent_pixels_composited_on_black():
    clear = np.asarray(to_rgb(Image.new("RGBA", (20, 20), (255, 0, 0, 0))))
    solid = np.asarray(to_rgb(Image.new("RGBA", (20, 20), (255, 0, 0, 255))))
    assert clear.sum() == 0 and (solid == [255, 0, 0]).all()


def test_jpeg_is_accepted(tmp_path):
    p = tmp_path / "cell.jpg"
    p.write_bytes(_png_bytes(fmt="JPEG"))
    assert Preprocessor(image_size=32)(p).shape == (32, 32, 3)


# ---------------- bad inputs: must raise ImageValidationError, never crash otherwise ----------------
def test_not_an_image():
    with pytest.raises(ImageValidationError):
        load_image(b"this is definitely not an image")


def test_empty_input():
    with pytest.raises(ImageValidationError):
        load_image(b"")


def test_truncated_png_is_rejected():
    data = _png_bytes(200, 200)
    with pytest.raises(ImageValidationError):
        load_image(data[: len(data) // 2])


def test_unsupported_extension_checked_before_reading(tmp_path):
    p = tmp_path / "cell.gif"
    p.write_bytes(_png_bytes())
    with pytest.raises(ImageValidationError):
        load_image(p, allowed_extensions=("png", "jpg", "jpeg"))


def test_extension_check_is_case_insensitive(tmp_path):
    p = _save(tmp_path, name="CELL.PNG")
    assert load_image(p, allowed_extensions=("png",)).size == (60, 40)


def test_unsupported_image_format_rejected():
    with pytest.raises(ImageValidationError):
        load_image(_png_bytes(fmt="BMP"))


def test_missing_file(tmp_path):
    with pytest.raises(ImageValidationError):
        load_image(tmp_path / "nope.png")


def test_oversized_upload(tmp_path):
    p = _save(tmp_path, w=200, h=200)
    with pytest.raises(ImageValidationError):
        load_image(p, max_bytes=100)
    with pytest.raises(ImageValidationError):
        load_image(p.read_bytes(), max_bytes=100)


def test_too_small_and_too_many_pixels():
    with pytest.raises(ImageValidationError):
        load_image(_png_bytes(8, 8), min_side=16)
    with pytest.raises(ImageValidationError):
        load_image(_png_bytes(40, 40), max_pixels=100)


def test_unsupported_input_type():
    with pytest.raises(ImageValidationError):
        load_image(12345)


# ---------------- optional smoke test on the real dataset ----------------
_REAL = os.environ.get("MALARIA_DATA_DIR")


@pytest.mark.skipif(not _REAL, reason="set MALARIA_DATA_DIR to run on real images")
def test_real_images_smoke():
    files = []
    for cls in ("Parasitized", "Uninfected"):
        files += sorted((Path(_REAL) / cls).glob("*.png"))[:25]
    assert files, "no real images found"
    pre = Preprocessor(image_size=224, scaling="unit")
    for f in files:
        x = pre(f)
        assert x.shape == (224, 224, 3) and np.isfinite(x).all() and 0 <= x.min() and x.max() <= 1
