"""SpeciesClassifier logic with a fake model (no TensorFlow needed)."""
import io

import numpy as np
from PIL import Image

from src.data.preprocessing import ImageValidationError
from src.inference.species import SpeciesClassifier


def png(w=90, h=60, color=(200, 80, 120)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


class FakeSpeciesModel:
    def __init__(self, probs):
        self.probs, self.seen = np.array([probs], dtype="float32"), []

    def __call__(self, x, training=False):
        self.seen.append(np.asarray(x))
        return self.probs


def test_picks_argmax_and_formats_name():
    c = SpeciesClassifier(FakeSpeciesModel([0.1, 0.05, 0.05, 0.8]), image_size=32)
    r = c.predict(png())
    assert r["species"] == "P. vivax" and abs(r["species_probability"] - 0.8) < 1e-6
    assert set(r) == {"species", "species_probability"}


def test_model_gets_raw_pixels_stretched_not_scaled_or_padded():
    m = FakeSpeciesModel([1, 0, 0, 0])
    SpeciesClassifier(m, image_size=32).predict(png(120, 60, (255, 255, 255)))
    x = m.seen[0]
    assert x.shape == (1, 32, 32, 3) and x.max() > 200 and x.min() > 200      # white stays white: no padding bars, no [-1,1] scaling


def test_bad_images_raise_validation_error():
    c = SpeciesClassifier(FakeSpeciesModel([1, 0, 0, 0]), image_size=32)
    for bad in (b"garbage", b""):
        try:
            c.predict(bad)
            assert False, "should have raised"
        except ImageValidationError:
            pass
    assert c.model.seen == []