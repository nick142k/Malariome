"""Web app + REST API tests with a fake predictor (no TensorFlow or trained model needed)."""
import base64
import io
import re

from PIL import Image

from app import create_app
from database import database as db


def png(w=64, h=64, color=(180, 90, 120)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


PIXEL = base64.b64encode(png(8, 8)).decode()


class FakePredictor:
    model_name = "FakeModel"

    def __init__(self, prob=0.9, thr=0.42, species=None, boom=False):
        self.prob, self.thr, self.species, self.boom = prob, thr, species, boom

    def predict(self, data):
        if self.boom:
            raise RuntimeError("secret internal detail")
        pos = self.prob >= self.thr
        out = {"prediction": "PARASITIZED" if pos else "UNINFECTED", "malaria_probability": self.prob, "threshold": self.thr,
               "model": "FakeModel", "model_version": "0.0-test"}
        if pos and self.species:
            out.update(species=self.species, species_probability=0.77)
        return out


class FakeWithGradcam(FakePredictor):
    def explain(self, data):
        return {"original": PIXEL, "heatmap": PIXEL, "overlay": PIXEL}


def make(tmp_path, predictor, web=None):
    app = create_app({"paths": {"upload_dir": tmp_path / "uploads", "database_path": tmp_path / "t.db"}, "web": web or {}}, predictor)
    return app, app.test_client(), tmp_path / "uploads", tmp_path / "t.db"


def post_api(client, data, name="cell.png", field="image"):
    return client.post("/api/predict", data={field: (io.BytesIO(data), name)}, content_type="multipart/form-data")


def post_web(client, data, name="cell.png", token="tok", follow=False):
    with client.session_transaction() as s:
        s["csrf"] = "tok"
    return client.post("/predict", data={"image": (io.BytesIO(data), name), "csrf_token": token},
                       content_type="multipart/form-data", follow_redirects=follow)


# ---------------- API ----------------
def test_health_healthy_and_degraded(tmp_path):
    _, c, *_ = make(tmp_path / "a", FakePredictor())
    assert c.get("/api/health").get_json() == {"status": "healthy", "model_loaded": True}
    _, c2, *_ = make(tmp_path / "b", None)
    assert c2.get("/api/health").get_json() == {"status": "degraded", "model_loaded": False}


def test_api_predict_success_logs_history_and_keeps_no_upload(tmp_path):
    _, c, up, dbp = make(tmp_path, FakePredictor(prob=0.95))
    r = post_api(c, png())
    j = r.get_json()
    assert r.status_code == 200 and j["success"] is True and j["prediction"] == "PARASITIZED"
    assert j["malaria_probability"] == 0.95 and j["threshold"] == 0.42 and j["model"] == "FakeModel"
    assert "species" not in j and not list(up.glob("*"))                     # nothing kept on disk
    row = db.list_predictions(dbp)[0]
    assert row["source"] == "api" and row["predicted_class"] == "PARASITIZED" and row["species"] is None


def test_api_uninfected_has_no_species_even_if_model_offers_one(tmp_path):
    _, c, *_ = make(tmp_path, FakePredictor(prob=0.02, species="P. falciparum"))
    j = post_api(c, png()).get_json()
    assert j["prediction"] == "UNINFECTED" and "species" not in j


def test_api_includes_species_when_parasitized(tmp_path):
    _, c, *_ = make(tmp_path, FakePredictor(prob=0.9, species="P. vivax"))
    j = post_api(c, png()).get_json()
    assert j["species"] == "P. vivax" and j["species_probability"] == 0.77


def test_api_rejects_garbage_disguised_as_png(tmp_path):
    _, c, up, _ = make(tmp_path, FakePredictor())
    r = post_api(c, b"not an image at all")
    assert r.status_code == 400 and r.get_json()["success"] is False and not list(up.glob("*"))


def test_api_rejects_truncated_png(tmp_path):
    _, c, *_ = make(tmp_path, FakePredictor())
    data = png(200, 200)
    assert post_api(c, data[: len(data) // 2]).status_code == 400


def test_api_rejects_unsupported_extension(tmp_path):
    _, c, *_ = make(tmp_path, FakePredictor())
    assert post_api(c, png(), name="cell.gif").status_code == 415
    assert post_api(c, png(), name="cell.exe").status_code == 415


def test_api_missing_image_field(tmp_path):
    _, c, *_ = make(tmp_path, FakePredictor())
    assert post_api(c, png(), field="file").status_code == 400


def test_api_oversized_upload(tmp_path):
    _, c, *_ = make(tmp_path, FakePredictor(), web={"upload_limit_mb": 0.001})
    r = post_api(c, png(300, 300))
    assert r.status_code == 413 and r.get_json()["success"] is False


def test_api_model_not_loaded_returns_503(tmp_path):
    _, c, *_ = make(tmp_path, None)
    assert post_api(c, png()).status_code == 503


def test_api_internal_errors_do_not_leak_details(tmp_path):
    _, c, *_ = make(tmp_path, FakePredictor(boom=True))
    r = post_api(c, png())
    assert r.status_code == 500 and "secret" not in r.get_data(as_text=True)


def test_unknown_api_route_is_json_404(tmp_path):
    _, c, *_ = make(tmp_path, FakePredictor())
    r = c.get("/api/nope")
    assert r.status_code == 404 and r.get_json()["success"] is False


# ---------------- web pages ----------------
def test_index_has_branding_and_disclaimer(tmp_path):
    _, c, *_ = make(tmp_path, FakePredictor())
    html = c.get("/").get_data(as_text=True)
    assert "AI-Assisted Malaria Screening" in html and "Analyze Image" in html and "Choose Image" in html
    assert "not a medical diagnostic device" in html


def test_web_predict_shows_result_saves_history_and_randomises_filename(tmp_path):
    _, c, up, dbp = make(tmp_path, FakePredictor(prob=0.97))
    r = post_web(c, png(), name="../../evil.png")
    html = r.get_data(as_text=True)
    assert r.status_code == 200 and "PARASITIZED" in html and "97.0%" in html and "0.42" in html and "FakeModel" in html
    assert "Grad-CAM" not in html                                              # fake without explain()
    saved = [p.name for p in up.glob("*")]
    assert len(saved) == 1 and re.fullmatch(r"[0-9a-f]{32}\.png", saved[0])
    assert db.list_predictions(dbp)[0]["filename"] == "evil.png"
    assert c.get(f"/uploads/{saved[0]}").status_code == 200


def test_web_predict_shows_gradcam_and_species(tmp_path):
    class Both(FakeWithGradcam):
        pass
    _, c, *_ = make(tmp_path, Both(prob=0.9, species="P. falciparum"))
    html = post_web(c, png()).get_data(as_text=True)
    assert "Grad-CAM explanation" in html and "does not establish biological causality" in html
    assert "P. falciparum" in html and "research" in html


def test_web_uninfected_page_has_no_species_block(tmp_path):
    _, c, *_ = make(tmp_path, FakePredictor(prob=0.03, species="P. vivax"))
    html = post_web(c, png()).get_data(as_text=True)
    assert "UNINFECTED" in html and "Species estimate" not in html


def test_web_predict_requires_csrf_token(tmp_path):
    _, c, _, dbp = make(tmp_path, FakePredictor())
    assert post_web(c, png(), token="wrong").status_code == 400
    assert db.count(dbp) == 0


def test_web_invalid_upload_redirects_with_message(tmp_path):
    _, c, up, dbp = make(tmp_path, FakePredictor())
    html = post_web(c, b"garbage", follow=True).get_data(as_text=True)
    assert "Cannot read image" in html and db.count(dbp) == 0 and not list(up.glob("*"))


def test_web_unsupported_type_and_oversize_messages(tmp_path):
    _, c, *_ = make(tmp_path, FakePredictor())
    assert "Unsupported file type" in post_web(c, png(), name="x.gif", follow=True).get_data(as_text=True)
    _, c2, *_ = make(tmp_path / "small", FakePredictor(), web={"upload_limit_mb": 0.001})
    assert "too large" in post_web(c2, png(300, 300), follow=True).get_data(as_text=True)


def test_web_model_not_loaded_shows_banner_and_blocks_predictions(tmp_path):
    _, c, *_ = make(tmp_path, None)
    assert "model is not loaded" in c.get("/").get_data(as_text=True)
    assert "model is not loaded" in post_web(c, png(), follow=True).get_data(as_text=True)


def test_history_lists_and_clear_needs_csrf(tmp_path):
    _, c, _, dbp = make(tmp_path, FakePredictor())
    post_web(c, png(), name="a.png"); post_web(c, png(), name="b.png")
    html = c.get("/history").get_data(as_text=True)
    assert "a.png" in html and "b.png" in html and db.count(dbp) == 2
    assert c.post("/history/clear", data={"csrf_token": "wrong"}).status_code == 400 and db.count(dbp) == 2
    with c.session_transaction() as s:
        s["csrf"] = "tok"
    r = c.post("/history/clear", data={"csrf_token": "tok"}, follow_redirects=True)
    assert "Cleared 2" in r.get_data(as_text=True) and db.count(dbp) == 0


def test_history_escapes_hostile_filenames(tmp_path):
    _, c, *_ = make(tmp_path, FakePredictor())
    post_web(c, png(), name='"><img src=x onerror=alert(1)>.png')
    assert "<img src=x" not in c.get("/history").get_data(as_text=True)


def test_uploads_route_blocks_traversal_and_odd_names(tmp_path):
    _, c, up, _ = make(tmp_path, FakePredictor())
    (tmp_path / "secret.txt").write_text("x")
    for bad in ["..%2fsecret.txt", "secret.txt", "../secret.txt", "A" * 32 + ".png", "0" * 32 + ".exe", "0" * 32 + ".png"]:
        assert c.get(f"/uploads/{bad}").status_code == 404


def test_get_predict_redirects_and_pages_render(tmp_path):
    _, c, *_ = make(tmp_path, FakePredictor())
    assert c.get("/predict").status_code == 302
    for page in ("/history", "/model", "/about"):
        assert c.get(page).status_code == 200
    r = c.get("/does-not-exist")
    assert r.status_code == 404 and "Page not found" in r.get_data(as_text=True)


def test_model_page_reads_metadata_and_reports_honestly(tmp_path):
    app, c, *_ = make(tmp_path, FakePredictor())
    html = c.get("/model").get_data(as_text=True)
    assert "NOT YET EVALUATED" in html or "Validation performance" in html         # never invents numbers
