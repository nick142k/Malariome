"""MalariaAI web application (Flask): upload page, result page, history, model card, about, REST API.

Run:  python app.py        (http://localhost:5000, served by waitress)

The app talks to the model ONLY through a predictor object (see docs/API.md, "Predictor contract"):
    predictor.predict(image_bytes) -> dict(prediction, malaria_probability, threshold, model, model_version[, species, species_probability])
    predictor.explain(image_bytes) -> dict(original, heatmap, overlay)   # optional, base64 PNG strings
If no predictor can be loaded the app still starts: /api/health reports "degraded" and predictions return 503.
"""
import json
import os
import re
import secrets
import time
import uuid
from pathlib import Path

from flask import Flask, abort, flash, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from database import database as db
from src.config import PROJECT_ROOT, load_config
from src.data.preprocessing import ImageValidationError, load_image

DISCLAIMER = ("AI-assisted screening only. This system is not a medical diagnostic device. "
              "Predictions must be verified by a qualified healthcare professional.")
GRADCAM_NOTE = ("Grad-CAM highlights image regions that contributed to the model output. "
                "It is an interpretability aid and does not establish biological causality.")
SAFE_NAME = re.compile(r"^[0-9a-f]{32}\.(png|jpg|jpeg)$")
UPLOAD_MAX_AGE_S = 3600


class UploadError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


def load_predictor(cfg):
    try:
        from src.inference.predictor import build_predictor
        return build_predictor(cfg)
    except Exception as exc:                       # missing model, missing TensorFlow, missing module ...
        print(f"[warn] Model not loaded ({type(exc).__name__}: {exc}). The app will run in degraded mode.")
        return None


def cleanup_uploads(upload_dir, max_age_s=UPLOAD_MAX_AGE_S):
    now = time.time()
    for f in Path(upload_dir).glob("*"):
        try:
            if f.is_file() and now - f.stat().st_mtime > max_age_s:
                f.unlink()
        except OSError:
            pass


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def create_app(overrides=None, predictor="auto"):
    cfg = load_config()
    overrides = overrides or {}
    paths = {k: Path(v) for k, v in cfg["abs_paths"].items()}
    paths.update({k: Path(v) for k, v in overrides.get("paths", {}).items()})
    web = {**cfg["web"], **overrides.get("web", {})}

    app = Flask(__name__, template_folder=str(PROJECT_ROOT / "web" / "templates"),
                static_folder=str(PROJECT_ROOT / "web" / "static"))
    max_bytes = int(web["upload_limit_mb"] * 1024 * 1024)
    app.config.update(MAX_CONTENT_LENGTH=max_bytes, SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_HTTPONLY=True)
    app.secret_key = os.environ.get("MALARIA_SECRET_KEY") or secrets.token_hex(32)
    allowed = {e.lower().lstrip(".") for e in web["allowed_extensions"]}
    upload_dir, db_path = paths["upload_dir"], paths["database_path"]
    upload_dir.mkdir(parents=True, exist_ok=True)
    db.init_db(db_path)
    cleanup_uploads(upload_dir)
    pred = load_predictor(cfg) if predictor == "auto" else predictor
    app.extensions["predictor"] = pred

    # ---------------- helpers ----------------
    def csrf_token():
        if "csrf" not in session:
            session["csrf"] = secrets.token_hex(16)
        return session["csrf"]

    def check_csrf():
        sent, good = request.form.get("csrf_token", ""), session.get("csrf", "")
        if not good or not secrets.compare_digest(sent, good):
            abort(400, "Invalid or missing form token. Reload the page and try again.")

    @app.context_processor
    def inject():
        return {"csrf_token": csrf_token, "disclaimer": DISCLAIMER, "gradcam_note": GRADCAM_NOTE,
                "model_ready": pred is not None, "allowed": sorted(allowed), "max_mb": web["upload_limit_mb"]}

    def process_upload(fs):
        """Validate, store under a random name, and return (display_name, stored_name, bytes)."""
        if fs is None or not fs.filename:
            raise UploadError("No image was uploaded.", 400)
        original = secure_filename(fs.filename)
        ext = Path(original).suffix.lower().lstrip(".")
        if ext not in allowed:
            raise UploadError("Unsupported file type. Allowed: " + ", ".join(sorted(allowed)) + ".", 415)
        data = fs.read()
        try:
            load_image(data, max_bytes=max_bytes)                      # real decode: rejects fake or corrupted images
        except ImageValidationError as exc:
            raise UploadError(str(exc), 400)
        stored = f"{uuid.uuid4().hex}.{ext}"
        (upload_dir / stored).write_bytes(data)
        return original or "upload", stored, data

    def run_model(data):
        try:
            result = pred.predict(data)
        except ImageValidationError as exc:
            raise UploadError(str(exc), 400)
        except Exception as exc:
            app.logger.exception("prediction failed")
            raise UploadError("The image could not be analysed.", 500) from exc
        gradcam = None
        if hasattr(pred, "explain"):
            try:
                gradcam = pred.explain(data)
            except Exception:
                app.logger.exception("Grad-CAM failed")
        return result, gradcam

    def log_prediction(name, result, source):
        db.add_prediction(db_path, filename=name, predicted_class=result["prediction"], probability=result["malaria_probability"],
                          threshold=result["threshold"], model_version=result.get("model_version"),
                          species=result.get("species"), species_probability=result.get("species_probability"), source=source)

    # ---------------- web pages ----------------
    @app.get("/")
    def index():
        return render_template("index.html")

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        if request.method == "GET":
            return redirect(url_for("index"))
        check_csrf()
        if pred is None:
            flash("The model is not loaded, so predictions are unavailable.", "error")
            return redirect(url_for("index"))
        cleanup_uploads(upload_dir)
        try:
            name, stored, data = process_upload(request.files.get("image"))
            result, gradcam = run_model(data)
        except UploadError as exc:
            flash(exc.message, "error")
            return redirect(url_for("index"))
        log_prediction(name, result, "web")
        return render_template("result.html", r=result, image=url_for("uploaded_file", name=stored), filename=name, gradcam=gradcam)

    @app.get("/uploads/<name>")
    def uploaded_file(name):
        if not SAFE_NAME.match(name):
            abort(404)
        return send_from_directory(upload_dir, name, max_age=0)

    @app.get("/history")
    def history():
        return render_template("history.html", rows=db.list_predictions(db_path), total=db.count(db_path))

    @app.post("/history/clear")
    def clear_history():
        check_csrf()
        n = db.clear_history(db_path)
        flash(f"Cleared {n} saved prediction{'s' if n != 1 else ''}.", "info")
        return redirect(url_for("history"))

    @app.get("/model")
    def model_page():
        name = getattr(pred, "model_name", None) if pred is not None else None
        meta = read_json(paths["models_dir"] / "metadata.json") or read_json(paths["models_dir"] / "transfer_model.meta.json")
        reports = {split: read_json(paths["reports_dir"] / "metrics" / f"{split}_metrics.json") for split in ("val", "test")}
        return render_template("model.html", meta=meta, reports=reports, model_name=name)

    @app.get("/about")
    def about():
        return render_template("about.html")

    # ---------------- REST API ----------------
    def public(result):
        keys = ["prediction", "malaria_probability", "threshold", "model", "model_version", "species", "species_probability"]
        return {k: result[k] for k in keys if k in result and result[k] is not None}

    @app.post("/api/predict")
    def api_predict():
        if pred is None:
            return jsonify(success=False, error="Model not loaded."), 503
        stored = None
        try:
            name, stored, data = process_upload(request.files.get("image"))
            result, _ = run_model(data)
        except UploadError as exc:
            return jsonify(success=False, error=exc.message), exc.status
        finally:
            if stored:                                              # the API never keeps uploaded images
                (upload_dir / stored).unlink(missing_ok=True)
        log_prediction(name, result, "api")
        return jsonify(success=True, **public(result))

    @app.get("/api/health")
    def api_health():
        return jsonify(status="healthy" if pred is not None else "degraded", model_loaded=pred is not None)

    # ---------------- errors ----------------
    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        msg = f"File is too large (limit {web['upload_limit_mb']} MB)."
        if request.path.startswith("/api/"):
            return jsonify(success=False, error=msg), 413
        flash(msg, "error")
        return redirect(url_for("index"))

    @app.errorhandler(400)
    def bad_request(e):
        if request.path.startswith("/api/"):
            return jsonify(success=False, error=getattr(e, "description", "Bad request.")), 400
        return render_template("error.html", code=400, message=getattr(e, "description", "Bad request.")), 400

    @app.errorhandler(404)
    def not_found(_):
        if request.path.startswith("/api/"):
            return jsonify(success=False, error="Not found."), 404
        return render_template("error.html", code=404, message="Page not found."), 404

    return app


if __name__ == "__main__":
    application = create_app()
    cfg = load_config()
    host, port = cfg["web"]["host"], int(cfg["web"]["port"])
    print(f"MalariaAI running at http://localhost:{port}")
    try:
        from waitress import serve
        serve(application, host=host, port=port)
    except ImportError:
        application.run(host=host, port=port)
