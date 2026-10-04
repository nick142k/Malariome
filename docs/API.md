# MalariaAI — REST API

AI-assisted screening only. This system is not a medical diagnostic device. Predictions must be verified by a qualified healthcare professional.

Base URL (local): `http://localhost:5000`. The API is stateless (no cookies, no CSRF token). Uploaded images are never kept by the API.

## POST /api/predict
`multipart/form-data` with one file field named **`image`** (PNG or JPG, size limit from `config.yaml` → `web.upload_limit_mb`).

```bash
curl -F "image=@cell.png" http://localhost:5000/api/predict
```

Success (200), cell flagged parasitized:
```json
{"success": true, "prediction": "PARASITIZED", "malaria_probability": 0.95, "threshold": 0.42,
 "model": "TransferModel", "model_version": "0.1-EXP-002"}
```
Uninfected: same fields with `"prediction": "UNINFECTED"`. When a species model is plugged in, `species` and
`species_probability` are added **only** when the prediction is PARASITIZED; they are absent otherwise.

`prediction` is `PARASITIZED` when `malaria_probability >= threshold`. The threshold comes from the model's metadata
(selected on validation data only), not from the code.

| Status | When | Body |
|---|---|---|
| 200 | prediction made | `success: true` + fields above |
| 400 | no `image` field, not a decodable PNG/JPG, truncated/corrupted, too small/large in pixels | `{"success": false, "error": "..."}` |
| 413 | file larger than the upload limit | same error shape |
| 415 | extension not in `web.allowed_extensions` | same error shape |
| 503 | model not loaded | same error shape |
| 500 | unexpected failure (details are logged, never returned) | generic message |

## GET /api/health
```json
{"status": "healthy", "model_loaded": true}
```
`"status": "degraded"` with `"model_loaded": false` means the app is up but no model could be loaded.

## Web pages
`/` upload · `/predict` (POST from the form) · `/history` (clear with the button; POST + form token) · `/model` · `/about`

## Predictor contract (how the app talks to the model)
```python
predictor.predict(image_bytes) -> {"prediction": "PARASITIZED"|"UNINFECTED", "malaria_probability": float,
                                   "threshold": float, "model": str, "model_version": str,
                                   # optional, only for parasitized cells:
                                   "species": str, "species_probability": float}
predictor.explain(image_bytes) -> {"original": b64png, "heatmap": b64png, "overlay": b64png}   # optional
```
`src/inference/predictor.py` must expose `build_predictor(cfg)` returning such an object. Invalid images must raise
`src.data.preprocessing.ImageValidationError`. The app never imports TensorFlow itself.
