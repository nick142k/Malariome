# MalariaAI — Architecture

Status legend: **PLANNED** = designed, not implemented · **IMPLEMENTED — NOT YET VERIFIED** · **VERIFIED**.
Only `config.yaml` loading (`src/config.py`) exists so far; it was run once and returned the expected values (VERIFIED for that check only). Everything else below is PLANNED.

## 1. Pipeline

```
Kaggle dataset (Parasitized / Uninfected)
  → verify_dataset.py                      (counts, corruption, duplicates, sizes)  [VERIFIED, Phase 3]
  → split_dataset (group-aware, seed 42)   → data/splits/{train,val,test}.csv
  → preprocessing.py  (one function, used by training AND inference)
  → Custom CNN (scratch)   |   MobileNetV2 (frozen head → fine-tune)   [+ EXP-000 baseline replication]
  → train.py  → models/*.keras + history + metadata.json
  → evaluate.py (SAME held-out test set; thresholds chosen on validation only)
  → model selection → Grad-CAM → predictor.py
  → Flask: web UI + POST /api/predict + GET /api/health → SQLite history
  → pytest → Docker (waitress)
```

## 2. Decisions and why

| Decision | Choice | Reason |
|---|---|---|
| Input size | 224×224×3, both models | MobileNetV2's native size; identical pipeline for both models. Most source images are ~130 px, so this upsamples — noted as a limitation |
| Resize | pad to square (black), then resize | Crops are not square (40–385 px); black matches the cell background and avoids stretching. `stretch` kept as a config option |
| Output head | 1 sigmoid unit, binary cross-entropy | Binary task; gives one probability and a tunable threshold (original used 2-unit softmax) |
| Positive class | Parasitized (P = probability of Parasitized) | Screening cares about missed infections |
| Split | 70/15/15, seed 42, **grouped** and stratified | Verified: 1,408 smear groups, 936 contain both classes → random split leaks. Group = patient ID if parseable from filenames, else smear ID. Split script must assert zero group overlap |
| Augmentation | rotation, flips (both axes), small zoom/shift, brightness/contrast; train only | Cells have no canonical orientation; no colour/hue shifts because stain colour may carry signal |
| Scaling | custom CNN: ÷255; MobileNetV2: its own `preprocess_input` ([-1,1]) | Chosen per model, stored in `metadata.json`, and read back by the predictor so inference always matches training. (The reference repo's failure came from a train/eval scaling mismatch) |
| Threshold | selected on validation data, never on test | Rule fixed in Phase 12 before test evaluation; `config.yaml` value 0.5 is a placeholder |
| Training | Colab/Kaggle GPU; local machine does inference, Flask, tests | Matches the project spec; CPU fallback in `train.py` |
| Serving | Flask + waitress | waitress runs on Windows and in Docker |
| Storage | SQLite, no personal data | filename (original, display only), class, probability, threshold, model version, timestamp |
| Reference code | Not copied | Reference repo has no license; credited as reference only |

## 3. Models

**Custom CNN (from scratch)** — Conv→BN→ReLU→MaxPool→Dropout ×2, Conv→BN→ReLU, GlobalAveragePooling, Dense, Dropout, Sigmoid. Filter counts to be tuned experimentally; parameter count, layer shapes: **NOT YET MEASURED**.

**Transfer model** — MobileNetV2 with ImageNet weights: (1) freeze base, (2) train new head, (3) unfreeze upper layers, (4) fine-tune at `fine_tune_learning_rate`. EfficientNetB0 optional. Which model is deployed: **NOT YET EVALUATED** — chosen on sensitivity, specificity, F1, ROC-AUC, PR-AUC, speed, size, explainability, not accuracy alone.

**EXP-000** — replication of the reference baseline (64×64, softmax) with its validation-scaling bug fixed, for an honest before/after comparison.

## 4. Module map

| Path | Responsibility | Status |
|---|---|---|
| `config.yaml`, `src/config.py` | all settings, path resolution (env override `MALARIA_DATA_DIR`) | VERIFIED (load test) |
| `src/data/splitter.py` | manifest + grouped stratified split, writes data/splits/*.csv | IMPLEMENTED — NOT YET VERIFIED (synthetic data only) |
| `src/data/{loader,preprocessing,augmentation}.py` | loading, shared preprocessing, train-only augmentation | PLANNED |
| `src/models/{custom_cnn,transfer_learning,model_loader}.py` | model builders and loading | PLANNED |
| `src/training/{trainer,callbacks}.py`, `train.py` | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, history | PLANNED |
| `src/evaluation/*`, `evaluate.py` | metrics, confusion matrix, ROC/PR, threshold table, misclassification analysis | PLANNED |
| `src/explainability/gradcam.py` | Grad-CAM on the deployed model's last conv layer | PLANNED |
| `src/inference/predictor.py`, `predict.py` | `predict_image()`, robust to bad uploads | PLANNED |
| `app.py`, `web/`, `database/database.py` | Flask UI, REST API, SQLite | PLANNED |
| `tests/` | pytest suite | PLANNED |
| `Dockerfile`, `docker-compose.yml` | container | PLANNED |

## 5. Security plan (web)
`secure_filename` + random server-side names, extension allow-list, size limit from config, uploads stored outside static files, every upload re-opened with Pillow before use, no user input in paths or commands, errors returned as JSON/messages instead of crashes.

## 6. Disclaimer shown in UI and docs
"AI-assisted screening only. This system is not a medical diagnostic device. Predictions must be verified by a qualified healthcare professional."
