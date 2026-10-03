# Reference Analysis — afloresep/CNN-Malaria-Detection

**Status of this document:** Phase 1 deliverable. Every statement below comes from reading the repository ZIP supplied for this project (`CNN-Malaria-Detection-master.zip`) or the repository's public front page. Scripts were **read, not executed**. Anything that could not be confirmed is marked **NOT VERIFIED**.

---

## 1. Original project

| Item | Finding |
|---|---|
| Repository | https://github.com/afloresep/CNN-Malaria-Detection (branch `master`, 20 commits at time of review, 1 star, 0 forks) |
| README title | "Malaria Detection and Species Classification" |
| README intent | Classify red-blood-cell images as infected/uninfected and also predict parasite species |
| What the code actually does | Binary classification only (`Parasitized` vs `Uninfected`). No species-classification code was found in any file reviewed |
| README quality | Mostly template text: placeholder clone URL, unfilled Results section, no metrics |
| License | **No LICENSE file** in the repository root (README lists a "License" heading with no content) |
| Dependencies | `requirements.txt`: matplotlib 3.7.2, numpy 1.24.3, pandas 2.0.3, scikit-learn 1.3.0, scipy 1.11.1, setuptools 65.5.0, tensorflow 2.13.0. Written with `=` instead of `==`, so `pip install -r` will reject it |

### File inventory

```
.gitignore
README.md
requirements.txt
data/cell_images/{Parasitized,Uninfected}/   (empty in the ZIP — images not included)
models/descriptions.md                       author's experiment notes
models/first-model.py                        baseline CNN script (10 epochs)
models/keras-95.py                           script version of the "keras accuracy 95" notebook
models/malaria-detection-using-keras-accuracy-95 (1).ipynb
results/previsualization.png                 only file in results/
src/binary-classification.py                 same as first-model.py except epochs=50
src/binary-classification.ipynb              recorded run outputs
src/preprocessing.py  src/model.py  src/train.py  src/evaluate.py
src/tools.py  src/visualization.py
utils/imports.py
```

## 2. Original architecture

**Model A — baseline CNN** (`first-model.py`, `binary-classification.py`, `model.py::design_model`). Source comment credits "Sumit Kumar et al." (arXiv 2303.03397; the paper itself was not checked).

- Input 64×64×3
- Conv2D(32,3×3,ReLU) → MaxPool(2×2) → BatchNorm → Dropout(0.2)
- Conv2D(32,3×3,ReLU) → MaxPool(2×2) → BatchNorm → Dropout(0.2)
- Flatten → Dense(512,ReLU) → BN → Dropout(0.2) → Dense(256,ReLU) → BN → Dropout(0.2)
- Dense(2, softmax)
- Parameters (from notebook / `descriptions.md` summary): **3,357,090 total; 3,355,426 trainable; 1,664 non-trainable**

**Model B — deeper custom CNN** (`keras-95.py`, notebook): five blocks of two Conv2D layers (16→32→64→96→128 filters, dilated convs in blocks 4–5), MaxPool + Dropout(0.5) + BatchNorm per block, Flatten, Dense(256), Dense(2, softmax). Notebook summary: **505,234 parameters** (503,794 trainable).

**Model C — pretrained backbone factory** (`get_model`): Xception, ResNet50, InceptionV3, InceptionResNetV2, DenseNet201, NASNetMobile/Large with a concatenated GlobalMax + GlobalAvg + Flatten head. The notebook instantiates NASNetMobile (4,339,414 parameters). The `weights` argument is not set explicitly except for InceptionV3.

## 3. Dataset

- **Source (README):** Kaggle — `iarunava/cell-images-for-detecting-malaria`.
- **Structure (code):** `data/cell_images/Parasitized/` and `data/cell_images/Uninfected/`; class mapping comment `{'Parasitized': 0, 'Uninfected': 1}`.
- **Images:** not shipped in the ZIP (both folders empty). **Image counts, file formats and native image dimensions are NOT VERIFIED from files.**
- **Counts implied by recorded notebook outputs:**
  - 19,290 train + 4,134 validation + 4,134 test = **27,558 images**, 2 classes, equal per-class split sizes.
  - The notebook also prints "Found 55,116 files belonging to 5 classes" — i.e. 2 × 27,558, consistent with the original class folders plus the copied `train/test/validation` folders being scanned together (inference, not confirmed).
  - Code comments say 13,780 images per class / 9,646 train; the notebook outputs imply 13,779 per class / 9,645 train. **Verified 2026-10-03 on the user's copy:** each class folder holds 13,780 files, of which one is a `.db` file (Thumbs.db-type), leaving 13,779 images per class (27,558 total). The notebook counts are therefore correct and the code comments were off by one (they counted the non-image file).
- **Example filename in a code comment:** `C13NThinF_IMG_20150614_131318_cell_179.png` — suggests cells are cropped from larger smear images. If several crops from one smear land in different splits, results may be optimistic. **Needs checking in Phase 3.**
- Dataset license/terms: **NOT VERIFIED.**

## 4. Training methodology

| Aspect | Baseline (`first-model.py`) | Keras-95 notebook |
|---|---|---|
| Split | `tools.py::split_folder_to_train_test_valid`: per class, `train_test_split(test_size=0.3, random_state=42)` then 50/50 → ~70/15/15, seed 42, files **copied** into train/test/validation folders | `validation_split=0.25` of the whole set (20,670 train / 6,888 val); no test set |
| Image size | 64×64 | generator `target_size=(224,224)` but model built with `input_shape=(96,96,3)` — mismatch |
| Batch size | 64 | 176 |
| Optimizer / LR | Adam, 1e-3 | Adam, 1e-4 |
| Loss | categorical cross-entropy (softmax, 2 units) | categorical cross-entropy |
| Metrics | categorical accuracy, AUC | accuracy |
| Epochs | 10 (`first-model.py`), 50 (`binary-classification.py`) | 10 |
| Augmentation | train only: rescale 1/255, zoom 0.2, rotation 15°, shifts 0.05 | rescale, horizontal + vertical flip, rotation 15°, shifts — **also applied to the validation generator** (built from the same augmenting generator) |
| Callbacks | EarlyStopping(`val_auc`, mode `min`, patience 20) | ModelCheckpoint(`val_acc`), EarlyStopping(`val_loss`, patience 2), ReduceLROnPlateau(0.6) |
| Class weights | none | `get_weight` exists; set to placeholder string `'lol'` in the script |
| Model saving | none found in baseline scripts | checkpoint path variable only (`model_file`) |

**Recorded results (`src/binary-classification.ipynb`, 50-epoch run):** training accuracy rises to about 0.946 and training AUC to about 0.981 in the epochs shown, while validation accuracy stays at ≈0.50 with validation loss in the hundreds to thousands. Final test evaluation: accuracy 0.5000, loss ≈ 72,831; the printed confusion matrix has a first row of `[0 2067]`, i.e. one class is never predicted. The author's own note in `models/descriptions.md` flags the validation problem as unresolved. The keras-95 notebook has no completed epoch output, so the "accuracy 95" in its filename is **NOT VERIFIED**.

## 5. Existing strengths

1. Clear binary framing and a documented dataset source.
2. Per-class (stratified) 70/15/15 split with a fixed seed — the same idea this project needs.
3. Train-only augmentation in the baseline script; augmentation is mild (small rotation, shift, zoom).
4. Includes batch normalization, dropout and AUC tracking.
5. Honest experiment notes: the author recorded the failed validation run instead of hiding it.
6. Cites a source paper for the baseline architecture.

## 6. Existing limitations

1. **Validation/test preprocessing mismatch (likely cause of the 0.50 result):** training generator uses `rescale=1/255`, but `validation_data_generator` and `test_data_generator` are `ImageDataGenerator()` with no rescale, so evaluation images are 0–255. This is a code fact; that it explains the failure was not tested.
2. Final classification report/confusion matrix is computed on the **validation** set, not the test set.
3. Early stopping watches `val_auc` with `mode='min'` (wrong direction) while the logged name is `val_auc_5`, so it likely never triggers correctly.
4. Validation data is augmented in the keras-95 setup; no held-out test set there.
5. Source modules are notebook fragments and **cannot run standalone**: `train.py` and `evaluate.py` use undefined variables and `fit_generator`/`numpy.math`; `preprocessing.py` passes invalid arguments (`hodizontal_flip`, `rotation_angle`); `model.py::get_model` uses names it never imports; `utils/imports.py` imports `keras.layers.pooling._GlobalPooling1D` and packages (seaborn, mlxtend, tqdm, cv2) missing from `requirements.txt`.
6. No inference code, saved model, API, UI, tests, Docker, explainability, threshold analysis or documented results.
7. Softmax with 2 units and categorical loss for a binary task; no threshold handling.
8. Split copies files on disk and is guarded only by a folder-count check; no manifest for reproducibility.
9. README promises species classification that the code does not implement.
10. No license; `.gitignore` contains duplicate entries.

## 7. What is being retained

- The Kaggle dataset and the `Parasitized` / `Uninfected` folder structure.
- The idea of a per-class stratified 70/15/15 split with seed 42 (re-implemented, with a manifest).
- The baseline CNN as an optional **reference-baseline replication** (proposed `EXP-000`, with the preprocessing mismatch fixed) so we can compare against the original design honestly.
- Mild augmentation magnitudes as a starting point.

No code is copied. With no license present, the new implementation is written from scratch, and the repository is credited as a reference.

## 8. What is being redesigned

- Preprocessing: one shared function for training and inference; validation/test never augmented and always preprocessed identically.
- Output head: single sigmoid unit + binary cross-entropy.
- Split: manifest CSVs in `data/splits/` with seed and counts logged, instead of copying files.
- Callbacks: EarlyStopping / ModelCheckpoint / ReduceLROnPlateau on the correct monitor and direction.
- Evaluation: held-out test set only for final reporting; threshold chosen on validation data.
- Dependencies: valid, pinned `requirements.txt`.

## 9. What is newly implemented

Custom CNN (GAP head), MobileNetV2 (+ optional EfficientNetB0) staged fine-tuning, `train.py` / `evaluate.py` / `predict.py`, experiment tracking, full metrics and threshold analysis, model comparison, Grad-CAM, inference engine, Flask app and REST API, SQLite history, security hardening, pytest suite, Docker, model card, limitations, report set and viva material.

## 10. Proposed improved architecture (to be finalized in Phase 2)

```
Kaggle dataset → validation/dedup check → stratified split manifest (seed 42)
→ shared preprocessing (RGB, resize, model-specific scaling)
→ Custom CNN (scratch)  |  MobileNetV2 (frozen → fine-tune)
→ train.py (callbacks, history, metadata) → evaluate.py (same test set, thresholds from val)
→ model selection → Grad-CAM → predictor.py → Flask (UI + /api/predict, /api/health) → SQLite
→ pytest → Docker
```

Open decisions for the user: image size (128 vs 224 — 224 favours MobileNetV2), and whether to include the `EXP-000` baseline replication.

## 11. Evidence not yet collected

Dataset counts, image dimensions, duplicate/corruption checks and patient/smear-level leakage risk — all Phase 3 (needs the actual images).
