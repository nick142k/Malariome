# Malariome

## Explainable Deep Learning for Malaria Parasite Detection and Species Classification

Malariome is an end-to-end computer vision and deep learning project for automated analysis of malaria blood-cell microscopy images.

The system is designed as a multi-stage pipeline:

1. Detect whether a blood-cell image is **Parasitized** or **Uninfected**
2. If a parasite is detected, classify the parasite species
3. Generate a **Grad-CAM explanation** showing image regions contributing to the prediction
4. Present the result through a **Flask web application**

The project combines dataset exploration, patient/source-level data splitting, custom CNN development, transfer learning, species classification, explainability, quantitative evaluation, and web deployment.

---

## Project Objective

The primary objective of Malariome is to develop and evaluate a reproducible deep-learning pipeline for malaria microscopy image analysis.

The project focuses on three related goals:

- **Binary malaria detection** — determine whether a cell image contains a malaria parasite.
- **Parasite species classification** — identify the parasite species when infection is detected.
- **Explainable prediction** — provide visual evidence for model predictions using Grad-CAM.

The final system is intended as a research and demonstration system rather than a clinical diagnostic device.

---
## Dataset
This project uses two publicly available malaria image datasets for two complementary tasks: binary malaria detection and parasite species classification.

1. Cell Images for Detecting Malaria — Binary Detection
The binary detection component uses the Cell Images for Detecting Malaria dataset, originally associated with the National Library of Medicine (NIH) malaria dataset and distributed through Kaggle.

The dataset contains:

27,558 images

13,779 Parasitized

13,779 Uninfected

RGB PNG images

Two-class classification: Parasitized vs Uninfected

Before model development, the dataset was audited for image integrity, duplicates, dimensions, class distribution, and potential data leakage.

For this project, the data was divided at the patient level, rather than randomly at the image level:

Split	Images	Patients	Parasitized	Uninfected
Train	19,759	144	9,823	9,936
Validation	3,868	28	1,945	1,923
Test	3,931	28	2,011	1,920
Total	27,558	200	13,779	13,779

Patient-level splitting was used to reduce the risk of information leakage between training, validation, and test sets.

Dataset exploration and verification found:

0 corrupted/unreadable images

0 exact duplicate images

0 patient overlap between train, validation, and test sets

0 smear-group overlap between splits

All 27,558 images accounted for in the final splits

2. Malaria Parasite Image Database (MP-IDB) — Species Classification
The species-classification component uses a processed subset of the Malaria Parasite Image Database (MP-IDB).

Source repository:

MP-IDB: https://github.com/andrealoddo/MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis

The dataset provides microscopic malaria parasite images representing four Plasmodium species:

Split	Images	Patients	Parasitized	Uninfected
Train	19,759	144	9,823	9,936
Validation	3,868	28	1,945	1,923
Test	3,931	28	2,011	1,920
Total	27,558	200	13,779	13,779

Patient-level splitting was used to reduce the risk of information leakage between training, validation, and test sets.

Dataset exploration and verification found:

Plasmodium falciparum

Plasmodium malariae

Plasmodium ovale

Plasmodium vivax species-classification dataset is a processed subset of MP-IDB used specifically for this project; it should not be interpreted as the complete original

For this project, the processed dataset contains 210 images from 17 source groups:

Species	Sources	Images
P. falciparum	10	104
P. malariae	3	37
P. ovale	2	29
P. vivax	2	40
Total	17	210

Because multiple images can originate from the same microscopic source/smear, the species dataset was split at the source level rather than treating every image as an independent sample. Each source was assigned to exactly one of the train, validation, or test sets.

Dataset Usage in the Project
The two datasets serve different stages of the overall system:

Cell Images for Detecting Malaria
              │
              ▼
     Binary Detection Model
              │
       ┌──────┴──────┐
       │             │
  Uninfected     Parasitized
                       │
                       ▼
             Species Classification
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   Falciparum      Malariae         Ovale / Vivax

The binary dataset is therefore used to determine whether malaria parasites are present, while the MP-IDB dataset is used for the subsequent species-level classification task.

Note: The species-classification dataset is a processed subset of MP-IDB used specifically for this project; it should not be interpreted as the complete original MP-IDB dataset.

## System Overview

```text
                    Input blood-cell image
                             |
                             v
                    Image preprocessing
                             |
                             v
                  Binary parasite detector
                       /             \
                      /               \
             UNINFECTED            PARASITIZED
                 |                       |
                 |                       v
                 |                Species classifier
                 |                       |
                 |                       v
                 |                  Grad-CAM
                 |                       |
                 +-----------+-----------+
                             |
                             v
                       Flask web result

The binary detector is the first decision stage. Species classification is performed for images classified as parasitized, while Grad-CAM provides an interpretable visualization of the model's decision.

Main Components
1. Binary Parasite Detection
Two deep-learning approaches were investigated:

Custom CNN

MobileNetV2 transfer learning

The custom CNN provides a compact task-specific baseline, while MobileNetV2 provides a stronger transfer-learning comparison.

The final evaluation showed that the transfer-learning model achieved stronger overall discrimination on the independent test set.

2. Species Classification
A separate species-classification stage was developed for:

Plasmodium falciparum

Plasmodium malariae

Plasmodium ovale

Plasmodium vivax

Because the available species data are substantially smaller and less balanced than the binary detection dataset, species classification is treated as a separate experimental component rather than being combined directly with the binary detector.

3. Explainability
Grad-CAM is integrated into the inference pipeline to produce a visual heatmap highlighting regions that contributed to the model prediction.

This provides an additional layer of interpretability beyond the predicted class and confidence score.

4. Web Application
The trained models are integrated into a Flask application.

A user can:

Upload a microscopy image

Run the malaria detector

View the prediction and confidence

Obtain species classification when applicable

View the Grad-CAM visualization

Dataset
The main binary classification dataset contains:

Property	Value
Total images	27,558
Parasitized	13,779
Uninfected	13,779
Classes	2
Image format	PNG
Color mode	RGB
Patient groups	200
Corrupted images	0
Exact duplicate groups	0

The dataset was extensively explored before model development.

The exploratory analysis included:

class distribution

image dimensions

aspect ratios

image integrity

duplicate detection

pixel statistics

per-class visual/statistical characteristics

patient/group structure

potential shortcut features

split leakage checks

Detailed dataset analysis is available in:

notebooks/01_dataset_exploration.ipynb

and the generated evidence is stored under:

project_reports/

Data Splitting
A patient-level split is used rather than randomly splitting individual images.

The final binary dataset contains:

Split	Images	Patient groups	Parasitized	Uninfected
Train	19,759	144	9,823	9,936
Validation	3,868	28	1,945	1,923
Test	3,931	28	2,011	1,920
Total	27,558	200	13,779	13,779

The split was generated using patient identifiers with seed 42.

Independent verification found:

0 train/validation patient overlap

0 train/test patient overlap

0 validation/test patient overlap

0 smear-group overlap

0 path overlap

all images accounted for

This is important because multiple cell images may originate from the same patient or smear. Random image-level splitting could otherwise produce overly optimistic estimates of generalization.

Preprocessing
Input images are converted to RGB and resized to:

224 × 224 × 3

The project uses padded square resizing rather than simple stretching.

The training pipeline includes augmentation such as:

rotation

horizontal flipping

vertical flipping

zoom

translation

brightness variation

contrast variation

Augmentation is applied to training data only.

Model-specific input scaling is used:

Custom CNN: unit-range scaling

MobileNetV2: [-1, 1] scaling

Configuration is centralized in:

config.yaml

Model Development
Custom CNN
A compact custom convolutional neural network was developed as a task-specific baseline.

Key characteristics:

approximately 102k total parameters

224 × 224 RGB input

Adam optimizer

learning rate initially 0.001

batch size 32

data augmentation

validation-based model selection

The model was trained for 10 epochs in the final recorded experiment.

Best validation performance occurred during the later training epochs.

MobileNetV2 Transfer Learning
MobileNetV2 was evaluated as a transfer-learning alternative.

The training procedure used two stages:

Stage 1 — Transfer-learning head
The pretrained backbone was initially frozen while the classification head was trained.

Stage 2 — Fine-tuning
The upper portion of the backbone was unfrozen and fine-tuned using a substantially lower learning rate.

The final experiment used:

2,259,265 total parameters

1,512,001 trainable parameters in the final configuration

initial learning rate: 0.001

fine-tuning learning rate: 1e-5

top 30 layers unfrozen

batch size: 32

The recorded run completed 6 epochs of Stage 1 and 7 epochs of Stage 2 before the training session was interrupted. The best recorded Stage 2 checkpoint was retained.

Binary Detection Results
Evaluation was performed on an independent patient-level test set containing 3,931 images.

Custom CNN
Metric	Test
ROC-AUC	0.9861
PR-AUC	0.9876
Accuracy	95.57%
Sensitivity	95.72%
Specificity	95.42%
Precision	95.63%
F1	95.68%

MobileNetV2
Metric	Test
ROC-AUC	0.9920
PR-AUC	0.9928
Accuracy	95.98%
Sensitivity	94.93%
Specificity	97.08%
Precision	97.15%
F1	96.03%

The results indicate strong binary classification performance for both approaches.

MobileNetV2 achieved higher ROC-AUC and PR-AUC and slightly higher overall accuracy, while also producing higher specificity. The custom CNN achieved slightly higher sensitivity at the reported 0.5 threshold.

The project therefore retains the comparison rather than treating accuracy alone as the model-selection criterion.

Confidence Intervals
Patient-level bootstrap confidence intervals were calculated using 1,000 valid bootstrap resamples.

For the final MobileNetV2 test evaluation:

Metric	95% CI
ROC-AUC	0.9847 – 0.9970
PR-AUC	0.9779 – 0.9971
Accuracy	0.9457 – 0.9758
Sensitivity	0.9262 – 0.9844
Specificity	0.9494 – 0.9747
Precision	0.9126 – 0.9804
F1	0.9261 – 0.9761

Patient-level bootstrap resampling is used because the images are grouped by patient and individual images cannot be assumed to be statistically independent.

Threshold Evaluation
The system evaluates both:

the standard threshold of 0.50

a validation-selected operating threshold

For the final transfer-learning model, the selected threshold was 0.42.

On the independent test set at threshold 0.42:

Accuracy: 96.11%

Sensitivity: 95.92%

Specificity: 96.30%

Precision: 96.45%

F1: 96.19%

The threshold was selected using validation data and then evaluated on the independent test set.

Validation-set results at the selected threshold should be interpreted carefully because the threshold was selected on that same validation set.

Species Classification
A separate species-classification experiment was conducted using source-level splitting.

The evaluated species were:

Falciparum
Malariae
Ovale
Vivax

The species dataset contains fewer observations and substantial class imbalance.

The source-level splitting strategy ensures that individual sources are assigned to exactly one split.

The final source assignment was verified to have no source overlap between:

training

validation

testing

The species classifier produced the following recorded confusion matrix:

              Predicted
             F   M   O   V

Actual F    80   0   0   0
Actual M     0  63   8   3
Actual O     0   5  39  14
Actual V     0   3   8 133

The confusion matrix indicates strong performance for Falciparum and Vivax, while confusion between Malariae and Ovale/Vivax is more pronounced.

Because of the smaller and imbalanced species dataset, these results should not be interpreted as equivalent to the binary detection results.

Explainability
Grad-CAM is implemented to visualize regions contributing to model predictions.

The explainability stage is intended to answer an important question:

Which image regions influenced the model's classification?

The generated heatmap is presented alongside the prediction in the web application.

Grad-CAM is used as an interpretability aid rather than as a guarantee that the model has learned clinically valid biological features.

Web Application
The project includes a Flask web application.

The main inference flow is:

Image Upload
     |
     v
Preprocessing
     |
     v
Binary Detection
     |
     +------ Uninfected ------> Result
     |
     v
Parasitized
     |
     v
Species Classification
     |
     v
Grad-CAM
     |
     v
Browser Result

The application accepts common image formats including:

PNG
JPG
JPEG

The configured maximum upload size is 5 MB.

The default local development server is configured for:

127.0.0.1:5000

Repository Structure
Malariome/
│
├── app.py
├── config.yaml
├── evaluate.py
├── predict.py
├── predict_pipeline.py
├── train.py
├── train_species.py
├── train_transfer.py
├── requirements.txt
│
├── database/
│   └── ...
│
├── docs/
│   └── ...
│
├── models/
│   └── ...
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb
│
├── project_reports/
│   ├── evidence/
│   ├── metrics/
│   └── ...
│
├── reports/
│   ├── figures/
│   ├── metrics/
│   └── smoke/
│
├── scripts/
│   └── ...
│
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── explainability/
│   ├── inference/
│   ├── models/
│   └── training/
│
├── tests/
│   └── ...
│
└── web/
    └── templates/

Installation
Clone the repository:

git clone https://github.com/nick142k/Malariome.git
cd Malariome

Create a virtual environment:

Windows
python -m venv .venv
.venv\Scripts\Activate.ps1

Linux/macOS
python -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

The project uses TensorFlow, NumPy, Pandas, scikit-learn, Matplotlib, Pillow, OpenCV, Flask, PyYAML and pytest among its principal dependencies. 
G
GitHub

Dataset Configuration
The dataset directory is configured in:

config.yaml

The expected structure is:

data/
└── raw/
    └── cell_images/
        ├── Parasitized/
        └── Uninfected/

The configuration also supports overriding the dataset location using:

$env:MALARIA_DATA_DIR = "D:\path\to\cell_images"

The default configuration uses patient-level grouping and a nominal 70/15/15 train/validation/test split. 
G
GitHub

Running the Application
After the trained model artifacts are available:

python app.py

Then open:

http://127.0.0.1:5000

Upload a cell image through the web interface.

The application returns the appropriate prediction workflow, including species classification and explainability for parasitized predictions.

Evaluation
Evaluation can be performed using the project's evaluation scripts.

The repository contains dedicated evaluation functionality under:

src/evaluation/

including:

metric calculation

batch inference

report generation

Generated evaluation evidence is stored under:

project_reports/evidence/evaluation/

Testing
The project includes automated tests covering major components of the pipeline.

The final test suite recorded:

67 passed, 1 skipped

Tests include coverage for:

API behavior

preprocessing

evaluation metrics

species inference

staged-training summaries

Run the test suite with:

pytest

Research Evidence
The project maintains generated evidence separately from source code.

Important artifacts include:

project_reports/
├── evidence/
│   ├── evaluation/
│   ├── species/
│   └── training/
│
└── metrics/
    ├── dataset_metrics.json
    ├── performance_metrics.json
    └── training_metrics.json

Training histories are preserved for the major experiments, including:

EXP-001
Custom CNN

EXP-002
MobileNetV2
Stage 1
Stage 2

This allows the final results to be traced back to the recorded experiments.

Reproducibility
The project uses a fixed random seed:

42

Patient-level grouping is used during binary dataset splitting.

The project also records:

model configuration

image size

augmentation

optimizer

learning rate

training histories

evaluation metrics

confidence intervals

confusion matrices

This is intended to make the experimental process auditable and reproducible.

Limitations
The current system has several important limitations.

Dataset limitations
The system is evaluated on a specific microscopy dataset and therefore may not generalize to images from different laboratories, staining protocols, microscopes, acquisition devices, or populations.

Species classification
The species dataset is considerably smaller than the binary detection dataset and is imbalanced across species.

Consequently, species-level results should be interpreted more cautiously than the binary detector results.

Explainability
Grad-CAM provides a visualization of model activation but does not establish causal or clinical validity.

Clinical use
This project is a research prototype and should not be used as a standalone clinical diagnostic system.

A clinical deployment would require substantially broader external validation and prospective evaluation.

Project Status
The core research and application pipeline is complete.

Current capabilities include:

 Dataset exploration

 Dataset integrity checks

 Patient-level splitting

 Leakage verification

 Custom CNN training

 Transfer-learning training

 Model comparison

 Independent test evaluation

 Confidence interval estimation

 Species classification

 Species source-level splitting

 Grad-CAM explainability

 Flask web application

 Automated testing

 Experimental evidence collection

The remaining work is primarily documentation, research reporting, and thesis preparation.

Documentation
Detailed documentation is being organized under:

docs/

The planned documentation includes:

Project Report

Architecture

Dataset and Exploration

Model Development

Experiments

Evaluation

Species Classification

Explainability

Deployment

Testing

Limitations

Disclaimer
Malariome is an academic/research project.

It is not a medical device and its predictions should not be used as a substitute for diagnosis by a qualified healthcare professional.

License
License information for the project and underlying datasets should be reviewed before redistribution or commercial use.


### A few things I deliberately did **not** claim

I noticed some configuration comments in the current repository are older than the results you've now generated—for example, the configuration still describes the deployed threshold as a placeholder and contains some development-phase wording. 
So I **didn't copy those comments blindly into the README**. The README should describe the **final verified state**, while `config.yaml` can remain an implementation artifact until we decide whether to update it.

Likewise, I didn't claim that the species model is production-ready or that Grad-CAM proves clinical correctness.

### One important improvement before we commit this README

We should next create the **detailed `docs/PROJECT_REPORT.md`** from the actual project evidence. That is where we'll preserve the full development history—including the exploratory work, Custom CNN vs MobileNetV2 experiments, training stages, species experiments, leakage investigation, threshold work, Grad-CAM, deployment, and testing.

The README should stay relatively concise; the project report should contain the **full technical story**.

Malariome GitHub repository