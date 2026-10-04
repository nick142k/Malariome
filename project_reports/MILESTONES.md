MalariaAI: AI-Assisted Malaria Detection and Species Classification System
1. Abstract
Malaria is a parasitic disease caused by Plasmodium species and remains an important global public-health concern. Microscopic examination of blood-smear images is commonly used for malaria detection and species identification, but manual examination can be time-consuming and depends heavily on trained personnel.

This project, MalariaAI, develops an AI-assisted computer-vision pipeline for malaria screening from microscopic cell images. The system uses a two-stage deep-learning architecture. The first model determines whether a blood-cell image is parasitized or uninfected. If a parasite is detected, a second model attempts to identify the Plasmodium species as P. falciparum, P. malariae, P. ovale, or P. vivax.

The parasite-detection component uses a MobileNetV2 transfer-learning model, while the species classifier uses a separate MobileNetV2 architecture with class-weighted training to address substantial class imbalance.

The completed system is integrated into a Flask-based web application that supports image upload, prediction history, REST API access, model information, and prediction results. Unit and integration testing have also been performed.

The current system successfully implements parasite detection and species classification. Malaria developmental-stage classification is identified as a future extension and is not claimed by the current model.

2. Introduction
2.1 Background
Malaria is caused by infection with parasites of the genus Plasmodium. Microscopic examination of blood-smear images remains an important diagnostic technique, particularly for determining the presence of parasites and, where possible, identifying their species and developmental stages.

Traditional microscopy requires trained personnel and can become difficult to scale when large numbers of samples need to be examined.

Recent advances in deep learning and computer vision provide an opportunity to develop computer-assisted systems capable of analyzing microscopic images automatically.

The objective of MalariaAI is to develop an AI-assisted screening system that can analyze a microscopic cell image and provide:

A malaria parasite detection result.

A malaria probability score.

Species identification when a parasite is detected.

Confidence for the species prediction.

A web-based interface for practical demonstration and testing.

3. Problem Statement
The primary problem addressed by this project is the automated analysis of malaria microscopic images.

The system must distinguish between:

Parasitized cells.

Uninfected cells.

For images classified as parasitized, the system should further identify the likely Plasmodium species:

Plasmodium falciparum

Plasmodium malariae

Plasmodium ovale

Plasmodium vivax

A secondary future objective is the identification of malaria developmental stages such as ring, trophozoite, schizont, and gametocyte.

4. Objectives
4.1 Primary Objectives
The project objectives are:

Develop a deep-learning model for malaria parasite detection.

Use transfer learning to improve classification performance.

Evaluate the detection model using validation and test data.

Develop a separate model for malaria species classification.

Address species-class imbalance during training.

Build a reusable inference pipeline connecting both models.

Integrate the models into a web application.

Provide a REST API for programmatic predictions.

Perform automated unit and integration testing.

4.2 Future Objective
The planned future extension is:

Develop a third-stage classifier capable of identifying malaria developmental stages.

5. System Architecture
The system follows a sequential two-stage architecture.

                    Microscopic Cell Image
                            |
                            v
                  +----------------------+
                  | Parasite Detection   |
                  |     MobileNetV2      |
                  +----------+-----------+
                             |
                +------------+------------+
                |                         |
                v                         v
           UNINFECTED                PARASITIZED
                |                         |
                v                         v
              STOP               Species Classification
                                         |
                         +---------------+---------------+
                         |       |       |               |
                         v       v       v               v
                    Falciparum Malariae Ovale          Vivax

The important design decision is that species classification is performed only when the first model identifies the image as parasitized.

This prevents the species classifier from unnecessarily assigning a malaria species to an image classified as uninfected.

6. Dataset
6.1 Parasite Detection Dataset
The original malaria cell-image dataset contains two primary categories:

Parasitized

Uninfected

The project uses patient-aware splitting to reduce the possibility of patient-level information leakage between training and evaluation data.

The parasite detector is based on these two categories.

6.2 Species Dataset
The species dataset was organized under:

data/species/MP-IDB/

The four species are:

Species	Images
Falciparum	1,505
Malariae	74
Ovale	58
Vivax	144
Total	1,781

The dataset is strongly imbalanced, with P. falciparum representing the largest class and P. ovale the smallest.

7. Class Imbalance Handling
Class imbalance was explicitly addressed during species-model training.

The calculated class weights were:

Species	Class Weight
Falciparum	0.2939
Malariae	6.3616
Ovale	8.2849
Vivax	3.1250

Higher weights were assigned to minority classes so that errors involving those classes contributed more strongly to the training loss.

This was important because simply training on the raw class distribution could result in a model that strongly favors P. falciparum.

8. Parasite Detection Model
8.1 Architecture
The parasite detector uses MobileNetV2 transfer learning.

The model was selected because it provides a useful balance between:

Classification performance.

Model size.

Computational cost.

Suitability for deployment.

The trained model is stored as:

models/transfer_model.keras

The selected classification threshold is:

0.42

This threshold was determined using validation data and subsequently frozen for test evaluation.

9. Parasite Detection Evaluation
The validation evaluation produced the following results:

Metric	Result
Accuracy	0.951
Precision	0.953
Recall / Sensitivity	0.951
Specificity	0.952
F1 Score	0.952
ROC-AUC	0.988
PR-AUC	0.989
Threshold	0.42

The reported metrics were accompanied by patient-bootstrap confidence intervals.

The model therefore demonstrated strong performance on the validation set.

The project evaluation process also enforces separation between validation-based threshold selection and final test evaluation to reduce evaluation leakage.

10. Computational Performance
The parasite detector benchmark was performed on CPU.

Metric	Result
Parameters	2,259,265
Model size	21.73 MB
Single-image mean inference	258.93 ms
Single-image median inference	243.83 ms
Batch-32 inference	113.9 ms/image
TensorFlow	2.20.0

Preprocessing time was not included in the reported inference timing.

11. Species Classification Model
11.1 Architecture
The species classifier also uses MobileNetV2 transfer learning.

The architecture consists of:

Input: 224 × 224 × 3
        |
Data augmentation
        |
MobileNetV2 backbone
        |
Global Average Pooling
        |
Dropout
        |
Dense(128)
        |
Dropout
        |
Dense(4)

The four output classes are:

0 = Falciparum
1 = Malariae
2 = Ovale
3 = Vivax

The trained model is stored as:

models/species_model.keras

The model contains approximately:

2,422,468 total parameters

with approximately:

164,484 trainable parameters

12. Species Model Training
The model was trained using class-weighted learning.

Training stopped early at epoch 23, with the best validation model restored from epoch 17.

The best validation accuracy was:

88.48%

The best validation loss was:

0.3024

This indicates that the model learned useful discriminative features for the four species while avoiding unnecessary additional training after validation performance stopped improving.

13. Species Classification Results
The final validation classification report was:

Species	Precision	Recall	F1	Support
Falciparum	1.0000	1.0000	1.0000	80
Malariae	0.8873	0.8514	0.8690	74
Ovale	0.7091	0.6724	0.6903	58
Vivax	0.8867	0.9236	0.9048	144

Overall:

Metric	Result
Accuracy	0.8848
Macro Precision	0.8708
Macro Recall	0.8618
Macro F1	0.8660
Weighted F1	0.8838

The results show particularly strong performance for P. falciparum and P. vivax.

The P. ovale class remains the weakest-performing class, which is consistent with its substantially smaller dataset size.

14. Species Confusion Matrix
The validation confusion matrix was:

                 Predicted
              F    M    O    V

Actual F      80   0    0    0
Actual M       0  63    8    3
Actual O       0   5   39   14
Actual V       0   3    8  133

The largest confusion occurs between:

P. ovale and P. vivax

P. malariae and P. ovale

This indicates that additional representative samples, particularly for P. ovale, could improve future performance.

15. Inference Pipeline
A dedicated inference adapter was implemented at:

src/inference/predictor.py

The predictor exposes a common interface to the Flask application.

The prediction process is:

Image
  |
  v
Parasite Detection
  |
  +-- probability < 0.42 --> UNINFECTED
  |
  +-- probability >= 0.42 --> PARASITIZED
                                  |
                                  v
                           Species Classifier
                                  |
                                  v
                    Species + Species Probability

This design keeps the web application independent of the internal model implementation.

16. Web Application
The project includes a Flask web application.

The application provides:

Image upload.

Malaria prediction.

Species prediction.

Prediction history.

Model information.

About page.

REST API.

Health endpoint.

Upload validation.

CSRF protection.

Upload size restrictions.

Temporary-file cleanup.

Error handling.

The main application file is:

app.py

The web templates are located under:

web/templates/

including:

index.html
result.html
history.html
model.html
about.html
error.html
base.html

The application can be launched using:

python app.py

and accessed locally through:

http://localhost:5000

17. REST API
The web application provides an API endpoint:

POST /api/predict

The API can return:

{
  "success": true,
  "prediction": "PARASITIZED",
  "malaria_probability": 0.91,
  "threshold": 0.42,
  "model": "MobileNetV2 transfer_model",
  "species": "Falciparum",
  "species_probability": 0.87
}

For an uninfected image, species classification is not performed.

The application also provides:

GET /api/health

to report whether the model pipeline is loaded.

18. Testing
Automated testing was performed throughout development.

The final unit-test result was:

65 passed in 11.79s

Therefore:

65/65 tests passed.

Integration testing was also successfully completed after connecting the parasite and species models through the inference pipeline.

The complete system was subsequently launched through the Flask application and tested locally.

19. Explainability
The application architecture includes support for an optional explain() method in the predictor interface.

This is intended to support Grad-CAM or similar visual explanation techniques.

The application describes such visualizations as interpretability aids rather than biological proof of causality.

Further Grad-CAM implementation can be added as a future enhancement.

20. Security and Reliability Features
Several basic application-level protections have been implemented:

Secure filename handling.

Randomized uploaded-file names.

File-extension validation.

Image decoding validation.

Upload size limits.

CSRF protection for web forms.

Session security configuration.

Temporary upload cleanup.

API-specific error responses.

Model availability checking.

Degraded-mode application startup if the model cannot be loaded.

These features improve the reliability and safety of the demonstration system.

21. Current Limitations
Despite promising results, the system has several limitations.

21.1 Species Dataset Imbalance
The species dataset is heavily imbalanced:

Falciparum = 1505
Malariae   = 74
Ovale      = 58
Vivax      = 144

Although class weighting was applied, the smaller classes remain more difficult to classify.

21.2 Ovale Performance
The P. ovale F1 score is approximately 0.69.

More representative and diverse training samples are likely required to improve this class.

21.3 Species Model Scope
The species model identifies four species:

P. falciparum

P. malariae

P. ovale

P. vivax

It should not be interpreted as covering every possible malaria parasite species.

21.4 Developmental Stage Identification
The current system does not reliably identify malaria developmental stages.

Stages such as:

Ring

Trophozoite

Schizont

Gametocyte

require appropriately stage-labeled training data and a dedicated classifier.

Therefore, the current project should be described as:

Malaria parasite detection and species classification, rather than complete species-and-stage diagnosis.

21.5 Medical Use
The system is an AI-assisted research/screening system and is not a replacement for professional laboratory diagnosis.

Predictions should be verified by qualified healthcare or laboratory professionals.

22. Future Work
The next development phase should focus on the following:

22.1 Malaria Stage Classification
Acquire a properly annotated dataset containing developmental-stage labels and develop a third model:

Parasite Detection
        ↓
Species Identification
        ↓
Stage Identification

22.2 Improve Ovale Classification
Increase the number and diversity of P. ovale images and investigate:

Targeted augmentation.

Better sampling strategies.

Fine-tuning of the MobileNetV2 backbone.

Additional transfer-learning architectures.

Per-class calibration.

22.3 Fine-Tuning
The current models rely heavily on pretrained MobileNetV2 representations.

Future experiments can unfreeze selected backbone layers and perform low-learning-rate fine-tuning.

22.4 Explainable AI
Implement Grad-CAM for the parasite and species models so users can visualize image regions influencing predictions.

22.5 External Validation
The system should eventually be tested on an independent dataset collected from a different source to determine how well it generalizes beyond the training distribution.

22.6 Deployment
Future work may include containerization and deployment using a production server, together with appropriate security, monitoring, and model-version management.

23. Project Directory Structure
The major project components currently include:

malaria-detection/
│
├── app.py
├── predict.py
├── evaluate.py
├── train_species.py
│
├── data/
│   ├── raw/
│   │   └── cell_images/
│   │       ├── Parasitized/
│   │       └── Uninfected/
│   │
│   └── species/
│       ├── MP-IDB/
│       └── classification/
│
├── models/
│   ├── custom_cnn.keras
│   ├── custom_cnn.meta.json
│   ├── transfer_model.keras
│   ├── transfer_model.meta.json
│   ├── species_model.keras
│   └── species_model.meta.json
│
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── explainability/
│   ├── inference/
│   │   └── predictor.py
│   ├── models/
│   └── training/
│
├── web/
│   ├── templates/
│   └── static/
│
├── project_reports/
│
└── backup/

24. Conclusion
The MalariaAI project has successfully progressed from binary malaria-cell classification to a multi-stage AI-assisted screening pipeline.

The first model provides parasite detection with strong validation performance, achieving approximately 95.1% validation accuracy and a 0.988 ROC-AUC.

A second MobileNetV2 model was developed for species classification. Despite significant class imbalance, class-weighted training produced an overall validation accuracy of foundation for an AI-assisted malaria screening and species-classification platform, while clearly maintaining the distinction between research 88.48% and a macro F1 score of 0.866.

The models have been integrated through a dedicated inference layer, allowing the system to first determine whether a parasite is present and then identify the likely species when appropriate.

The resulting pipeline has been integrated into a Flask web application with image upload, prediction history, model information, and REST API functionality.

Automated testing has also been completed successfully, with 65 out of 65 unit tests passing, followed by successful integration testing.

The major remaining research extension is malaria developmental-stage classification. This should be implemented only after obtaining appropriately annotated stage-level data.

Overall, the current system provides a functional foundation for an AI-assisted malaria screening and species-classification platform, while clearly maintaining the distinction between research-oriented machine-learning predictions and clinical diagnosis.