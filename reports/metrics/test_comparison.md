### Model comparison - test split (95% patient-bootstrap CI in brackets)

| Metric | Custom CNN | Transfer Model (MobileNetV2) |
|---|---:|---:|
| Accuracy | 0.954 (0.935-0.973) | 0.961 (0.946-0.976) |
| Precision | 0.953 (0.890-0.976) | 0.965 (0.913-0.980) |
| Recall / Sensitivity | 0.958 (0.920-0.982) | 0.959 (0.926-0.984) |
| Specificity | 0.950 (0.916-0.977) | 0.963 (0.949-0.975) |
| F1 | 0.955 (0.911-0.972) | 0.962 (0.926-0.976) |
| ROC-AUC | 0.986 (0.971-0.993) | 0.992 (0.985-0.997) |
| PR-AUC (average precision) | 0.988 (0.960-0.994) | 0.993 (0.978-0.997) |
| Threshold used | 0.44 | 0.42 |
| Parameters | NOT YET MEASURED | NOT YET MEASURED |
| Model size (MB) | NOT YET MEASURED | NOT YET MEASURED |
| Inference, single image (ms) | NOT YET MEASURED | NOT YET MEASURED |

Predictions are from a patient-level split; thresholds were chosen on validation data only.
