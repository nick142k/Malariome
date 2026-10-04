### Model comparison - val split (95% patient-bootstrap CI in brackets)

| Metric | Transfer Model (MobileNetV2) |
|---|---:|
| Accuracy | 0.951 (0.940-0.965) |
| Precision | 0.953 (0.909-0.970) |
| Recall / Sensitivity | 0.951 (0.929-0.964) |
| Specificity | 0.952 (0.930-0.971) |
| F1 | 0.952 (0.924-0.965) |
| ROC-AUC | 0.988 (0.982-0.993) |
| PR-AUC (average precision) | 0.989 (0.976-0.994) |
| Threshold used | 0.42 |
| Parameters | 2,259,265 |
| Model size (MB) | 21.73 |
| Inference, single image (ms) | 258.93 (CPU) |

Predictions are from a patient-level split; thresholds were chosen on validation data only.

Thresholds were selected on this same validation data, so threshold-dependent numbers here are optimistic.
