### Model comparison - val split (95% patient-bootstrap CI in brackets)

| Metric | Custom CNN | Transfer Model (MobileNetV2) |
|---|---:|---:|
| Accuracy | 0.929 (0.888-0.960) | 0.951 (0.940-0.965) |
| Precision | 0.912 (0.790-0.970) | 0.953 (0.909-0.970) |
| Recall / Sensitivity | 0.951 (0.931-0.970) | 0.951 (0.929-0.964) |
| Specificity | 0.907 (0.826-0.966) | 0.952 (0.930-0.971) |
| F1 | 0.931 (0.864-0.962) | 0.952 (0.924-0.965) |
| ROC-AUC | 0.980 (0.970-0.987) | 0.988 (0.982-0.993) |
| PR-AUC (average precision) | 0.980 (0.950-0.990) | 0.989 (0.976-0.994) |
| Threshold used | 0.44 | 0.42 |
| Parameters | NOT YET MEASURED | 2,259,265 |
| Model size (MB) | NOT YET MEASURED | 21.73 |
| Inference, single image (ms) | NOT YET MEASURED | 258.93 (CPU) |

Predictions are from a patient-level split; thresholds were chosen on validation data only.

Thresholds were selected on this same validation data, so threshold-dependent numbers here are optimistic.
