# Experiment Log

Entries are appended; earlier experiments are never overwritten.

## EXP-001 — custom_cnn (CPU/GPU run)

- **Date (UTC):** 2026-10-04 00:42
- **Model:** custom_cnn (102,241 parameters, 101,793 trainable)
- **Dataset:** Kaggle cell_images, patient-level split (train 19,759 / val 3,868 / test 3,931 images; seed 42)
- **Image size:** 224x224 (pad_square, scaling=unit)
- **Batch size:** 32
- **Epochs:** 10 run of 10 planned (best epoch 8)
- **Optimizer / LR:** adam / 0.001 (ReduceLROnPlateau on val_auc)
- **Augmentation (train only):** {"rotation_degrees": 20, "horizontal_flip": true, "vertical_flip": true, "zoom": 0.1, "translation": 0.05, "brightness": 0.1, "contrast": 0.1}
- **Validation strategy:** fixed patient-level validation split; best epoch chosen by val_auc
- **Validation metrics at best epoch (threshold 0.5):** loss=0.1901, accuracy=0.9307, precision=0.9174, recall=0.9476, auc=0.9798, pr_auc=0.9797
- **Training time:** 274.5 min on CPU
- **Environment:** TensorFlow 2.20.0, Keras 3.15.1, seed 42
- **Test set:** NOT YET EVALUATED
- **Observations:** Validation AUC rose from 0.905 (epoch 1) to 0.974 (epoch 3), then plateaued between 0.970 and 0.980; best epoch 8 (val AUC 0.9798) is within 0.0002 of epoch 9, a difference well inside the epoch-to-epoch noise seen here. Train and validation accuracy are close at the best epoch (0.937 / 0.931): no sign of overfitting; training AUC is below validation AUC because augmentation and dropout act only in training. Validation recall varied between 0.886 and 0.960 across epochs, so the 0.5 threshold is unstable at this stage. Learning rate dropped to 5e-4 at epoch 7 and 2.5e-4 at epoch 10; early stopping did not trigger. Total training time 274.5 min on CPU.
- **Conclusion:** The custom CNN reaches validation AUC of about 0.98 at threshold 0.5 after 10 CPU epochs and looks limited by training length or capacity rather than overfitting. Not yet compared with transfer learning. Test set NOT YET EVALUATED.

