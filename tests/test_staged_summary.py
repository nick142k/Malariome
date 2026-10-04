"""Pure-Python test of how the two training stages are combined (no TensorFlow)."""
from src.training.staged_trainer import combine_stage_summaries


def test_stage2_wins_when_better_and_epochs_are_global():
    s1 = {"epochs_run": 6, "best_epoch": 5, "val_auc": 0.95}
    s2 = {"epochs_run": 8, "best_epoch": 3, "val_auc": 0.98}
    out, chosen = combine_stage_summaries(s1, s2)
    assert chosen == "stage2" and out["val_auc"] == 0.98 and out["best_epoch"] == 9 and out["epochs_run"] == 14


def test_stage1_kept_when_finetuning_hurts():
    s1 = {"epochs_run": 6, "best_epoch": 5, "val_auc": 0.97}
    s2 = {"epochs_run": 4, "best_epoch": 1, "val_auc": 0.96}
    out, chosen = combine_stage_summaries(s1, s2)
    assert chosen == "stage1" and out["best_epoch"] == 5 and out["epochs_run"] == 10


def test_tie_goes_to_stage2():
    s1 = {"epochs_run": 2, "best_epoch": 2, "val_auc": 0.9}
    s2 = {"epochs_run": 2, "best_epoch": 1, "val_auc": 0.9}
    assert combine_stage_summaries(s1, s2)[1] == "stage2"
