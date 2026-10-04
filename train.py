"""Train a model:  python train.py --model custom_cnn --exp-id EXP-001 --epochs 10 --patience 3

Checks before a long run:
  python train.py --dry-run     builds the model and prints its summary (no data needed)
  python train.py --smoke       2 tiny batches end to end (needs the dataset); saves nothing
"""
import argparse

from src.config import load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="custom_cnn (default: config training.model_name)")
    ap.add_argument("--exp-id", default="EXP-001")
    ap.add_argument("--title", default=None, help="short title for the experiment log")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--patience", type=int, default=3, help="early-stopping patience (epochs without val_auc gain)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    model_name = a.model or cfg["training"]["model_name"]
    if a.dry_run:
        from src.models.model_loader import build_model
        m = build_model(model_name, cfg)
        m.summary()
        print("Parameters:", m.count_params())
        return
    from src.training.trainer import train
    train(model_name, a.exp_id, cfg, a.epochs, a.batch_size, a.lr, a.seed, a.patience, a.smoke, a.title)


if __name__ == "__main__":
    main()
