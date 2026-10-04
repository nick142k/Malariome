"""Two-stage MobileNetV2 training:  python train_transfer.py --exp-id EXP-002

Checks first:
  python train_transfer.py --dry-run     builds the model, prints parameter counts for both stages (downloads ImageNet weights)
  python train_transfer.py --smoke       2 tiny batches per stage; saves nothing
Offline debugging only (random weights, smoke/dry-run only): add --no-pretrained
"""
import argparse

from src.config import load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-id", default="EXP-002")
    ap.add_argument("--title")
    ap.add_argument("--stage1-epochs", type=int)
    ap.add_argument("--finetune-epochs", type=int)
    ap.add_argument("--unfreeze-layers", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--finetune-lr", type=float)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    if a.dry_run:
        from src.models.transfer_learning import build_mobilenetv2, unfreeze_top_layers
        import numpy as np
        model, base = build_mobilenetv2(cfg["data"]["image_size"], weights=None if a.no_pretrained else "imagenet")
        count = lambda: sum(int(np.prod(w.shape)) for w in model.trainable_weights)
        print(f"Total parameters: {model.count_params():,} | stage 1 trainable: {count():,}")
        n = a.unfreeze_layers or cfg.get("transfer", {}).get("unfreeze_layers", 30)
        unfreeze_top_layers(base, n)
        print(f"After unfreezing the top {n} base layers (BatchNorm frozen): trainable {count():,}")
        return
    from src.training.staged_trainer import train_transfer
    train_transfer(a.exp_id, cfg, a.stage1_epochs, a.finetune_epochs, a.unfreeze_layers, a.lr, a.finetune_lr, a.batch_size,
                   a.seed, a.patience, a.smoke, not a.no_pretrained, a.title)


if __name__ == "__main__":
    main()
