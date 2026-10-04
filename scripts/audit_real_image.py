from pathlib import Path
import sys

# Add project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.preprocessing import Preprocessor


cfg = load_config()
preprocessor = Preprocessor.from_config(cfg)

splits_dir = Path(cfg["abs_paths"]["splits_dir"])
data_dir = Path(cfg["abs_paths"]["data_dir"])

total = 0
failed = []

for split in ("train", "val", "test"):
    df = pd.read_csv(splits_dir / f"{split}.csv")
    print(f"{split}: {len(df)} images")

    for rel_path in df["path"]:
        total += 1
        image_path = data_dir / rel_path

        try:
            x = preprocessor.resize_image(image_path)

            if x.shape != (224, 224, 3):
                failed.append((split, rel_path, f"wrong shape: {x.shape}"))
            elif x.dtype != np.float32:
                failed.append((split, rel_path, f"wrong dtype: {x.dtype}"))
            elif not np.isfinite(x).all():
                failed.append((split, rel_path, "non-finite values"))

        except Exception as e:
            failed.append((split, rel_path, f"{type(e).__name__}: {e}"))

print()
print("Total audited:", total)
print("Failed:", len(failed))

if failed:
    print("Status: FAIL")
    print()
    print("First failures:")
    for item in failed[:20]:
        print(item)
else:
    print("Status: PASS")
