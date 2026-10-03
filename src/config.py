"""Central configuration loader. No machine-specific paths: everything is
resolved relative to the project root, with one env-var override for the dataset."""
import os
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path=None):
    """Load config.yaml and return a dict with resolved absolute paths under cfg['abs_paths']."""
    with open(path or CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    abs_paths = {k: PROJECT_ROOT / v for k, v in cfg["paths"].items()}
    env_data = os.environ.get("MALARIA_DATA_DIR")
    if env_data:
        abs_paths["data_dir"] = Path(env_data)
    cfg["abs_paths"] = abs_paths
    split = cfg["data"]["split"]
    if abs(split["train"] + split["val"] + split["test"] - 1.0) > 1e-9:
        raise ValueError("data.split ratios must sum to 1.0")
    return cfg
