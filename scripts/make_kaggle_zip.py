"""Build the code package to upload to Kaggle: ONLY what training needs (no images, models, venv, reports, notebooks).

Usage (project root, venv active):   python scripts/make_kaggle_zip.py [--out PATH]
Default output: <folder above the project>/malaria-code.zip
Included: src/**/*.py, config.yaml, train.py, train_transfer.py, data/splits/{train,val,test}.csv + split_summary.json
"""
import argparse
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_MB = 20           # safety net: nothing bigger than this belongs in a code package


def collect():
    files = sorted(p for p in (ROOT / "src").rglob("*.py") if "__pycache__" not in p.parts)
    files += [ROOT / n for n in ("config.yaml", "train.py", "train_transfer.py")]
    files += [ROOT / "data" / "splits" / n for n in ("train.csv", "val.csv", "test.csv", "split_summary.json")]
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT.parent / "malaria-code.zip"))
    out = Path(ap.parse_args().out)

    files = collect()
    missing = [str(f.relative_to(ROOT)) for f in files if not f.is_file()]
    if missing:
        sys.exit("Missing required files:\n  " + "\n  ".join(missing) + "\nRun the splitter / unzip the phase files first.")
    big = [f for f in files if f.stat().st_size > MAX_FILE_MB * 1024 * 1024]
    if big:
        sys.exit(f"Refusing to package files over {MAX_FILE_MB} MB: {[str(b.relative_to(ROOT)) for b in big]}")

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    scaling = cfg.get("data", {}).get("scaling_by_model", {})
    if scaling.get("mobilenetv2") != "minus1_1":
        sys.exit("config.yaml: data.scaling_by_model.mobilenetv2 must be 'minus1_1' (MobileNetV2 expects [-1, 1] input).")

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT).as_posix())
    total = sum(f.stat().st_size for f in files) / 1e6
    print(f"Packaged {len(files)} files ({total:.1f} MB uncompressed, {out.stat().st_size / 1e6:.1f} MB zip)")
    for f in files:
        if not f.suffix == ".py" or f.parent == ROOT:
            print("  ", f.relative_to(ROOT).as_posix())
    print(f"  + {sum(1 for f in files if f.suffix == '.py' and f.parent != ROOT)} python files under src/")
    print("Saved:", out)


if __name__ == "__main__":
    main()
