"""Group-aware, stratified train/val/test split.

Why grouped: many cell crops come from the same smear photo (and the same patient).
A random split would put sibling crops into train AND test and inflate test scores.
Every group (patient, or smear as fallback) therefore lands in exactly ONE split.

Usage (from the project root, venv active):
    python -m src.data.splitter
    python -m src.data.splitter --data-dir "D:\\path\\to\\cell_images" --group-by smear_id

Writes data/splits/{train,val,test}.csv, manifest.csv and split_summary.json.
Images are never copied or moved.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import load_config


IMG_EXTS = {".png", ".jpg", ".jpeg"}

# Patient identifier patterns found in the real dataset.
#
# Examples:
#   C100P61ThinF...       -> C100P61
#   C39P4thinF...         -> C39P4
#   C1_thinF...           -> C1
#   C37BP2_thinF...       -> C37BP2
#   C167P128ReThinF...    -> C167P128Re
#   C51AP12thinF...       -> C51AP12
#   C180P141NThinF...     -> C180P141N
#   C68P29N_ThinF...      -> C68P29N
#   C6NThinF...           -> C6N
#
# The non-greedy alphabetic parts are important: they prevent "ThinF"
# from being swallowed into the patient ID.
_PATIENT_RE = re.compile(
    r"^(C\d+[A-Za-z]*?(?:P\d+[A-Za-z]*?)?)(?=thin|_)",
    re.IGNORECASE,
)

_SMEAR_RE = re.compile(
    r"^(.*)_cell_\d+",
    re.IGNORECASE,
)


def parse_ids(filename):
    """Return (patient_id, smear_id, patient_parsed_ok) from a file name."""
    stem = Path(filename).stem

    sm = _SMEAR_RE.match(stem)
    smear = sm.group(1) if sm else stem

    pm = _PATIENT_RE.match(stem)

    if pm:
        return pm.group(1).upper(), smear, True

    # Fallback: use smear ID so the file is still grouped.
    return smear, smear, False


def build_manifest(data_dir, class_names, positive_class):
    rows = []

    for cls in class_names:
        folder = Path(data_dir) / cls

        if not folder.is_dir():
            raise FileNotFoundError(f"Class folder not found: {folder}")

        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                pid, sid, ok = parse_ids(f.name)

                rows.append(
                    {
                        "path": f"{cls}/{f.name}",
                        "class": cls,
                        "label": int(cls == positive_class),
                        "patient_id": pid,
                        "smear_id": sid,
                        "patient_parsed": ok,
                    }
                )

    return pd.DataFrame(rows)


def split_manifest(df, group_col, seed, ratios):
    """Two-stage StratifiedGroupKFold: carve out test, then validation from the rest."""

    n_test = max(2, round(1 / ratios["test"]))

    sgkf = StratifiedGroupKFold(
        n_splits=n_test,
        shuffle=True,
        random_state=seed,
    )

    rest_idx, test_idx = next(
        sgkf.split(
            df,
            df["label"],
            df[group_col],
        )
    )

    rest = df.iloc[rest_idx]

    n_val = max(
        2,
        round((1 - ratios["test"]) / ratios["val"]),
    )

    sgkf2 = StratifiedGroupKFold(
        n_splits=n_val,
        shuffle=True,
        random_state=seed,
    )

    tr_idx, va_idx = next(
        sgkf2.split(
            rest,
            rest["label"],
            rest[group_col],
        )
    )

    out = df.copy()
    out["split"] = ""

    out.iloc[
        test_idx,
        out.columns.get_loc("split"),
    ] = "test"

    out.loc[
        rest.index[va_idx],
        "split",
    ] = "val"

    out.loc[
        rest.index[tr_idx],
        "split",
    ] = "train"

    return out


def verify_split(df, group_col):
    """Hard checks. Raises AssertionError if any leakage or coverage problem exists."""

    assert (
        df["split"] != ""
    ).all(), "some images have no split"

    assert (
        df["path"].is_unique
    ), "duplicate paths in manifest"

    g = df.groupby(group_col)["split"].nunique()

    leaking = g[g > 1]

    assert leaking.empty, (
        f"{len(leaking)} groups appear in more than one split"
    )

    for s in ("train", "val", "test"):
        assert (
            df["split"] == s
        ).any(), f"split {s} is empty"


def summarize(df, group_col, seed):
    n = len(df)

    summ = {
        "seed": seed,
        "group_col": group_col,
        "total_images": n,
        "groups_total": int(df[group_col].nunique()),
        "patient_id_parsed_for_all_files": bool(
            df["patient_parsed"].all()
        ),
        "files_using_smear_fallback": int(
            (~df["patient_parsed"]).sum()
        ),
        "group_overlap_between_splits": 0,
        "splits": {},
    }

    for s in ("train", "val", "test"):
        d = df[df["split"] == s]

        summ["splits"][s] = {
            "images": len(d),
            "ratio": round(len(d) / n, 4),
            "groups": int(d[group_col].nunique()),
            "parasitized": int(d["label"].sum()),
            "uninfected": int((1 - d["label"]).sum()),
            "parasitized_fraction": round(
                float(d["label"].mean()),
                4,
            ),
        }

    return summ


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "--data-dir",
        help="folder containing the class folders "
        "(default: config / MALARIA_DATA_DIR)",
    )

    ap.add_argument(
        "--out-dir",
        help="where to write manifests "
        "(default: config splits_dir)",
    )

    ap.add_argument(
        "--group-by",
        choices=["patient_id", "smear_id"],
        help="override config data.split.group_by",
    )

    a = ap.parse_args(argv)

    cfg = load_config()

    data_dir = (
        Path(a.data_dir)
        if a.data_dir
        else cfg["abs_paths"]["data_dir"]
    )

    out_dir = (
        Path(a.out_dir)
        if a.out_dir
        else cfg["abs_paths"]["splits_dir"]
    )

    group_col = (
        a.group_by
        or cfg["data"]["split"]["group_by"]
    )

    seed = cfg["training"]["random_seed"]

    sp = cfg["data"]["split"]

    ratios = {
        "train": sp["train"],
        "val": sp["val"],
        "test": sp["test"],
    }

    if not Path(data_dir).is_dir():
        sys.exit(
            f"Dataset folder not found: {data_dir}\n"
            "Set data_dir in config.yaml or "
            "$env:MALARIA_DATA_DIR."
        )

    df = build_manifest(
        data_dir,
        cfg["classes"]["names"],
        cfg["classes"]["positive_class"],
    )

    if df.empty:
        sys.exit("No images found.")

    print(
        f"Images found: {len(df)}  "
        f"| distinct patient_ids: {df['patient_id'].nunique()}  "
        f"| distinct smear_ids: {df['smear_id'].nunique()}"
    )

    fallback_count = int(
        (~df["patient_parsed"]).sum()
    )

    print(
        "Files where patient id could not be parsed "
        f"(smear fallback used): {fallback_count}"
    )

    bad = df[
        ~df["patient_parsed"]
    ]["path"]

    if len(bad):
        print(
            "Examples of file names the patient pattern "
            "did NOT match:"
        )

        for name in bad.sample(
            min(8, len(bad)),
            random_state=0,
        ).tolist():
            print("  ", name)

    print("Example ids per class:")

    for cls in cfg["classes"]["names"]:
        print(
        f"  {cls}: "
        f"{df[df['class'] == cls]['patient_id'].drop_duplicates().head(3).tolist()}"        
        )
    df = split_manifest(
        df,
        group_col,
        seed,
        ratios,
    )

    verify_split(
        df,
        group_col,
    )

    summ = summarize(
        df,
        group_col,
        seed,
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        out_dir / "manifest.csv",
        index=False,
    )

    for s in ("train", "val", "test"):
        df[df["split"] == s].to_csv(
            out_dir / f"{s}.csv",
            index=False,
        )

    (
        out_dir / "split_summary.json"
    ).write_text(
        json.dumps(summ, indent=2),
        encoding="utf-8",
    )

    print(
        "\nSplit OK: no group appears in more than one split."
    )

    print(
        json.dumps(
            summ,
            indent=2,
        )
    )

    print(
        "\nWritten to:",
        out_dir,
    )


if __name__ == "__main__":
    main()