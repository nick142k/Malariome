"""Dataset verification for MalariaAI (Phase 3).

Usage:  python verify_dataset.py "<path to folder that contains Parasitized and Uninfected>"
Needs:  pip install pillow numpy
Output: prints a summary and writes dataset_report.json next to this script.
Read-only: it never modifies or moves your images.
"""
import sys, os, json, hashlib, re, random
from collections import Counter, defaultdict
from PIL import Image
import numpy as np

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def find_root(p):
    """Return the folder that directly contains the class folders."""
    for dirpath, dirs, _ in os.walk(p):
        low = {d.lower(): d for d in dirs}
        if "parasitized" in low and "uninfected" in low:
            return dirpath, low["parasitized"], low["uninfected"]
    sys.exit("Could not find Parasitized/ and Uninfected/ under: " + p)

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root, pcls, ucls = find_root(sys.argv[1])
    classes = [pcls, ucls]
    rep = {"root_found": root, "classes": classes, "per_class": {}}
    all_other = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d not in classes]
    rep["other_folders_in_root"] = all_other

    hashes = defaultdict(list)
    smears = defaultdict(set)
    sizes = Counter(); modes = Counter(); fmts = Counter()
    bad = []; non_img = Counter()
    samples = []
    random.seed(42)

    for c in classes:
        files = sorted(os.listdir(os.path.join(root, c)))
        n_img = 0; ext = Counter()
        for f in files:
            e = os.path.splitext(f)[1].lower()
            if e not in EXTS:
                non_img[f"{c}:{e or 'noext'}"] += 1; continue
            ext[e] += 1; n_img += 1
            path = os.path.join(root, c, f)
            try:
                with open(path, "rb") as fh:
                    hashes[hashlib.md5(fh.read()).hexdigest()].append((c, f))
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    sizes[im.size] += 1; modes[im.mode] += 1; fmts[im.format] += 1
                    if len(samples) < 600 and random.random() < 0.05:
                        samples.append(np.asarray(im.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0)
            except Exception as ex:
                bad.append((c, f, type(ex).__name__))
            m = re.match(r"^(.*)_cell_\d+", f)
            if m: smears[m.group(1)].add(c)
        rep["per_class"][c] = {"files_total": len(files), "images": n_img, "extensions": dict(ext)}

    counts = [rep["per_class"][c]["images"] for c in classes]
    rep["total_images"] = sum(counts)
    rep["class_ratio_parasitized_to_uninfected"] = round(counts[0] / counts[1], 4) if counts[1] else None
    rep["non_image_files"] = dict(non_img)
    rep["corrupted"] = {"count": len(bad), "examples": bad[:10]}
    rep["formats"] = dict(fmts); rep["color_modes"] = dict(modes)
    wl = [k[0] for k in sizes for _ in range(sizes[k])]; hl = [k[1] for k in sizes for _ in range(sizes[k])]
    rep["image_size"] = {"unique_sizes": len(sizes),
        "width": {"min": min(wl), "max": max(wl), "median": float(np.median(wl))},
        "height": {"min": min(hl), "max": max(hl), "median": float(np.median(hl))},
        "most_common": [[list(k), v] for k, v in sizes.most_common(5)]} if wl else {}
    dups = [v for v in hashes.values() if len(v) > 1]
    cross = [v for v in dups if len({c for c, _ in v}) > 1]
    rep["exact_duplicates"] = {"groups": len(dups), "extra_copies": sum(len(v) - 1 for v in dups),
        "groups_spanning_both_classes": len(cross), "examples": dups[:3]}
    rep["smear_key_check"] = {"filenames_matching_pattern": sum(1 for _ in smears),
        "smear_groups": len(smears),
        "groups_in_both_classes": sum(1 for v in smears.values() if len(v) > 1)}
    if samples:
        a = np.stack(samples)
        rep["pixel_stats_sample_64px"] = {"n_images": len(samples),
            "mean_rgb": [round(float(x), 4) for x in a.mean((0, 1, 2))],
            "std_rgb": [round(float(x), 4) for x in a.std((0, 1, 2))]}

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_report.json")
    with open(out, "w") as fh: json.dump(rep, fh, indent=2)
    for k, v in rep.items(): print(f"{k}: {json.dumps(v)}")
    print("\nSaved:", out)

if __name__ == "__main__":
    main()
