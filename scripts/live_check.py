"""Black-box check of a RUNNING MalariaAI web app (default http://localhost:5000).

  python scripts/live_check.py [--url http://localhost:5000] [--n 10]

What it does (validation images only, never the test split):
  1. health endpoint and the four web pages respond
  2. sends --n parasitized and --n uninfected validation images to POST /api/predict
  3. compares the app's probability with the one saved by evaluate.py for the SAME image
     (reports/predictions/<deployed model>_val.csv): a mismatch means the app preprocesses differently from evaluation
  4. checks the cascade rule: species fields only for PARASITIZED, never for UNINFECTED
  5. checks error handling (garbage file, wrong extension, missing field)
Writes project_reports/evidence/web/live_check.json
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def request(url, data=None, headers=None):
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except urllib.error.URLError as e:
        sys.exit(f"Cannot reach {url}: {e.reason}. Is the app running (python app.py)?")


def post_file(url, content, filename, field="image"):
    b = uuid.uuid4().hex
    body = (f'--{b}\r\nContent-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + content + f"\r\n--{b}--\r\n".encode()
    status, raw = request(url, body, {"Content-Type": f"multipart/form-data; boundary={b}"})
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, {"_raw": raw[:200].decode("utf-8", "replace")}


def compare_probabilities(app_probs, ref_probs, tol=1e-3):
    """Pure helper: summary of |app - evaluation| over the same images."""
    d = np.abs(np.asarray(app_probs, float) - np.asarray(ref_probs, float))
    return {"n": int(d.size), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            "within_tolerance": int((d <= tol).sum()), "tolerance": tol, "all_match": bool((d <= tol).all())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:5000")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    base = a.url.rstrip("/")
    checks, report = [], {"url": base}

    def check(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))

    status, raw = request(base + "/api/health")
    health = json.loads(raw) if status == 200 else {}
    check("health endpoint", status == 200 and health.get("model_loaded") is True, str(health))
    for page in ("/", "/history", "/model", "/about"):
        s, body = request(base + page)
        check(f"page {page}", s == 200 and (page != "/" or b"not a medical diagnostic device" in body), f"HTTP {s}")

    from src.config import load_config
    from src.data.loader import load_split
    cfg = load_config()
    paths, labels, df = load_split("val", cfg)
    ref_file = Path(cfg["abs_paths"]["reports_dir"]) / "predictions" / f"{cfg['inference']['deployed_model']}_val.csv"
    ref = pd.read_csv(ref_file).set_index("path")["prob"] if ref_file.exists() else None
    if ref is None:
        print(f"NOTE: {ref_file} not found, probability comparison skipped (run: python evaluate.py --split val)")
    rng = np.random.RandomState(42)
    rows = []
    for lab in (1, 0):
        for i in rng.choice(np.flatnonzero(labels == lab), min(a.n, int((labels == lab).sum())), replace=False):
            t0 = time.perf_counter()
            s, j = post_file(base + "/api/predict", Path(paths[i]).read_bytes(), Path(paths[i]).name)
            rows.append({"path": df["path"].iloc[i], "true": int(lab), "status": s, "ms": (time.perf_counter() - t0) * 1000, **j})
    ok_rows = [r for r in rows if r["status"] == 200 and r.get("success")]
    check("API answered every request", len(ok_rows) == len(rows), f"{len(ok_rows)}/{len(rows)} succeeded")
    check("probabilities in [0, 1]", all(0 <= r["malaria_probability"] <= 1 for r in ok_rows))
    check("decision matches threshold", all((r["malaria_probability"] >= r["threshold"]) == (r["prediction"] == "PARASITIZED") for r in ok_rows))
    check("species absent for UNINFECTED", all("species" not in r for r in ok_rows if r["prediction"] == "UNINFECTED"))
    if ref is not None and ok_rows:
        cmp = compare_probabilities([r["malaria_probability"] for r in ok_rows], [ref[r["path"]] for r in ok_rows])
        report["probability_vs_evaluation"] = cmp
        check("app probability equals evaluation probability for the same images", cmp["all_match"],
              f"max diff {cmp['max_abs_diff']:.4f}, mean diff {cmp['mean_abs_diff']:.4f}, {cmp['within_tolerance']}/{cmp['n']} within {cmp['tolerance']}")
    if ok_rows:
        pos = [r for r in ok_rows if r["prediction"] == "PARASITIZED"]
        acc = np.mean([(r["prediction"] == "PARASITIZED") == bool(r["true"]) for r in ok_rows])
        report["accuracy_on_sample"] = float(acc)
        report["species_distribution_on_parasitized_calls"] = dict(Counter(r.get("species", "(none)") for r in pos))
        report["latency_ms"] = {"mean": float(np.mean([r["ms"] for r in ok_rows])), "max": float(np.max([r["ms"] for r in ok_rows]))}
        print(f"\nSample accuracy {acc:.2f} ({len(ok_rows)} images) | latency mean {report['latency_ms']['mean']:.0f} ms")
        print("Species chosen for cells called PARASITIZED:", report["species_distribution_on_parasitized_calls"])
    s, j = post_file(base + "/api/predict", b"not an image", "x.png")
    check("garbage file rejected (400)", s == 400 and j.get("success") is False, f"HTTP {s}")
    s, j = post_file(base + "/api/predict", b"GIF89a", "x.gif")
    check("wrong extension rejected (415)", s == 415, f"HTTP {s}")
    s, j = post_file(base + "/api/predict", b"x", "x.png", field="wrong")
    check("missing 'image' field rejected (400)", s == 400, f"HTTP {s}")

    report["checks"] = checks
    report["all_passed"] = all(c["passed"] for c in checks)
    out = Path(cfg["abs_paths"]["project_reports_dir"]) / "evidence" / "web" / "live_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\n{'ALL CHECKS PASSED' if report['all_passed'] else 'SOME CHECKS FAILED'} | saved {out}")
    sys.exit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
