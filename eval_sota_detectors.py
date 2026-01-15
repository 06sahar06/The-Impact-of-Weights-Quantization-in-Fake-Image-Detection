import os
import sys
import json
import glob
import csv
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

from sklearn.metrics import roc_auc_score, accuracy_score


WORKSPACE = Path(__file__).parent.resolve()
DETECTORS_ROOT = WORKSPACE / "Image-Deepfake-Detectors-Public-Library"
LAUNCHER = DETECTORS_ROOT / "launcher.py"


def ensure_paths() -> None:
    if not DETECTORS_ROOT.exists():
        raise SystemExit(
            f"Detectors repo not found at {DETECTORS_ROOT}. Clone it first."
        )
    if not LAUNCHER.exists():
        raise SystemExit("launcher.py not found. Repo may be incomplete.")


def find_detectors_with_weights(detectors: List[str]) -> List[str]:
    present = []
    missing = []
    for det in detectors:
        weight = DETECTORS_ROOT / "detectors" / det / "checkpoint" / "pretrained" / "weights" / "best.pt"
        if weight.exists():
            present.append(det)
        else:
            missing.append((det, weight))
    if missing:
        print("Skipping detectors without weights:")
        for det, w in missing:
            print(f" - {det}: expected {w}")
    return present


def list_real_images() -> List[Path]:
    # Use workspace demo_images as real samples (small set)
    real_dir = WORKSPACE / "demo_images"
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    imgs: List[Path] = []
    for pat in exts:
        imgs.extend(real_dir.glob(pat))
    return imgs


def list_fake_images_by_quant() -> Dict[str, List[Path]]:
    # Collect generated images from output/img2img and output/txt2img
    out_root = WORKSPACE / "output"
    quants = {"fp16": [], "fp8": [], "fp4": []}
    for sub in ["img2img", "txt2img"]:
        d = out_root / sub
        if not d.exists():
            continue
        for q in quants.keys():
            quants[q].extend(sorted(d.glob(f"*_{q}_*.png")))
    return quants


def run_detection(detector: str, image: Path, py_exec: Path) -> Tuple[float, str]:
    # Calls: python launcher.py --detect --detector <det> --image <img> --weights pretrained --output <json>
    out_dir = WORKSPACE / "results" / "detections" / detector
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{image.stem}.json"

    cmd = [
        str(py_exec),
        str(LAUNCHER),
        "--detect",
        "--detector",
        detector,
        "--image",
        str(image),
        "--weights",
        "pretrained",
        "--output",
        str(out_json),
    ]
    # Use repository working directory to ensure relative paths resolve
    proc = subprocess.run(cmd, cwd=str(DETECTORS_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"Detection failed for {detector} on {image}")
    if not out_json.exists():
        raise FileNotFoundError(f"Detection output not found: {out_json}")

    data = json.loads(out_json.read_text(encoding="utf-8"))
    # Expect keys: prediction ('fake'/'real'), confidence in [0,1]
    conf = float(data.get("confidence", 0.0))
    pred = str(data.get("prediction", ""))
    return conf, pred


def compute_metrics(scores: List[float], labels: List[int]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    # AUC requires both classes
    if len(set(labels)) > 1:
        try:
            metrics["auc"] = float(roc_auc_score(labels, scores))
        except Exception:
            metrics["auc"] = float("nan")
    else:
        metrics["auc"] = float("nan")

    # Accuracy at threshold 0.5, interpreting score as P(fake)
    preds = [1 if s >= 0.5 else 0 for s in scores]
    try:
        metrics["accuracy@0.5"] = float(accuracy_score(labels, preds))
    except Exception:
        metrics["accuracy@0.5"] = float("nan")
    return metrics


def main():
    ensure_paths()

    # Use current venv python to keep deps consistent
    py_exec = Path(sys.executable)

    # Detectors to try
    all_detectors = ["CLIP-D", "NPR", "P2G", "R50_TF", "R50_nodown"]
    detectors = find_detectors_with_weights(all_detectors)
    if not detectors:
        print("No detectors with weights found.\n"
              "Please download weights and place them at: detectors/<DETECTOR>/checkpoint/pretrained/weights/best.pt\n"
              f"Repo root: {DETECTORS_ROOT}")
        return

    real_imgs = list_real_images()
    fake_by_quant = list_fake_images_by_quant()

    results_dir = WORKSPACE / "results"
    results_dir.mkdir(exist_ok=True)
    out_csv = results_dir / "sota_metrics.csv"

    rows: List[Dict[str, object]] = []

    for det in detectors:
        for quant, fake_imgs in fake_by_quant.items():
            if not fake_imgs:
                continue

            # Build evaluation set: all real + all fake for this quant
            eval_imgs = [(img, 0) for img in real_imgs] + [(img, 1) for img in fake_imgs]
            if not eval_imgs:
                continue

            scores: List[float] = []
            labels: List[int] = []

            print(f"Evaluating {det} on quant={quant}: {len(eval_imgs)} images...")
            for img, lab in eval_imgs:
                try:
                    conf, _ = run_detection(det, img, py_exec)
                except Exception as e:
                    print(f"  [warn] Skipping {img.name}: {e}")
                    continue
                scores.append(conf)
                labels.append(lab)

            if not scores:
                continue

            m = compute_metrics(scores, labels)
            row = {
                "detector": det,
                "quant": quant,
                "n_real": sum(1 for _ in real_imgs),
                "n_fake": sum(1 for _ in fake_imgs),
                "n_eval": len(scores),
                "auc": m.get("auc"),
                "accuracy@0.5": m.get("accuracy@0.5"),
            }
            rows.append(row)
            print(f"  -> AUC={row['auc']}, ACC@0.5={row['accuracy@0.5']}")

    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved metrics to {out_csv}")
    else:
        print("No results computed (likely no weights or no images found).")


if __name__ == "__main__":
    main()
