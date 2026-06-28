#!/usr/bin/env python3
"""
Main demo script — processes real test images and saves all 6 required outputs.

Meets project requirements:
  Stage 1 Enhance → 2 Segment → 3 Clean → 4 Detect → 5 Decide
  Per image: original, enhanced, mask, cleaned mask, detection, decision
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401 — adds src/ to sys.path

import cv2
import numpy as np
import pandas as pd

from lowlight_cv.config import DEFAULT_SEED, MIN_REAL_IMAGES, SHOWCASE_DIR
from lowlight_cv.data import load_real_test_images
from lowlight_cv.data.dataset import build_synthetic
from lowlight_cv.io.export import make_pipeline_strip
from lowlight_cv.metrics import enhancement_metrics
from lowlight_cv.pipeline import LowLightPipeline
from lowlight_cv.reporting import write_presentation, write_report_skeleton


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline showcase on real images")
    parser.add_argument("--n", type=int, default=6, help="number of test images")
    parser.add_argument("--enhance", default="auto")
    parser.add_argument("--segment", default="auto")
    parser.add_argument("--allow-synthetic", action="store_true", help="fallback if no real images")
    args = parser.parse_args()

    random.seed(DEFAULT_SEED)
    np.random.seed(DEFAULT_SEED)
    SHOWCASE_DIR.mkdir(parents=True, exist_ok=True)

    records, stats = load_real_test_images(args.n)
    using_synthetic = False
    if len(records) < MIN_REAL_IMAGES:
        if args.allow_synthetic:
            print(f"Warning: only {len(records)} real images; adding synthetic fallback.")
            synth = build_synthetic(max(0, args.n - len(records)))
            for i, (low, high, gt) in enumerate(synth):
                records.append(
                    {
                        "name": f"synth_{i:03d}",
                        "source": "synthetic",
                        "low": low,
                        "high": high,
                        "gt": gt,
                        "path": "",
                    }
                )
            using_synthetic = True
        else:
            print(
                f"ERROR: need at least {MIN_REAL_IMAGES} real images, got {len(records)}.\n"
                "Add photos to data/custom/ or ensure LOL download works.\n"
                "Use --allow-synthetic only for GT experiments, not for final demo."
            )
            sys.exit(1)

    pipe = LowLightPipeline(enhance=args.enhance, segment=args.segment)
    summary = []
    montage_results = []

    for rec in records[: args.n]:
        result = pipe.process(rec["low"])
        out_dir = SHOWCASE_DIR / rec["name"]
        pipe.export(result, out_dir, rec["name"])
        if len(montage_results) < 3:
            montage_results.append(result)

        row = {
            "name": rec["name"],
            "source": rec["source"],
            "enhancer": result["decision"]["enhancer"],
            "segmenter": result["decision"]["segmenter"],
            "count": result["decision"]["count"],
            "coverage": result["decision"]["coverage"],
            "load": result["decision"]["load"],
            "label": result["decision"]["label"],
            "status": result["decision"]["status"],
            "message": result["decision"]["message"],
        }
        if rec.get("high") is not None:
            row.update(enhancement_metrics(result["enhanced"], rec["high"]))
        summary.append(row)

    df = pd.DataFrame(summary)
    df.to_csv(SHOWCASE_DIR / "summary.csv", index=False)

    strips = [make_pipeline_strip(r, target_h=180) for r in montage_results]
    if strips:
        demo = cv2.vconcat(strips)
        cv2.imwrite(str(SHOWCASE_DIR / "demo_montage.png"), demo)

    pres = write_presentation(summary, SHOWCASE_DIR, stats)
    report = write_report_skeleton(summary, SHOWCASE_DIR, stats)

    print("=" * 60)
    print("Low-Light Enhancement & Detection — Showcase")
    print("=" * 60)
    print(f"Real images: LOL={stats['lol']} custom={stats['custom']} synthetic_fallback={using_synthetic}")
    print(f"Pipeline: enhance={args.enhance} segment={args.segment}")
    print(f"Outputs: {SHOWCASE_DIR}")
    print("")
    for row in summary:
        print(f"  [{row['name']}] {row['label']} | {row['count']} obj | {row['message']}")
    print("")
    print(f"Saved: {SHOWCASE_DIR / 'summary.csv'}")
    print(f"Saved: {SHOWCASE_DIR / 'demo_montage.png'}")
    print(f"Saved: {pres}")
    print(f"Saved: {report}")
    print("")
    print("Open any folder under outputs/showcase/ — 6 PNG files + pipeline strip per image.")


if __name__ == "__main__":
    main()
