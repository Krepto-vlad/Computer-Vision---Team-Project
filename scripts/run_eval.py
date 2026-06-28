#!/usr/bin/env python3
"""Run the pipeline over all samples and report GT-based segmentation/detection metrics."""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from lowlight_cv.config import DEFAULT_SEED, OUTPUT_DIR
from lowlight_cv.data import load_samples
from lowlight_cv.data.dataset import build_synthetic
from lowlight_cv.evaluation import detection_prf, gt_boxes_from_mask, mask_metrics
from lowlight_cv.metrics import enhancement_metrics
from lowlight_cv.pipeline import LowLightPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enhance", default="auto")
    parser.add_argument("--segment", default="auto")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--synthetic", action="store_true", help="force synthetic dataset with GT masks")
    args = parser.parse_args()

    random.seed(DEFAULT_SEED)
    np.random.seed(DEFAULT_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        samples = build_synthetic(args.n)
        is_synthetic = True
    else:
        samples, is_synthetic = load_samples(args.n)
    pipe = LowLightPipeline(enhance=args.enhance, segment=args.segment)

    records = []
    for i, (low, high, gt) in enumerate(samples):
        result = pipe.process(low)
        row = {"sample": i, "enhancer": result["decision"]["enhancer"], **enhancement_metrics(result["enhanced"], high)}
        if gt is not None:
            row.update(mask_metrics(result["clean"], gt))
            row.update(detection_prf(result["boxes"], gt_boxes_from_mask(gt)))
        records.append(row)

    df = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / "eval_results.csv"
    df.to_csv(csv_path, index=False)

    print(f"Dataset: {'synthetic' if is_synthetic else 'LOL'} | samples: {len(samples)}")
    if "iou" in df.columns:
        print(f"mean IoU={df['iou'].mean():.3f} Dice={df['dice'].mean():.3f} F1={df['f1'].mean():.3f}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
