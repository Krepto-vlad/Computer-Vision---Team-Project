import glob
import os
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np

from lowlight_cv.config import CUSTOM_DATA_DIR, DATA_DIR, DEFAULT_SYNTHETIC_SAMPLES, IMAGE_EXTENSIONS


def try_download_lol(data_dir: Path | str = DATA_DIR):
    """Download and extract LOL dataset; return list of (low_path, high_path) or None."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        import gdown
    except ImportError:
        return None

    target = data_dir / "lol.zip"
    file_id = "157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB"
    try:
        gdown.download(id=file_id, output=str(target), quiet=True)
        if target.exists() and target.stat().st_size > 10_000_000:
            with zipfile.ZipFile(target) as z:
                z.extractall(data_dir)
            low = sorted(glob.glob(str(data_dir / "**" / "low" / "*.png"), recursive=True))
            high = sorted(glob.glob(str(data_dir / "**" / "high" / "*.png"), recursive=True))
            if low and high:
                return list(zip(low, high))
    except Exception as e:
        print("LOL download failed:", e)
    return None


def make_synthetic_scene(h=256, w=256, n_objects=5):
    """Generate a synthetic scene with random shapes and a binary GT mask."""
    normal = np.full((h, w, 3), 55, np.float32)
    normal += np.random.randn(h, w, 3) * 6
    normal = normal.clip(0, 255).astype(np.uint8)
    mask = np.zeros((h, w), np.uint8)
    palette = [
        (60, 180, 230),
        (120, 220, 120),
        (230, 120, 120),
        (200, 200, 60),
        (180, 120, 220),
        (90, 200, 220),
    ]
    for _ in range(n_objects):
        color = random.choice(palette)
        shape = random.choice(["circle", "rect", "tri"])
        cx, cy = random.randint(40, w - 40), random.randint(40, h - 40)
        r = random.randint(18, 40)
        if shape == "circle":
            cv2.circle(normal, (cx, cy), r, color, -1)
            cv2.circle(mask, (cx, cy), r, 255, -1)
        elif shape == "rect":
            cv2.rectangle(normal, (cx - r, cy - r), (cx + r, cy + r), color, -1)
            cv2.rectangle(mask, (cx - r, cy - r), (cx + r, cy + r), 255, -1)
        else:
            pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]])
            cv2.fillPoly(normal, [pts], color)
            cv2.fillPoly(mask, [pts], 255)
    normal = cv2.GaussianBlur(normal, (3, 3), 0)
    return normal, mask


def darken(normal, gamma=2.6, gain=0.35, noise=8):
    """Simulate low-light capture from a normal-lit image."""
    f = (normal.astype(np.float32) / 255.0) ** gamma * gain
    low = f * 255.0 + np.random.randn(*normal.shape) * noise
    return low.clip(0, 255).astype(np.uint8)


def build_synthetic(n=DEFAULT_SYNTHETIC_SAMPLES, data_dir: Path | str = DATA_DIR):
    """Build synthetic low/high/mask triplets and save PNGs to data_dir."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for i in range(n):
        normal, mask = make_synthetic_scene(n_objects=random.randint(3, 7))
        low = darken(normal)
        cv2.imwrite(str(data_dir / f"low_{i:02d}.png"), low)
        cv2.imwrite(str(data_dir / f"high_{i:02d}.png"), normal)
        cv2.imwrite(str(data_dir / f"mask_{i:02d}.png"), mask)
        pairs.append((low, normal, mask))
    return pairs


def load_custom_images(custom_dir: Path | str = CUSTOM_DATA_DIR):
    """Load user-provided real low-light images from data/custom/."""
    custom_dir = Path(custom_dir)
    custom_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(custom_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        records.append(
            {
                "name": path.stem,
                "source": "custom",
                "low": img,
                "high": None,
                "gt": None,
                "path": str(path),
            }
        )
    return records


def load_real_test_images(n=6, data_dir: Path | str = DATA_DIR, custom_dir: Path | str = CUSTOM_DATA_DIR):
    """
    Load real test images for demo/report.

    Priority: LOL low-light pairs → custom photos.
    Returns (records, stats) where stats describes sources used.
    """
    data_dir = Path(data_dir)
    records = []
    stats = {"lol": 0, "custom": 0}

    lol = try_download_lol(data_dir)
    if lol:
        for i, (lp, hp) in enumerate(lol):
            if len(records) >= n:
                break
            low = cv2.imread(lp)
            if low is None:
                continue
            records.append(
                {
                    "name": f"lol_{i:03d}",
                    "source": "LOL",
                    "low": low,
                    "high": cv2.imread(hp),
                    "gt": None,
                    "path": lp,
                }
            )
            stats["lol"] += 1

    for rec in load_custom_images(custom_dir):
        if len(records) >= n:
            break
        records.append(rec)
        stats["custom"] += 1

    return records[:n], stats


def load_samples(n=DEFAULT_SYNTHETIC_SAMPLES, data_dir: Path | str = DATA_DIR):
    """
    Load LOL or synthetic samples.

    Returns (samples, is_synthetic) where each sample is (low_bgr, high_bgr, mask_or_none).
    """
    lol = try_download_lol(data_dir)
    if lol:
        samples = []
        for lp, hp in lol[:n]:
            samples.append((cv2.imread(lp), cv2.imread(hp), None))
        return samples, False

    print("Using synthetic dataset")
    return build_synthetic(n, data_dir), True
