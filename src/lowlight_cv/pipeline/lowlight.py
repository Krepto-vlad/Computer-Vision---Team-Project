import cv2
import numpy as np

from lowlight_cv.config import DETECT_MIN_AREA_RATIO
from lowlight_cv.decision.scene import decide
from lowlight_cv.detection.features import detect_objects, draw_boxes
from lowlight_cv.enhancement.methods import ENHANCERS
from lowlight_cv.io.export import save_required_outputs
from lowlight_cv.metrics.enhancement import quality_score
from lowlight_cv.morphology.ops import clean_mask
from lowlight_cv.segmentation.methods import SEGMENTERS, threshold_otsu
from lowlight_cv.utils.viz import show


def _segment_score(mask, img_shape):
    """Prefer compact foreground masks — reject huge Otsu blobs on real photos."""
    h, w = img_shape[:2]
    fg = (mask > 127).astype(np.uint8)
    ratio = fg.sum() / (h * w * 255.0)
    if ratio < 0.008 or ratio > 0.42:
        return -1e9
    n, _ = cv2.connectedComponents(fg)
    n = max(n - 1, 0)
    if n == 0 or n > 30:
        return -5e8 if n == 0 else 40 - (n - 30) * 3
    target_ratio, target_n = 0.14, 5
    score = 100.0
    score -= abs(ratio - target_ratio) * 200
    score -= abs(np.log1p(n) - np.log1p(target_n)) * 14
    if ratio > 0.28:
        score -= (ratio - 0.28) * 350
    return score


class LowLightPipeline:
    """Enhancement → Segmentation → Morphology → Detection → Decision."""

    STAGES = ("enhance", "segment", "clean", "detect", "decide")

    def __init__(self, enhance="auto", segment="auto"):
        self.enhance = enhance
        self.segment = segment

    def _enhance(self, img):
        if self.enhance == "auto":
            best, best_s, name = img, -1e9, "none"
            for k, fn in ENHANCERS.items():
                out = fn(img)
                s = quality_score(out)
                if s > best_s:
                    best, best_s, name = out, s, k
            return best, name
        return ENHANCERS[self.enhance](img), self.enhance

    def _segment(self, img):
        if self.segment == "auto":
            best, best_s, name = None, -1e9, "otsu"
            for k, fn in SEGMENTERS.items():
                mask = fn(img)
                s = _segment_score(mask, img.shape)
                if s > best_s:
                    best, best_s, name = mask, s, k
            if best is None:
                best, name = threshold_otsu(img), "otsu"
            return best, name
        fn = SEGMENTERS.get(self.segment, threshold_otsu)
        return fn(img), self.segment

    def process(self, img):
        enhanced, enh_name = self._enhance(img)
        mask, seg_name = self._segment(enhanced)
        clean = clean_mask(mask)
        boxes = detect_objects(clean, min_area_ratio=DETECT_MIN_AREA_RATIO)
        decision = decide(boxes, img.shape, enhancer=enh_name, segmenter=seg_name)
        return {
            "input": img,
            "enhanced": enhanced,
            "mask": mask,
            "clean": clean,
            "boxes": boxes,
            "detection": draw_boxes(enhanced, boxes),
            "decision": decision,
        }

    def export(self, result, out_dir, stem):
        return save_required_outputs(result, out_dir, stem)

    def visualize(self, result):
        imgs = [
            result["input"],
            result["enhanced"],
            cv2.cvtColor(result["mask"], cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(result["clean"], cv2.COLOR_GRAY2BGR),
            result["detection"],
        ]
        titles = [
            "1. Original",
            f"2. Enhanced [{result['decision']['enhancer']}]",
            f"3. Segment [{result['decision']['segmenter']}]",
            "4. Clean",
            f"5. Detect → {result['decision']['label']}",
        ]
        show(imgs, titles, cols=5, size=3)
