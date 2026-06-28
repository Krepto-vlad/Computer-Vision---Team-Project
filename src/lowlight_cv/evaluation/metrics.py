import cv2
import numpy as np


def _binarize(mask):
    return (mask > 127).astype(np.uint8)


def iou(pred, gt):
    p, g = _binarize(pred), _binarize(gt)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter / union) if union else 1.0


def dice(pred, gt):
    p, g = _binarize(pred), _binarize(gt)
    total = p.sum() + g.sum()
    return float(2 * np.logical_and(p, g).sum() / total) if total else 1.0


def pixel_accuracy(pred, gt):
    return float((_binarize(pred) == _binarize(gt)).mean())


def mask_metrics(pred, gt):
    return {
        "iou": round(iou(pred, gt), 3),
        "dice": round(dice(pred, gt), 3),
        "pixel_acc": round(pixel_accuracy(pred, gt), 3),
    }


def _bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def gt_boxes_from_mask(mask, min_area_ratio=0.002):
    n, _, stats, _ = cv2.connectedComponentsWithStats(_binarize(mask) * 255, 8)
    h, w = mask.shape[:2]
    min_area = min_area_ratio * h * w
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area >= min_area:
            boxes.append((int(x), int(y), int(bw), int(bh)))
    return boxes


def detection_prf(pred_boxes, gt_boxes, iou_thr=0.5):
    preds = [b["bbox"] if isinstance(b, dict) else b for b in pred_boxes]
    matched, tp = set(), 0
    for p in preds:
        best, best_iou = -1, iou_thr
        for j, g in enumerate(gt_boxes):
            if j in matched:
                continue
            v = _bbox_iou(p, g)
            if v >= best_iou:
                best, best_iou = j, v
        if best >= 0:
            matched.add(best)
            tp += 1
    fp, fn = len(preds) - tp, len(gt_boxes) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
    }
