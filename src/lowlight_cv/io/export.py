import json
from pathlib import Path

import cv2
import numpy as np


def render_decision_card(decision, width=640, height=200):
    """Render final decision as an interpretable image (required output #6)."""
    card = np.full((height, width, 3), 32, np.uint8)
    lines = [
        "FINAL DECISION",
        f"Status: {decision['status'].upper()}  |  Label: {decision['label']}",
        decision["message"],
        f"Objects: {decision['count']}  |  Coverage: {decision['coverage']:.1%}  |  Load: {decision['load']}",
        f"Enhancer: {decision.get('enhancer', 'n/a')}  |  Segmenter: {decision.get('segmenter', 'n/a')}",
    ]
    y = 36
    for i, line in enumerate(lines):
        scale = 0.65 if i == 0 else 0.52
        color = (0, 220, 255) if i == 0 else (230, 230, 230)
        if i == 1 and decision.get("alert"):
            color = (0, 80, 255)
        cv2.putText(card, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += 34
    return card


def make_pipeline_strip(result, target_h=240):
    """Horizontal strip: original → enhanced → segment → clean → detect → decision."""
    decision_img = render_decision_card(result["decision"], width=420, height=target_h)
    stages = [
        ("1. Original", result["input"]),
        ("2. Enhanced", result["enhanced"]),
        ("3. Segment", cv2.cvtColor(result["mask"], cv2.COLOR_GRAY2BGR)),
        ("4. Clean", cv2.cvtColor(result["clean"], cv2.COLOR_GRAY2BGR)),
        ("5. Detect", result["detection"]),
        ("6. Decide", decision_img),
    ]
    tiles = []
    for title, img in stages:
        h, w = img.shape[:2]
        scale = target_h / h
        resized = cv2.resize(img, (max(1, int(w * scale)), target_h))
        bar = np.full((28, resized.shape[1], 3), 48, np.uint8)
        cv2.putText(bar, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, resized]))
    return cv2.hconcat(tiles)


def save_required_outputs(result, out_dir: Path, stem: str):
    """
    Save all 6 mandatory outputs per test image + JSON + pipeline strip.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "01_original": result["input"],
        "02_enhanced": result["enhanced"],
        "03_segmentation_mask": result["mask"],
        "04_cleaned_mask": result["clean"],
        "05_detection": result["detection"],
    }
    saved = {}
    for name, img in outputs.items():
        path = out_dir / f"{stem}_{name}.png"
        cv2.imwrite(str(path), img)
        saved[name] = str(path)

    decision_img = render_decision_card(result["decision"])
    decision_path = out_dir / f"{stem}_06_decision.png"
    cv2.imwrite(str(decision_path), decision_img)
    saved["06_decision"] = str(decision_path)

    json_path = out_dir / f"{stem}_decision.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result["decision"], f, indent=2, ensure_ascii=False)
    saved["decision_json"] = str(json_path)

    strip_path = out_dir / f"{stem}_pipeline_strip.png"
    cv2.imwrite(str(strip_path), make_pipeline_strip(result))
    saved["pipeline_strip"] = str(strip_path)

    return saved
