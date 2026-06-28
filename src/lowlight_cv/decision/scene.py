from lowlight_cv.config import (
    ALERT_COVERAGE,
    ALERT_COVERAGE_WITH_COUNT,
    ALERT_MIN_OBJECTS,
    ALERT_OBJECTS_WITH_COVERAGE,
)


def _is_alert(count, coverage):
    """ALERT only for extreme overload — normal detections stay OK."""
    if count >= ALERT_MIN_OBJECTS:
        return True
    if coverage >= ALERT_COVERAGE:
        return True
    if count >= ALERT_OBJECTS_WITH_COVERAGE and coverage >= ALERT_COVERAGE_WITH_COUNT:
        return True
    return False


def decide(boxes, shape, enhancer=None, segmenter=None):
    h, w = shape[:2]
    coverage = sum(b["area"] for b in boxes) / (h * w)
    count = len(boxes)
    status = "objects_detected" if count > 0 else "empty"
    load = "high" if coverage > 0.45 else "medium" if coverage > 0.12 else "low"

    if status == "empty":
        message = "Empty scene: no target regions after enhancement and segmentation."
        label = "NO_OBJECTS"
        alert = False
    elif _is_alert(count, coverage):
        message = f"Alert: {count} regions cover {coverage:.1%} of the frame (possible overload)."
        label = "ALERT"
        alert = True
    elif load == "medium":
        message = f"Scene OK: {count} region(s) detected ({coverage:.1%} coverage)."
        label = "OK"
        alert = False
    else:
        message = f"Scene OK: {count} compact region(s) ({coverage:.1%} coverage)."
        label = "OK"
        alert = False

    out = {
        "count": count,
        "coverage": round(float(coverage), 3),
        "status": status,
        "load": load,
        "label": label,
        "alert": alert,
        "message": message,
    }
    if enhancer is not None:
        out["enhancer"] = enhancer
    if segmenter is not None:
        out["segmenter"] = segmenter
    return out
