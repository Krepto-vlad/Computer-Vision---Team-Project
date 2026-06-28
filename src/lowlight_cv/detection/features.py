import cv2
import numpy as np

from lowlight_cv.segmentation.methods import to_gray


def harris(img):
    gray = np.float32(to_gray(img))
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    vis = img.copy()
    vis[dst > 0.01 * dst.max()] = (0, 0, 255)
    return vis, int((dst > 0.01 * dst.max()).sum())


def shi_tomasi(img):
    gray = to_gray(img)
    corners = cv2.goodFeaturesToTrack(gray, 200, 0.01, 10)
    vis = img.copy()
    if corners is not None:
        for c in corners.astype(int).reshape(-1, 2):
            cv2.circle(vis, tuple(c), 3, (0, 255, 0), -1)
    return vis, 0 if corners is None else len(corners)


def fast_detect(img):
    gray = to_gray(img)
    kp = cv2.FastFeatureDetector_create().detect(gray, None)
    return cv2.drawKeypoints(img, kp, None, color=(255, 0, 0)), len(kp)


def orb_detect(img):
    gray = to_gray(img)
    kp, _ = cv2.ORB_create(500).detectAndCompute(gray, None)
    return cv2.drawKeypoints(img, kp, None, color=(0, 255, 255)), len(kp)


def detect_objects(mask, min_area_ratio=0.002):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]
    min_area = min_area_ratio * h * w
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            x, y, bw, bh = cv2.boundingRect(c)
            boxes.append({"bbox": (x, y, bw, bh), "area": float(area)})
    return boxes


def draw_boxes(img, boxes):
    vis = img.copy()
    for i, b in enumerate(boxes):
        x, y, w, h = b["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, str(i + 1), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return vis
