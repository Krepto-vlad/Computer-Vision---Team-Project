import cv2
import numpy as np


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def threshold_global(img, t=110):
    return cv2.threshold(to_gray(img), t, 255, cv2.THRESH_BINARY)[1]


def threshold_otsu(img):
    return cv2.threshold(to_gray(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def threshold_adaptive(img, block=21, c=4):
    return cv2.adaptiveThreshold(
        to_gray(img),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        c,
    )


def watershed_segment(img):
    gray = to_gray(img)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    sure_fg = cv2.threshold(dist, 0.45 * dist.max(), 255, 0)[1].astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    vis = img.copy()
    vis[markers == -1] = (0, 0, 255)
    return vis, markers


def grabcut_segment(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    rect = (int(w * 0.08), int(h * 0.08), int(w * 0.84), int(h * 0.84))
    cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)


def connected_components(binary):
    n, labels = cv2.connectedComponents(binary)
    hue = (labels * 179 / max(n - 1, 1)).astype(np.uint8)
    hsv = cv2.merge([hue, np.full_like(hue, 255), np.where(labels > 0, 255, 0).astype(np.uint8)])
    return n - 1, cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def hsv_segment(img, s_thresh=50, v_thresh=80):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mask = np.zeros(s.shape, np.uint8)
    mask[(s > s_thresh) & (v > v_thresh)] = 255
    return mask


def canny_segment(img, low=50, high=150, dilate_k=3):
    gray = to_gray(img)
    edges = cv2.Canny(gray, low, high)
    kernel = np.ones((dilate_k, dilate_k), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)
    return mask


SEGMENTERS = {
    "global": threshold_global,
    "otsu": threshold_otsu,
    "adaptive": threshold_adaptive,
    "grabcut": grabcut_segment,
    "hsv": hsv_segment,
    "canny": canny_segment,
}
