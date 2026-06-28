import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _entropy(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-9)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _noise_level(gray):
    den = cv2.medianBlur(gray, 3)
    return float((gray.astype(np.float32) - den.astype(np.float32)).std())


def enhancement_metrics(enhanced, reference=None):
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    m = {
        "brightness": round(float(gray.mean()), 2),
        "contrast": round(float(gray.std()), 2),
        "entropy": round(_entropy(gray), 3),
        "edge_density": round(float(edges.mean()), 3),
        "noise": round(_noise_level(gray), 3),
    }
    if reference is not None:
        ref = cv2.resize(reference, (enhanced.shape[1], enhanced.shape[0]))
        m["psnr"] = round(float(peak_signal_noise_ratio(ref, enhanced)), 2)
        m["ssim"] = round(float(structural_similarity(ref, enhanced, channel_axis=2)), 3)
    return m


def quality_score(img):
    m = enhancement_metrics(img)
    return (
        m["contrast"] * 0.4
        + m["entropy"] * 8.0
        + m["edge_density"] * 1.5
        - m["noise"] * 1.2
    )
