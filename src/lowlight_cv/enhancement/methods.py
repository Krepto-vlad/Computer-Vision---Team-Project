import math

import cv2
import numpy as np


def gamma_correction(img, gamma=0.5):
    inv = 1.0 / gamma
    table = ((np.arange(256) / 255.0) ** inv * 255).astype(np.uint8)
    return cv2.LUT(img, table)


def adaptive_gamma(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = gray.mean() / 255.0
    gamma = float(np.clip(math.log(0.5) / math.log(mean + 1e-6), 0.35, 1.0))
    return gamma_correction(img, gamma)


def hist_equalization(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def clahe_enhance(img, clip=3.0, grid=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _norm_channel(ch):
    lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
    ch = np.clip((ch - lo) / (hi - lo + 1e-6), 0, 1)
    return (ch * 255).astype(np.uint8)


def single_scale_retinex(img, sigma=80):
    f = img.astype(np.float32) + 1.0
    r = np.log(f) - np.log(cv2.GaussianBlur(f, (0, 0), sigma) + 1.0)
    out = np.zeros_like(img)
    for c in range(3):
        out[:, :, c] = _norm_channel(r[:, :, c])
    return out


def multi_scale_retinex(img, sigmas=(15, 80, 250)):
    f = img.astype(np.float32) + 1.0
    r = np.zeros_like(f)
    for s in sigmas:
        r += np.log(f) - np.log(cv2.GaussianBlur(f, (0, 0), s) + 1.0)
    r /= len(sigmas)
    out = np.zeros_like(img)
    for c in range(3):
        out[:, :, c] = _norm_channel(r[:, :, c])
    return out


def msrcr(img, sigmas=(15, 80, 250), alpha=125.0, beta=46.0):
    f = img.astype(np.float32) + 1.0
    r = np.zeros_like(f)
    for s in sigmas:
        r += np.log(f) - np.log(cv2.GaussianBlur(f, (0, 0), s) + 1.0)
    r /= len(sigmas)
    crf = beta * (np.log(alpha * f) - np.log(f.sum(axis=2, keepdims=True) + 1.0))
    r = r * crf
    out = np.zeros_like(img)
    for c in range(3):
        out[:, :, c] = _norm_channel(r[:, :, c])
    return out


ENHANCERS = {
    "gamma": lambda x: gamma_correction(x, 0.45),
    "adaptive_gamma": adaptive_gamma,
    "hist_eq": hist_equalization,
    "clahe": clahe_enhance,
    "ssr": single_scale_retinex,
    "msr": multi_scale_retinex,
    "msrcr": msrcr,
}
