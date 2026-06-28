import cv2


def _kernel(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def morph(mask, op, k=5, it=1):
    ops = {
        "erode": cv2.MORPH_ERODE,
        "dilate": cv2.MORPH_DILATE,
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
        "gradient": cv2.MORPH_GRADIENT,
        "tophat": cv2.MORPH_TOPHAT,
        "blackhat": cv2.MORPH_BLACKHAT,
    }
    return cv2.morphologyEx(mask, ops[op], _kernel(k), iterations=it)


def clean_mask(mask, k=5):
    m = morph(mask, "open", k, 1)
    m = morph(m, "close", k, 2)
    return m
