import math

import cv2
import matplotlib.pyplot as plt


def show(images, titles=None, cols=4, size=3.2, gray=False):
    """Display a grid of OpenCV BGR or grayscale images."""
    n = len(images)
    rows = math.ceil(n / cols)
    plt.figure(figsize=(cols * size, rows * size))
    for i, im in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if im.ndim == 2 or gray:
            plt.imshow(im, cmap="gray")
        else:
            plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        if titles:
            plt.title(titles[i], fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
