import cv2
import numpy as np


def sharpness(image: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    smooth = cv2.GaussianBlur(image, (0, 0), .7)
    lap = cv2.Laplacian(smooth, cv2.CV_32F)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    if mask is None:
        mask = np.ones_like(image, dtype=bool)
    if not mask.any():
        return {'laplacian': 0., 'gradient': 0.}
    return {'laplacian': float(lap[mask].var()), 'gradient': float((gx*gx+gy*gy)[mask].mean())}
