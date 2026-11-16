"""OpenCV vascular visibility proxies on a normalized 2-D grayscale frame."""
from functools import lru_cache

import cv2
import numpy as np

from ..preprocessing import crop
from .vessel import vessel_features


@lru_cache(maxsize=32)
def structure_element(size: int, shape: str) -> np.ndarray:
    shapes = {'ellipse': cv2.MORPH_ELLIPSE, 'rect': cv2.MORPH_RECT, 'cross': cv2.MORPH_CROSS}
    if size < 3 or size % 2 != 1 or shape not in shapes:
        raise ValueError('Black-hat needs an odd kernel >= 3 and ellipse, rect, or cross shape')
    return cv2.getStructuringElement(shapes[shape], (size, size))


def sobel_vessel_score(image: np.ndarray, *, roi=None, background: np.ndarray | None = None,
                       background_threshold: float = .05, threshold: float = .05,
                       black_ridges: bool = True) -> tuple[np.ndarray, dict[str, float]]:
    frame = np.asarray(crop(image, roi), dtype=np.float32)
    gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3, scale=.25)
    gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3, scale=.25)
    response = cv2.magnitude(gx, gy)
    if background is not None:
        if background.shape != image.shape:
            raise ValueError('Sobel background shape must match frame')
        difference = crop(background, roi) - frame
        foreground = difference > background_threshold if black_ridges else -difference > background_threshold
        # One-pixel dilation includes the boundary of newly enhanced structures.
        mask = cv2.dilate(foreground.astype(np.uint8), structure_element(3, 'rect'))
        response *= mask
    return response, vessel_features(response, threshold)


def blackhat_vessel_score(image: np.ndarray, *, roi=None, size: int = 9, shape: str = 'ellipse',
                          threshold: float = .03, black_ridges: bool = True) -> tuple[np.ndarray, dict[str, float]]:
    frame = np.asarray(crop(image, roi), dtype=np.float32)
    operation = cv2.MORPH_BLACKHAT if black_ridges else cv2.MORPH_TOPHAT
    response = cv2.morphologyEx(frame, operation, structure_element(size, shape), borderType=cv2.BORDER_REFLECT)
    return response, vessel_features(response, threshold)
