"""Float32 2-D Frangi with fixed gamma and explicit ridge polarity."""
from functools import lru_cache
from math import sqrt

import cv2
import numpy as np


@lru_cache(maxsize=32)
def gaussian_kernels(sigma: float) -> tuple[np.ndarray, np.ndarray]:
    # The reference applies two first derivatives at sigma/sqrt(2).
    # Trim tails below float32 relevance instead of keeping ~140 taps at sigma=1.
    width = sigma / sqrt(2)
    radius = int(8 * width + .5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    gaussian = np.exp(-.5 * (x / width)**2)
    gaussian /= gaussian.sum()
    # sepFilter2D correlates, so use the flipped first derivative.
    derivative = x * gaussian / width**2
    return gaussian.astype(np.float32), derivative.astype(np.float32)


def frangi_scale(image: np.ndarray, sigma: float, gamma: float) -> np.ndarray:
    gaussian, derivative = gaussian_kernels(sigma)

    def separable(source, x, y):
        return cv2.sepFilter2D(source, cv2.CV_32F, x, y, borderType=cv2.BORDER_REFLECT)

    dy = separable(image, gaussian, derivative)
    dx = separable(image, derivative, gaussian)
    yy = separable(dy, gaussian, derivative)
    xy = separable(dy, derivative, gaussian)
    xx = separable(dx, derivative, gaussian)
    center = (yy + xx) * .5
    spread = cv2.magnitude(xy, (yy - xx) * .5)
    high, low = center + spread, center - spread
    high_is_larger = np.abs(high) > np.abs(low)
    small = np.where(high_is_larger, low, high)
    large = np.where(high_is_larger, high, low)
    ratio = np.abs(small) / np.maximum(large, 1e-10)
    strength_squared = small**2 + large**2
    response = np.exp(-2 * ratio**2) * (1 - np.exp(-strength_squared / (2 * gamma**2)))
    # The clipped ratio alone can leak the opposite polarity when the minor eigenvalue is zero.
    response[large <= 0] = 0
    return response


def fast_frangi(image: np.ndarray, sigmas, gamma: float, black_ridges: bool,
                scale_cache: dict[float, np.ndarray] | None = None) -> np.ndarray:
    """Optional cache is scoped to one frame, polarity and gamma by the caller."""
    image = np.asarray(image, dtype=np.float32)
    source = image if black_ridges else -image
    response = np.zeros_like(image)
    for sigma in sigmas:
        if scale_cache is None:
            scale = frangi_scale(source, sigma, gamma)
        else:
            if sigma not in scale_cache:
                scale_cache[sigma] = frangi_scale(source, sigma, gamma)
            scale = scale_cache[sigma]
        np.maximum(response, scale, out=response)
    return response
