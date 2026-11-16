from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class Prepared:
    analysis: np.ndarray
    display: np.ndarray
    limits: tuple[float, float]


def crop(image: np.ndarray, roi: list[int] | None) -> np.ndarray:
    if roi is None:
        return image
    if len(roi) != 4 or any(int(x) != x for x in roi):
        raise ValueError('ROI must have four integer coordinates')
    x, y, w, h = map(int, roi)
    if x < 0 or y < 0 or w < 1 or h < 1 or x+w > image.shape[-1] or y+h > image.shape[-2]:
        raise ValueError('ROI outside image')
    return image[..., y:y+h, x:x+w]


def preprocess(pixels: np.ndarray, cfg: dict) -> Prepared:
    if pixels.ndim != 3 or not pixels.size or not np.isfinite(pixels).all():
        raise ValueError('Expected finite T,H,W pixels')
    images = crop(pixels, cfg['roi'])
    # One sequence-wide transform. Bounded spatial/temporal sampling, no framewise windowing.
    sample = images[::max(1, len(images)//32), ::4, ::4]
    lo, hi = np.percentile(sample, cfg['percentiles']).astype(float)
    normalized = np.clip((images-lo)/max(hi-lo, 1e-6), 0, 1).astype(np.float32)
    display = (normalized*255).astype(np.uint8)
    scale = min(1., cfg['max_size']/max(images.shape[1:]))
    size = (max(1, round(images.shape[2]*scale)), max(1, round(images.shape[1]*scale)))
    result = []
    for image in normalized:
        frame = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        if cfg['denoise_sigma'] > 0:
            frame = cv2.GaussianBlur(frame, (0, 0), cfg['denoise_sigma'])
        if cfg['clahe']:
            frame = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply((frame*255).astype(np.uint8))/255.
        result.append(frame.astype(np.float32))
    return Prepared(np.stack(result), display, (lo, hi))
