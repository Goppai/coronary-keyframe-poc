import numpy as np
from skimage.filters import frangi, sato


def enhance(image: np.ndarray, method: str, cfg: dict) -> np.ndarray:
    if method not in ('frangi', 'sato'):
        raise ValueError('Unknown vessel filter')
    fn = frangi if method == 'frangi' else sato
    options = {'gamma': cfg['frangi_gamma']} if method == 'frangi' else {}
    return fn(image, sigmas=cfg['sigmas'], black_ridges=cfg['black_ridges'], **options).astype(np.float32, copy=False)


def vessel_features(response: np.ndarray, threshold: float) -> dict[str, float]:
    area = float((response > threshold).mean())
    strength = float(response.mean())
    return {'area': area, 'strength': strength, 'score': area*strength}
