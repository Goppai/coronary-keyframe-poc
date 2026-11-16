import numpy as np
from .vessel import enhance

def compute(image: np.ndarray, cfg: dict) -> np.ndarray:
    return enhance(image, 'frangi', cfg)
