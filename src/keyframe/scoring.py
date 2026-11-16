import numpy as np
from .algorithms.contrast import normalize

METHODS = ('contrast_only', 'frangi_only', 'sato_only', 'contrast_frangi', 'contrast_sato', 'multi_feature', 'contrast_frangi_motion', 'contrast_sato_motion')


def fuse(features: dict[str, np.ndarray], method: str, weights: dict) -> np.ndarray:
    if method not in METHODS:
        raise ValueError('Unsupported method')
    c = features['contrast']
    f, s = normalize(features['frangi_score']), normalize(features['sato_score'])
    q = (normalize(features['laplacian'])+normalize(features['gradient']))/2
    m = normalize(features['flow_displacement'])
    if method == 'contrast_only':
        return c.copy()
    if method in ('frangi_only', 'sato_only'):
        return f if method == 'frangi_only' else s
    if method == 'multi_feature':
        v = (f+s)/2
        return weights['vessel']*v+weights['contrast']*c+weights['sharpness']*q-weights['motion']*m
    v = f if 'frangi' in method else s
    score = weights['vessel']*v + weights['contrast']*c
    return score-weights['motion']*m if method.endswith('_motion') else score
