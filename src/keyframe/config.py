from copy import deepcopy
from pathlib import Path
from math import isfinite
from typing import Any
import yaml

DEFAULT = {
    'preprocessing': {'max_size': 256, 'roi': None, 'percentiles': [1, 99], 'denoise_sigma': .6, 'clahe': False},
    'contrast': {'reference': 'median', 'reference_frames': 5, 'smooth_sigma': 1.2, 'min_dynamic': .012, 'threshold_fraction': .45, 'min_interval_frames': 3, 'adaptive_alpha': .03},
    'vessel': {'sigmas': [1, 2, 3], 'black_ridges': True, 'response_threshold': .02, 'frangi_gamma': .03, 'candidate_only': True},
    'round2': {'optimized_sigmas': [1, 2, 3], 'sobel_threshold': .05, 'sobel_background_threshold': .05,
               'blackhat_size': 9, 'blackhat_shape': 'ellipse', 'blackhat_threshold': .03, 'score_smooth_sigma': .8,
               'filling_fallback': False, 'end_diastole': False, 'end_diastole_weight': .03},
    'sharpness': {'region': 'full'},
    'motion': {'max_corners': 80, 'roi': None},
    'selection': {'top_k': 3, 'min_frame_distance': 5, 'max_similarity': .995, 'preserve_intervals': True, 'min_vessel_score': .0001},
    'weights': {'vessel': .45, 'contrast': .3, 'sharpness': .2, 'motion': .05},
}

def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg = deepcopy(DEFAULT)
    if path:
        custom = yaml.safe_load(Path(path).read_text()) or {}
        for group, values in custom.items():
            if group not in cfg or not isinstance(values, dict) or set(values) - set(cfg[group]):
                raise ValueError(f'Unknown configuration group/key: {group}')
            cfg[group].update(values)
    p, c, v, s = (cfg[k] for k in ('preprocessing', 'contrast', 'vessel', 'selection'))
    if p['max_size'] < 16 or not 0 <= p['percentiles'][0] < p['percentiles'][1] <= 100:
        raise ValueError('Invalid preprocessing size/percentiles')
    if c['reference'] not in ('median', 'low_motion', 'adaptive') or c['reference_frames'] < 1:
        raise ValueError('Invalid background reference')
    if c['smooth_sigma'] < 0 or c['min_dynamic'] <= 0 or not 0 < c['threshold_fraction'] < 1 or c['min_interval_frames'] < 1 or not 0 < c['adaptive_alpha'] <= 1:
        raise ValueError('Invalid contrast parameters')
    if v['frangi_gamma'] <= 0:
        raise ValueError('frangi_gamma must be positive and fixed across frames')
    if not v['sigmas'] or min(v['sigmas']) <= 0 or v['response_threshold'] < 0:
        raise ValueError('Invalid vessel parameters')
    if s['top_k'] < 1 or s['min_frame_distance'] < 1 or not 0 <= s['max_similarity'] <= 1:
        raise ValueError('Invalid selection parameters')
    if cfg['sharpness']['region'] not in ('full', 'vessel') or cfg['motion']['max_corners'] < 1:
        raise ValueError('Invalid sharpness/motion parameters')
    round2 = cfg['round2']
    if not round2['optimized_sigmas'] or min(round2['optimized_sigmas']) <= 0:
        raise ValueError('Round 2 optimized sigmas must be positive')
    kernel_size = round2['blackhat_size']
    if (not isinstance(kernel_size, int) or kernel_size < 3 or kernel_size % 2 != 1
            or round2['blackhat_shape'] not in ('ellipse', 'rect', 'cross')):
        raise ValueError('Invalid Black-hat structure element')
    optional_mask = round2['sobel_background_threshold']
    if (optional_mask is not None and optional_mask < 0) or min(round2[k] for k in ('sobel_threshold', 'blackhat_threshold', 'score_smooth_sigma')) < 0:
        raise ValueError('Invalid round 2 response thresholds or smoothing')
    if (not all(isinstance(round2[key], bool) for key in ('filling_fallback', 'end_diastole'))
            or not isinstance(round2['end_diastole_weight'], (float, int))
            or not isfinite(round2['end_diastole_weight']) or not 0 <= round2['end_diastole_weight'] <= .1):
        raise ValueError('Invalid round 2 filling recovery or cardiac phase settings')
    if any(x < 0 for x in cfg['weights'].values()) or sum(cfg['weights'].values()) <= 0:
        raise ValueError('Weights must be nonnegative and not all zero')
    return cfg
