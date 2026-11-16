from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter1d

@dataclass
class ContrastResult:
    frame_scores: np.ndarray
    candidate_intervals: list[tuple[int, int]]
    confidence: float
    features: dict[str, np.ndarray]
    warnings: list[str]
    reference: np.ndarray


def normalize(values: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(values, [5, 95])
    return np.clip((values-lo)/max(hi-lo, 1e-8), 0, 1)


def detect_contrast(images: np.ndarray, cfg: dict) -> ContrastResult:
    n = len(images)
    flat = images.reshape(n, -1)
    hist = np.array([np.histogram(f, bins=32, range=(0, 1))[0]/len(f) for f in flat])
    count = min(cfg['reference_frames'], n)
    ref = np.median(images[:count], axis=0)
    if cfg['reference'] == 'low_motion' and count > 1:
        diffs = np.mean(np.abs(np.diff(images[:count], axis=0)), axis=(1, 2))
        ref = images[int(np.argmin(diffs))+1].copy()
    initial_reference = ref
    darkening, differences = [], []
    for image in images:
        delta = ref-image
        darkening.append(float(np.maximum(delta, 0).mean()))
        differences.append(float(np.abs(delta).mean()))
        if cfg['reference'] == 'adaptive':
            # Freeze background on substantial darkening; cannot disentangle motion/injection.
            if darkening[-1] < cfg['min_dynamic']:
                ref = (1-cfg['adaptive_alpha'])*ref + cfg['adaptive_alpha']*image
    raw = np.asarray(darkening)
    smooth = gaussian_filter1d(raw, cfg['smooth_sigma']) if cfg['smooth_sigma'] > 0 else raw
    dynamic = float(np.ptp(smooth))
    score = normalize(smooth)
    mask = score >= cfg['threshold_fraction'] if dynamic >= cfg['min_dynamic'] else np.zeros(n, bool)
    edges = np.diff(np.r_[False, mask, False].astype(int))
    intervals = [(int(a), int(b-1)) for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]) if b-a >= cfg['min_interval_frames']]
    warnings = ['Background darkening can include motion, catheter and exposure changes; confidence is heuristic.']
    if not intervals:
        warnings.append('No reliable filling interval: absent contrast, already-filled reference, or insufficient dynamics.')
    features = {'mean': flat.mean(1), 'std': flat.std(1), 'p10': np.quantile(flat, .1, axis=1),
                'p50': np.quantile(flat, .5, axis=1), 'p90': np.quantile(flat, .9, axis=1),
                'dark_fraction': (flat < .25).mean(1), 'reference_abs_difference': np.array(differences),
                'darkening_raw': raw, 'darkening_smoothed': smooth,
                'histogram_change': np.r_[0, np.abs(np.diff(hist, axis=0)).sum(1)]}
    features.update({f'hist_{i:02}': hist[:, i] for i in range(32)})
    return ContrastResult(score, intervals, float(min(1, dynamic/(3*cfg['min_dynamic']))) if intervals else 0., features, warnings, initial_reference)
