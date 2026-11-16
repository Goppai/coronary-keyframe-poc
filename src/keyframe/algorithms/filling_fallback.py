"""Local vessel evidence for filling interval recovery."""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .cheap_vessel import structure_element
from .contrast import normalize


def recover_filling(images, config, *, black_ridges):
    # Work at a bounded size and measure dark (or bright) ridges against their local neighborhood.
    height, width = images.shape[1:]
    scale = min(1., 192/max(height, width))
    size = (max(1, round(width*scale)), max(1, round(height*scale)))
    area = np.zeros(len(images))
    strength = np.zeros(len(images))
    for index, image in enumerate(images):
        frame = cv2.resize(image, size, interpolation=cv2.INTER_AREA) if scale < 1 else image
        margin = max(1, round(min(frame.shape)*.07))
        if min(frame.shape) <= 2 * margin:
            break
        operation = cv2.MORPH_BLACKHAT if black_ridges else cv2.MORPH_TOPHAT
        response = cv2.morphologyEx(frame, operation, structure_element(9, 'ellipse'), borderType=cv2.BORDER_REFLECT)
        response = response[margin:-margin, margin:-margin]
        mask = (response >= .08).astype(np.uint8)
        # Short isolated responses are excluded before measuring ridge coverage.
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= max(8, round(mask.size*.00035))
        keep[0] = False
        valid = keep[labels]
        area[index] = valid.mean()
        strength[index] = np.where(valid, response, 0).mean()
    sigma = config['smooth_sigma']
    smooth = gaussian_filter1d(strength, sigma, mode='nearest') if sigma else strength
    # Require absolute response and temporal variation so normalization cannot amplify noise alone.
    base, high = np.percentile(smooth, [20, 95])
    dynamic = high - base
    score = normalize(smooth)
    mask = ((smooth >= base + config['threshold_fraction']*dynamic) & (area >= .0015) &
            (smooth >= .00025) & (dynamic >= .00018) & (high >= 1.18*base))
    edges = np.diff(np.r_[False, mask, False].astype(int))
    intervals = [(int(a), int(b-1)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
                 if b-a >= config['min_interval_frames']]
    return intervals, score, {'fallback_ridge_area': area, 'fallback_ridge_strength': strength,
                              'fallback_ridge_smoothed': smooth, 'fallback_ridge_score': score}


def earlier_ridges_dominate(global_intervals, local_intervals, strength):
    """Require separate, earlier local intervals and a clearly higher local response."""
    if not global_intervals or not local_intervals or local_intervals[-1][1] >= global_intervals[0][0]:
        return False
    global_peak = max(strength[start:end+1].max() for start, end in global_intervals)
    local_peak = max(strength[start:end+1].max() for start, end in local_intervals)
    return local_peak > 1.5 * global_peak
