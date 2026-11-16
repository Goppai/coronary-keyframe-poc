"""Experimental image-based end-diastolic references and low-motion markers."""
import numpy as np


IMAGE_PHASE_SCORE_MAX = .5


def image_phase_references(times, strain, intervals):
    """Require a repeated transition from expansion to contraction in one filling interval."""
    candidates = []
    for start, end in intervals:
        turns = []
        for i in range(start+2, end-1):
            before, after = strain[i-1:i+1], strain[i+1:i+3]
            if (np.all(before > 0) and np.all(after < 0)
                    and before.mean() >= .006 and after.mean() <= -.006
                    and times[i+2]-times[i-2] <= 350):
                turns.append(i)
        gaps = np.diff(times[turns])
        if gaps.size and np.all((gaps >= 350) & (gaps <= 1500)) and gaps.max()/gaps.min() <= 1.35:
            candidates.extend(turns)
    return candidates


def phase_evidence(frame_times, minima, intervals, windows, strain):
    """Return image-based phase estimates and periodic low-motion markers."""
    n = len(minima)
    score = np.full(n, np.nan)
    nearest_ms = np.full(n, np.nan)
    references = np.zeros(n, dtype=bool)
    low_motion = np.zeros(n, dtype=bool)
    for a, b in intervals:
        if sum(a <= start <= b for start, _ in windows) > 1:
            low_motion[a:b+1] = minima[a:b+1]
    source = 'image_low_motion' if low_motion.any() else 'unavailable'
    times = np.asarray(frame_times, dtype=float) if frame_times is not None else np.array([])
    valid_times = len(times) == n and n > 1 and np.isfinite(times).all() and np.all(np.diff(times) > 0)
    if valid_times:
        image_frames = image_phase_references(times, strain, intervals)
        if image_frames:
            references[image_frames] = True
            nearest_ms = np.abs(times[:, None] - times[image_frames]).min(axis=1)
            score = IMAGE_PHASE_SCORE_MAX * np.clip(1 - nearest_ms / np.median(np.diff(times)), 0, 1)
            source = 'image_geometry_estimate'
    return {'source': source, 'reference_frames': np.flatnonzero(references).tolist(),
            'low_motion_frames': np.flatnonzero(low_motion).tolist()}, {
                'end_diastole_score': score, 'end_diastole_distance_ms': nearest_ms,
                'end_diastole_reference': references, 'periodic_low_motion': low_motion}
