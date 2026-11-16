import cv2
import numpy as np
from ..preprocessing import crop
from .cheap_vessel import structure_element


def motion_features(images: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    images = crop(images, cfg['roi'])
    diff, flow, tracked = np.zeros(len(images)), np.zeros(len(images)), np.zeros(len(images), dtype=int)
    for t in range(1, len(images)):
        diff[t] = np.abs(images[t]-images[t-1]).mean()
        a, b = [(x*255).astype(np.uint8) for x in images[t-1:t+1]]
        points = cv2.goodFeaturesToTrack(a, maxCorners=cfg['max_corners'], qualityLevel=.02, minDistance=5)
        if points is None:
            continue
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None)
        if nxt is None:
            continue
        back, valid, _ = cv2.calcOpticalFlowPyrLK(b, a, nxt, None)
        if back is None:
            continue
        mask = (status.ravel() == 1) & (valid.ravel() == 1) & (np.linalg.norm((back-points).reshape(-1, 2), axis=1) < 1.)
        tracked[t] = mask.sum()
        if mask.any():
            flow[t] = np.median(np.linalg.norm((nxt-points).reshape(-1, 2)[mask], axis=1))/np.hypot(*a.shape)
    return {'frame_difference': diff, 'flow_displacement': flow, 'tracked_points': tracked}


def temporal_stability(images: np.ndarray, intervals, darkening: np.ndarray,
                       cfg: dict, smooth_sigma: float, *, phase_geometry: bool = False,
                       black_ridges: bool = True) -> dict[str, np.ndarray]:
    """Estimate motion only for pairs inside filling intervals, with explicit unknowns."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    images = crop(images, cfg['roi'])
    count = len(images)
    difference = np.full(count, np.nan)
    exposure_shift = np.full(count, np.nan)
    filling_change = np.full(count, np.nan)
    flow = np.full(count, np.nan)
    tracked = np.zeros(count, dtype=int)
    strain = np.full(count, np.nan)
    states = np.full(count, 'outside_or_no_pair', dtype='<U24')
    change_limit = max(.002, .15 * float(np.ptp(darkening)))
    diagonal = np.hypot(*images.shape[1:])
    for start, end in intervals:
        for t in range(start + 1, end + 1):
            first, second = images[t-1:t+1]
            delta = second - first
            shift = float(np.median(delta))
            exposure_shift[t] = shift
            difference[t] = np.abs(delta - shift).mean()
            filling_change[t] = abs(darkening[t] - darkening[t-1])
            if filling_change[t] > change_limit:
                states[t] = 'filling_change'
                continue
            states[t] = 'insufficient_tracks'
            a = (first * 255).astype(np.uint8)
            b = (np.clip(second - shift, 0, 1) * 255).astype(np.uint8)
            points = cv2.goodFeaturesToTrack(a, maxCorners=cfg['max_corners'], qualityLevel=.02, minDistance=5)
            if points is None or len(points) < 6:
                continue
            moved, forward, error = cv2.calcOpticalFlowPyrLK(a, b, points, None)
            if moved is None:
                continue
            back, backward, _ = cv2.calcOpticalFlowPyrLK(b, a, moved, None)
            if back is None:
                continue
            consistency = np.linalg.norm((back - points).reshape(-1, 2), axis=1)
            valid = (forward.ravel() == 1) & (backward.ravel() == 1) & (consistency < 1) & (error.ravel() < 20)
            tracked[t] = valid.sum()
            if tracked[t] < max(6, .4 * len(points)):
                continue
            flow[t] = np.median(np.linalg.norm((moved - points).reshape(-1, 2)[valid], axis=1)) / diagonal
            states[t] = 'estimated'
            if phase_geometry:
                strain[t] = vessel_area_change(first, points[valid], moved[valid], black_ridges)
    smooth = flow.copy()
    minima = np.zeros(count, dtype=bool)
    known = np.isfinite(flow)
    edges = np.diff(np.r_[False, known, False].astype(int))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if smooth_sigma > 0:
            smooth[start:stop] = gaussian_filter1d(flow[start:stop], smooth_sigma, mode='nearest')
        local, _ = find_peaks(-smooth[start:stop], prominence=.0003, distance=3)
        minima[start + local] = True
    features = {'local_frame_difference': difference, 'exposure_shift': exposure_shift,
            'filling_change': filling_change, 'stability_flow': flow, 'stability_smoothed': smooth,
            'stability_tracked_points': tracked, 'stability_status': states, 'stability_local_minimum': minima}
    if phase_geometry:
        features['vessel_area_change'] = strain
    return features


def vessel_area_change(image, points, moved, black_ridges):
    """Estimate signed affine area change on spatially distributed ridge tracks."""
    operation = cv2.MORPH_BLACKHAT if black_ridges else cv2.MORPH_TOPHAT
    response = cv2.morphologyEx(image, operation, structure_element(9, 'ellipse'))
    mask = cv2.dilate((response > .06).astype(np.uint8), structure_element(5, 'ellipse'))
    xy = points.reshape(-1, 2)
    h, w = image.shape
    x, y = np.rint(xy).astype(int).T
    x = x.clip(0, w-1)
    y = y.clip(0, h-1)
    valid = ((x > .08*w) & (x < .92*w) & (y > .08*h) & (y < .92*h) & (mask[y, x] > 0))
    if valid.sum() < 10:
        return np.nan
    original, target = xy[valid], moved.reshape(-1, 2)[valid]
    affine, inliers = cv2.estimateAffine2D(original, target, ransacReprojThreshold=max(1., min(h, w)*.006))
    if affine is None or inliers.sum() < max(8, .7*len(original)):
        return np.nan
    hull = cv2.convexHull(original[inliers.ravel().astype(bool)])
    if cv2.contourArea(hull) < .035*h*w:
        return np.nan
    determinant = np.linalg.det(affine[:, :2])
    return float(np.log(determinant)) if .75 < determinant < 1.33 else np.nan


def cycle_windows(intervals, minima: np.ndarray, frame_times_ms: list[float] | None) -> list[tuple[int, int]]:
    """Only split if >=3 minima have plausible, consistent spacing and known times."""
    if frame_times_ms is None:
        return list(intervals)
    times = np.asarray(frame_times_ms)
    windows = []
    for start, end in intervals:
        lows = np.flatnonzero(minima[start:end+1]) + start
        gaps = np.diff(times[lows])
        if len(gaps) >= 2 and np.all((gaps >= 350) & (gaps <= 1500)) and gaps.max() / gaps.min() <= 1.35:
            boundaries = ((lows[:-1] + lows[1:]) // 2 + 1).tolist()
            windows.extend((a, b-1) for a, b in zip([start, *boundaries], [*boundaries, end+1]))
        else:
            windows.append((start, end))
    return windows
