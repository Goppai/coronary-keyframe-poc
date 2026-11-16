import numpy as np
from scipy.signal import find_peaks


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.ravel().astype(float), b.ravel().astype(float)
    x, y = x-x.mean(), y-y.mean()
    den = np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/den) if den > 1e-10 else float(np.allclose(a, b))


def select_frames(scores: np.ndarray, images: np.ndarray, intervals: list[tuple[int, int]], cfg: dict, *, preferred_windows: list[tuple[int, int]] | None = None) -> list[dict]:
    if not cfg['preserve_intervals']:
        preferred_windows = None
    peaks = []
    for a, b in intervals:
        # Padding allows endpoints; plateaus yield a central local maximum.
        local, _ = find_peaks(np.r_[-np.inf, scores[a:b+1], -np.inf])
        peaks.extend((local+a-1).tolist())
    windows = preferred_windows if preferred_windows is not None else intervals
    if preferred_windows is not None:
        for a, b in windows:
            # Cycle boundaries may contain no global peak. Reserve a local maximum in each window.
            local, _ = find_peaks(np.r_[-np.inf, scores[a:b+1], -np.inf])
            peaks.extend((local+a-1).tolist())
    ordered = sorted(set(peaks), key=lambda i: (-scores[i], i))
    preferred = []
    if cfg['preserve_intervals']:
        for a, b in windows:
            own = [i for i in ordered if a <= i <= b]
            if own:
                preferred.append(own[0])
        preferred.sort(key=lambda i: -scores[i])
    window_of = {i: group for group, (a, b) in enumerate(windows) for i in range(a, b+1)} if preferred_windows is not None else {}
    chosen = []
    for i in preferred + [i for i in ordered if i not in preferred]:
        if len(chosen) >= cfg['top_k']:
            break
        if any(abs(i-j) < cfg['min_frame_distance'] or (window_of.get(i) == window_of.get(j)
                and similarity(images[i], images[j]) > cfg['max_similarity']) for j in chosen):
            continue
        chosen.append(i)
    chosen.sort(key=lambda i: -scores[i])
    return [{'frame_index': int(i), 'score': float(scores[i]), 'rank': rank+1} for rank, i in enumerate(chosen)]
