"""Second-round comparison on one shared decode, normalization and filling mask."""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, process_time
from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .algorithms.cheap_vessel import blackhat_vessel_score, sobel_vessel_score
from .algorithms.cardiac_phase import IMAGE_PHASE_SCORE_MAX, phase_evidence
from .algorithms.contrast import detect_contrast, normalize
from .algorithms.filling_fallback import earlier_ridges_dominate, recover_filling
from .algorithms.frangi_fast import fast_frangi
from .algorithms.motion import cycle_windows, temporal_stability
from .algorithms.vessel import enhance, vessel_features
from .benchmark import Monitor, environment, stage
from .dicom_reader import read_dicom
from .evaluation import evaluate
from .pipeline import Analysis
from .preprocessing import preprocess
from .selector import select_frames

METHOD_FILTERS = {
    'contrast_frangi': 'frangi',
    'contrast_sobel': 'sobel',
    'contrast_blackhat': 'blackhat',
    'contrast_blackhat_motion': 'blackhat',
    'contrast_frangi_fast': 'frangi_fast',
}

FrameFilter = Callable[[np.ndarray], tuple[np.ndarray, dict[str, float]]]


@dataclass
class ComparisonContext:
    images: np.ndarray
    background: np.ndarray
    features: dict[str, np.ndarray]
    intervals: list[tuple[int, int]]
    windows: list[tuple[int, int]]
    cfg: dict


def frame_filter(name: str, cfg: dict, background: np.ndarray, *, sigmas=None, size=None) -> FrameFilter:
    v, exp = cfg['vessel'], cfg['round2']
    dark = v['black_ridges']
    if name == 'sobel':
        mask_threshold = exp['sobel_background_threshold']
        return lambda frame: sobel_vessel_score(
            frame, background=background if mask_threshold is not None else None,
            background_threshold=mask_threshold or 0., threshold=exp['sobel_threshold'], black_ridges=dark)
    if name == 'blackhat':
        return lambda frame: blackhat_vessel_score(
            frame, size=size or exp['blackhat_size'], shape=exp['blackhat_shape'],
            threshold=exp['blackhat_threshold'], black_ridges=dark)
    scales = sigmas if sigmas is not None else (exp['optimized_sigmas'] if name == 'frangi_fast' else v['sigmas'])
    frangi_cfg = {**v, 'sigmas': scales}

    def apply(frame):
        if name == 'frangi_fast':
            response = fast_frangi(frame, scales, v['frangi_gamma'], dark)
        else:
            response = enhance(frame, 'frangi', frangi_cfg)
        return response, vessel_features(response, v['response_threshold'])
    return apply


def measure_frames(images: np.ndarray, indices: np.ndarray, operation: FrameFilter) -> dict[str, np.ndarray]:
    values = {key: np.zeros(len(images)) for key in ('score', 'strength', 'area')}
    values['computed'] = np.zeros(len(images), dtype=bool)
    values['computed'][indices] = True
    for i in indices:
        _, stats = operation(images[i])
        for key, value in stats.items():
            values[key][i] = value
    return values


def rank(ctx: ComparisonContext, score: np.ndarray, use_motion: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    candidate = ctx.features['candidate']
    raw = np.zeros_like(score)
    smooth = raw.copy()
    selected = []
    reason = 'no_filling_interval'
    if candidate.any():
        weights = ctx.cfg['weights']
        raw[candidate] = weights['vessel'] * normalize(score[candidate]) + weights['contrast'] * ctx.features['contrast'][candidate]
        if use_motion:
            raw[candidate] -= weights['motion'] * ctx.features['stability_penalty'][candidate]
        sigma = ctx.cfg['round2']['score_smooth_sigma']
        for start, end in ctx.intervals:
            segment = raw[start:end+1]
            smooth[start:end+1] = gaussian_filter1d(segment, sigma, mode='nearest') if sigma > 0 else segment
        bonus = ctx.features.get('end_diastole_bonus')
        if bonus is not None:
            smooth[candidate] += bonus[candidate]
        reason = 'insufficient_vessel_response'
        if score[candidate].max() >= ctx.cfg['selection']['min_vessel_score']:
            selected = select_frames(smooth, ctx.images, ctx.intervals, ctx.cfg['selection'], preferred_windows=ctx.windows)
            if bonus is not None:
                for frame in selected:
                    frame['end_diastole_bonus'] = float(bonus[frame['frame_index']])
            reason = None if selected else 'no_valid_candidate'
    return raw, smooth, {'selected_frames': selected, 'status': 'candidate_requires_review' if selected else 'rejected_requires_review', 'reason': reason}


def run_matrix(ctx: ComparisonContext, prepared_sizes: dict[int, np.ndarray], repeat: int, annotation: dict | None) -> dict:
    """Hold intervals and ranking inputs fixed while varying filter size and scales."""
    active = np.flatnonzero(ctx.features['candidate'])
    all_frames = np.arange(len(ctx.images))
    variants = []
    for backend in ('frangi', 'frangi_fast'):
        for size, scales, candidate_only in [(256, (1,2,3), True), (256, (1,2), True),
                                              (512, (1,2,3), True), (512, (1,2), True),
                                              (256, (1,2,3), False)]:
            variants.append((f'{backend}_{size}_s{len(scales)}_{"candidate" if candidate_only else "all"}',
                             backend, size, scales, candidate_only, None))
    for kernel in (5,9,15):
        variants.append((f'blackhat_256_k{kernel}', 'blackhat', 256, None, True, kernel))
    results = {}
    for name, backend, size, scales, candidate_only, kernel in variants:
        images = prepared_sizes[size]
        operation = frame_filter(backend, ctx.cfg, ctx.background, sigmas=scales, size=kernel)
        indices = active if candidate_only else all_frames
        wall, cpu = [], []
        for _ in range(repeat):
            start, cpu_start = perf_counter(), process_time()
            values = measure_frames(images, indices, operation)
            wall.append(perf_counter() - start)
            cpu.append(process_time() - cpu_start)
        raw, smooth, selection = rank(ctx, values['score'])
        if annotation is not None:
            selection['evaluation'] = evaluate(selection['selected_frames'], annotation)
        ctx.features.update({f'{name}_{key}': value for key, value in values.items()})
        ctx.features[f'final_raw_{name}'], ctx.features[f'final_{name}'] = raw, smooth
        results[name] = {**selection, 'shape': list(images.shape[1:]), 'sigmas': scales, 'kernel': kernel,
                         'candidate_only': candidate_only, 'computed_frames': len(indices),
                         'wall_seconds': wall, 'cpu_seconds': cpu, 'wall_mean_seconds': float(np.mean(wall)),
                         'cpu_mean_seconds': float(np.mean(cpu))}
    ref = [x['frame_index'] for x in results['frangi_256_s3_candidate']['selected_frames']]
    for row in results.values():
        actual = [x['frame_index'] for x in row['selected_frames']]
        row['same_ranked_frames_as_reference'] = actual == ref
        row['top1_delta_frames'] = actual[0] - ref[0] if actual and ref else None
    return {'repeat': repeat, 'scope': 'Filter and response statistics only; same candidate intervals at both sizes.',
            'variants': results, 'scale_reuse': measure_scale_reuse(prepared_sizes[256], active, ctx.cfg['vessel'], repeat)}


def measure_scale_reuse(images: np.ndarray, indices: np.ndarray, vessel: dict, repeat: int) -> dict:
    """Compare two independent fast scale sets with an identical per-frame cached pair."""
    gamma, dark = vessel['frangi_gamma'], vessel['black_ridges']
    rows = {}
    for reuse in (False, True):
        wall, cpu = [], []
        for _ in range(repeat):
            start, cpu_start = perf_counter(), process_time()
            for i in indices:
                cache = {} if reuse else None
                fast_frangi(images[i], (1,2), gamma, dark, cache)
                fast_frangi(images[i], (1,2,3), gamma, dark, cache)
            wall.append(perf_counter() - start)
            cpu.append(process_time() - cpu_start)
        rows['shared' if reuse else 'independent'] = {'wall_mean_seconds': float(np.mean(wall)),
                                                     'cpu_mean_seconds': float(np.mean(cpu)),
                                                     'scale_evaluations_per_run': len(indices) * (3 if reuse else 5)}
    return rows


def analyze_round2(path: str | Path, cfg: dict, sample_id: str = 'sample_001',
                   annotation: dict | None = None, matrix_repeat: int = 0, *, method: str | None = None,
                   compare_all: bool = False) -> Analysis:
    if method is not None and method not in METHOD_FILTERS:
        raise ValueError('Unsupported round 2 method')
    methods_to_run = METHOD_FILTERS if method is None or compare_all else {method: METHOD_FILTERS[method]}
    if annotation is not None and annotation.get('dicom_id') != sample_id:
        raise ValueError('Annotation dicom_id does not match sample ID')
    timings, cpu = {}, {}
    experimental = cfg['round2']
    with Monitor() as monitor:
        with stage(timings, 'dicom_decode', cpu):
            sequence = read_dicom(path)
            del sequence.raw_pixels
        with stage(timings, 'preprocessing', cpu):
            prepared = preprocess(sequence.pixels, cfg['preprocessing'])
        images = prepared.analysis
        n = len(images)
        if annotation is not None and any(i >= n for i in annotation.get('acceptable_frames', [])):
            raise ValueError('Annotation frame index outside sequence')
        if not matrix_repeat:
            del sequence.pixels
        dark = cfg['vessel']['black_ridges']
        with stage(timings, 'contrast', cpu):
            contrast = detect_contrast(images if dark else 1-images, cfg['contrast'])
            background = contrast.reference if dark else 1-contrast.reference
            intervals = contrast.candidate_intervals
        source = 'global_darkening' if intervals else 'unavailable'
        filling_score = contrast.frame_scores
        extra_features = {}
        local_intervals = None
        if experimental.get('filling_fallback', False):
            with stage(timings, 'filling_fallback', cpu):
                local_intervals, local_score, extra_features = recover_filling(images, cfg['contrast'], black_ridges=dark)
            if local_intervals and not intervals:
                intervals = local_intervals
                source, filling_score = 'local_ridges', local_score
                contrast.warnings.append('Global filling was inconclusive. Local ridge recovery found experimental candidate intervals.')
            elif earlier_ridges_dominate(intervals, local_intervals, extra_features['fallback_ridge_smoothed']):
                intervals = local_intervals
                source, filling_score = 'local_ridges_early', local_score
                contrast.warnings.append('Earlier local ridge evidence had a peak over 1.5 times the peak in the later global intervals.')
        candidate = np.zeros(n, dtype=bool)
        for start, end in intervals:
            candidate[start:end+1] = True
        with stage(timings, 'temporal_stability', cpu):
            stability = temporal_stability(images, intervals, contrast.features['darkening_raw'], cfg['motion'],
                experimental['score_smooth_sigma'], phase_geometry=experimental.get('end_diastole', False), black_ridges=dark)
            windows = cycle_windows(intervals, stability['stability_local_minimum'], sequence.metadata['frame_times_ms'])
            observed = np.isfinite(stability['stability_smoothed'])
            penalty = np.zeros(n)
            if observed.any():
                penalty[observed] = normalize(stability['stability_smoothed'][observed])
                penalty[candidate & ~observed] = np.median(penalty[observed])
        phase = None
        if experimental.get('end_diastole', False):
            with stage(timings, 'cardiac_phase', cpu):
                phase_weight = experimental.get('end_diastole_weight', .03)
                phase, phase_features = phase_evidence(sequence.metadata['frame_times_ms'],
                    stability['stability_local_minimum'], intervals, windows, stability['vessel_area_change'])
                phase_features['end_diastole_bonus'] = np.where(candidate, phase_weight *
                    np.nan_to_num(phase_features['end_diastole_score'], nan=0.), 0.)
                phase['max_score_bonus'] = phase_weight * IMAGE_PHASE_SCORE_MAX
                extra_features.update(phase_features)
        features = {'contrast': filling_score, 'candidate': candidate, **contrast.features, **stability, **extra_features,
                    'stability_penalty': penalty, 'motion_normalized': penalty, 'selection_window': np.zeros(n, dtype=int)}
        if experimental.get('filling_fallback', False):
            features['global_contrast'] = contrast.frame_scores
        for index, (start, end) in enumerate(windows, 1):
            features['selection_window'][start:end+1] = index
        ctx = ComparisonContext(images, background, features, intervals, windows, cfg)
        filters = {name: frame_filter(name, cfg, background) for name in dict.fromkeys(methods_to_run.values())}
        indices = np.flatnonzero(candidate)
        for name, operation in filters.items():
            frames = np.arange(n) if name.startswith('frangi') and not cfg['vessel']['candidate_only'] else indices
            with stage(timings, name, cpu):
                values = measure_frames(images, frames, operation)
            features.update({f'{name}_{key}': value for key, value in values.items()})
        shared_stages = tuple(timings_key for timings_key in (
            'dicom_decode', 'preprocessing', 'contrast', 'filling_fallback', 'temporal_stability', 'cardiac_phase'
        ) if timings_key in timings)
        methods, method_timings = {}, {}
        for method_name, name in methods_to_run.items():
            key = 'rank_' + method_name
            with stage(timings, key, cpu):
                raw, smooth, selection = rank(ctx, features[f'{name}_score'], method_name.endswith('_motion'))
            features[f'final_raw_{method_name}'], features[f'final_{method_name}'] = raw, smooth
            if annotation is not None:
                selection['evaluation'] = evaluate(selection['selected_frames'], annotation)
            methods[method_name] = selection
            # Filter times belong to each standalone combination. Black-hat was evaluated once in the combined run.
            method_timings[method_name] = {'wall_seconds': timings[name] + timings[key], 'cpu_seconds': cpu[name] + cpu[key],
                                      'filter_seconds': timings[name], 'filter_cpu_seconds': cpu[name], 'selection_seconds': timings[key],
                                      'estimated_total_seconds': sum(timings[s] for s in shared_stages) + timings[name] + timings[key],
                                      'estimated_total_cpu_seconds': sum(cpu[s] for s in shared_stages) + cpu[name] + cpu[key]}
        primary = method or 'contrast_frangi_fast'
    result = {'dicom_id': sample_id, 'metadata': sequence.metadata, 'analysis_shape': list(images.shape),
              'config': cfg, 'normalization_limits': prepared.limits,
              'score_semantics': 'sequence-relative heuristic ranking scores',
              'candidate_intervals': intervals, 'selection_windows': windows,
              'filling_source': source, 'global_candidate_intervals': contrast.candidate_intervals,
              'motion_states': dict(Counter(stability['stability_status'][candidate].tolist())),
              'contrast_confidence_heuristic': contrast.confidence, 'methods': methods,
              'primary_method': primary, 'selected_frames': methods[primary]['selected_frames'],
              'warnings': contrast.warnings, 'evaluation_available': annotation is not None,
              'privacy': 'Images may contain burned-in identity and sensitive medical information. Not anonymous.'}
    if phase is not None:
        result['cardiac_phase'] = phase
    if local_intervals is not None:
        result['local_candidate_intervals'] = local_intervals
    benchmark = {**monitor.result(), 'stages': timings, 'stage_cpu_seconds': cpu, 'method_timings': method_timings,
                 'shared_stages': list(shared_stages), 'frames': n, 'fps': n/monitor.elapsed, 'environment': environment(),
                 'scope': 'Combined core comparison with one decode. Excludes optional matrix and report export.'}
    if matrix_repeat:
        start, cpu_start = perf_counter(), process_time()
        prepared_sizes = {size: images if size == cfg['preprocessing']['max_size'] else
                          preprocess(sequence.pixels, {**cfg['preprocessing'], 'max_size': size}).analysis for size in (256,512)}
        del sequence.pixels
        result['experiment_matrix'] = run_matrix(ctx, prepared_sizes, matrix_repeat, annotation)
        benchmark['matrix_seconds'] = perf_counter() - start
        benchmark['matrix_cpu_seconds'] = process_time() - cpu_start
    def render(i):
        return {name: operation(images[i])[0] for name, operation in filters.items()}

    return Analysis(result, features, prepared, benchmark, render)
