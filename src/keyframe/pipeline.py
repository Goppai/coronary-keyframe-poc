from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable
import numpy as np
from .dicom_reader import read_dicom
from .preprocessing import preprocess, Prepared
from .algorithms.contrast import detect_contrast, normalize
from .algorithms.vessel import enhance, vessel_features
from .algorithms.sharpness import sharpness
from .algorithms.motion import motion_features
from .benchmark import Monitor, stage, environment
from .scoring import fuse, METHODS
from .selector import select_frames
from .evaluation import evaluate

@dataclass
class Analysis:
    result: dict
    features: dict[str, np.ndarray]
    prepared: Prepared
    benchmark: dict
    render_responses: Callable[[int], dict[str, np.ndarray]] | None = None


def analyze(path: str | Path, cfg: dict, method: str = 'multi_feature', compare_all: bool = False,
            sample_id: str = 'sample_001', annotation: dict | None = None) -> Analysis:
    if not re.fullmatch(r'[A-Za-z0-9_-]+', sample_id):
        raise ValueError('Use a pseudonymous alphanumeric sample ID')
    if method not in METHODS:
        raise ValueError('Unsupported method')
    if annotation and annotation.get('dicom_id') != sample_id:
        raise ValueError('Annotation dicom_id does not match sample ID')
    timings = {}
    methods = METHODS if compare_all else (method,)
    need_f = compare_all or 'frangi' in method or method == 'multi_feature'
    need_s = compare_all or 'sato' in method or method == 'multi_feature'
    with Monitor() as monitor:
        with stage(timings, 'dicom_decode'):
            sequence = read_dicom(path)
            del sequence.raw_pixels
        with stage(timings, 'preprocessing'):
            prepared = preprocess(sequence.pixels, cfg['preprocessing'])
        del sequence.pixels
        images = prepared.analysis
        n = len(images)
        if annotation:
            if any(i >= n for i in annotation.get('acceptable_frames', [])):
                raise ValueError('Annotation frame index outside sequence')
        with stage(timings, 'contrast'):
            contrast = detect_contrast(images, cfg['contrast'])
        features = {'contrast': contrast.frame_scores, **contrast.features}
        candidate = np.zeros(n, dtype=bool)
        for a, b in contrast.candidate_intervals:
            candidate[a:b+1] = True
        indices = np.flatnonzero(candidate) if cfg['vessel']['candidate_only'] else np.arange(n)
        masks = {}
        for name, needed in (('frangi', need_f), ('sato', need_s)):
            for feature in ('score', 'area', 'strength'):
                features[f'{name}_{feature}'] = np.zeros(n)
            features[f'{name}_computed'] = np.zeros(n, dtype=bool)
            with stage(timings, name):
                if needed:
                    for t in indices:
                        response = enhance(images[t], name, cfg['vessel'])
                        values = vessel_features(response, cfg['vessel']['response_threshold'])
                        for key, value in values.items():
                            features[f'{name}_{key}'][t] = value
                        features[f'{name}_computed'][t] = True
                        if cfg['sharpness']['region'] == 'vessel':
                            masks[t] = masks.get(t, np.zeros_like(response, bool)) | (response > cfg['vessel']['response_threshold'])
        with stage(timings, 'sharpness'):
            values = [sharpness(image, masks.get(t, np.zeros_like(image, bool)) if cfg['sharpness']['region'] == 'vessel' else None) for t, image in enumerate(images)]
            features.update({k: np.array([v[k] for v in values]) for k in ('laplacian', 'gradient')})
        with stage(timings, 'motion'):
            features.update(motion_features(images, cfg['motion']))
        features['candidate'] = candidate
        features['sharpness_normalized'] = (normalize(features['laplacian'])+normalize(features['gradient']))/2
        features['motion_normalized'] = normalize(features['flow_displacement'])
        selections, warnings = {}, list(contrast.warnings)
        with stage(timings, 'fusion_selection'):
            for name in methods:
                scores = fuse(features, name, cfg['weights'])
                features[f'final_{name}'] = scores
                low_vessel = False
                if name != 'contrast_only':
                    vessel_names = ('frangi', 'sato') if name == 'multi_feature' else (('frangi',) if 'frangi' in name else ('sato',))
                    low_vessel = all(features[f'{v}_score'].max() < cfg['selection']['min_vessel_score'] for v in vessel_names)
                selected = [] if low_vessel else select_frames(scores, images, contrast.candidate_intervals, cfg['selection'])
                selections[name] = {'selected_frames': selected, 'status': 'candidate_requires_review' if selected else 'rejected_requires_review',
                                    'reason': 'insufficient_vessel_response' if low_vessel else ('no_valid_candidate' if not selected else None)}
                if annotation is not None:
                    selections[name]['evaluation'] = evaluate(selected, annotation)
        result = {'dicom_id': sample_id, 'metadata': sequence.metadata, 'analysis_shape': list(images.shape),
                  'config': cfg, 'normalization_limits': prepared.limits,
                  'score_semantics': 'sequence-relative heuristic ranking scores, NOT calibrated probabilities',
                  'candidate_intervals': contrast.candidate_intervals, 'contrast_confidence_heuristic': contrast.confidence,
                  'methods': selections, 'primary_method': method, 'selected_frames': selections[method]['selected_frames'],
                  'warnings': warnings, 'evaluation_available': annotation is not None,
                  'privacy': 'Images may contain burned-in identity and sensitive medical information. Not anonymous.'}
    benchmark = {**monitor.result(), 'stages': timings, 'frames': n, 'environment': environment(),
                 'scope': 'DICOM decode through keyframe selection; excludes report/export and process startup',
                 'computed_vessel_frames': int(len(indices)) if (need_f or need_s) else 0,
                 'input_shape': [n, sequence.metadata['rows'], sequence.metadata['columns']],
                 'analysis_shape': list(images.shape),
                 'vessel_frames_per_filter': {v: int(features[v+'_computed'].sum()) for v in ('frangi','sato')}, 'methods': list(methods)}
    benchmark['fps'] = n/benchmark['total_seconds']
    return Analysis(result, features, prepared, benchmark)
