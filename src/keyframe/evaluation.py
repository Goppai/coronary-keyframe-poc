"""Reference evaluation; null acceptable_frames is NOT a negative label."""
from typing import Any


def evaluate(selected: list[dict], annotation: dict[str, Any]) -> dict[str, Any]:
    if 'acceptable_frames' not in annotation or not isinstance(annotation['acceptable_frames'], list):
        raise ValueError('Explicit acceptable_frames list required; [] means no valid frame')
    acceptable = set(annotation['acceptable_frames'])
    if any(not isinstance(i, int) or i < 0 for i in acceptable):
        raise ValueError('Annotation indices must be nonnegative integers')
    preferred = annotation.get('preferred_frame')
    if preferred is not None and preferred not in acceptable:
        raise ValueError('Preferred frame must belong to acceptable_frames')
    indices = [s['frame_index'] for s in selected]
    result = {'reference_has_valid_frame': bool(acceptable), 'rejected': not indices,
              'correct_rejection': not indices if not acceptable else None,
              'top1_hit': bool(indices and indices[0] in acceptable) if acceptable else None,
              'top3_hit': bool(set(indices[:3]) & acceptable) if acceptable else None,
              'acceptable_interval_hit': bool(set(indices) & acceptable) if acceptable else None,
              'top_k_hit': bool(set(indices) & acceptable) if acceptable else None,
              'minimum_acceptable_distance': min((abs(i-j) for i in indices for j in acceptable), default=None),
              'preferred_frame_distance': min((abs(i-preferred) for i in indices), default=None) if preferred is not None else None}
    return result


def aggregate(rows: list[dict]) -> dict:
    result = {'labelled_sequences': len(rows)}
    for key in ('top1_hit', 'top3_hit', 'top_k_hit', 'acceptable_interval_hit', 'correct_rejection', 'preferred_frame_distance', 'minimum_acceptable_distance'):
        vals = [r[key] for r in rows if r.get(key) is not None]
        result[key] = {'mean': sum(vals)/len(vals) if vals else None, 'denominator': len(vals)}
    return result
