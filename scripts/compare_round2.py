#!/usr/bin/env python3
"""Run second-round comparisons on a file or directory of XA sequences."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from pydicom import dcmread
from keyframe.config import load_config
from keyframe.dicom_reader import DicomError
from keyframe.evaluation import aggregate
from keyframe.report import export, write_json
from keyframe.round2 import METHOD_FILTERS, analyze_round2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--method', choices=METHOD_FILTERS, help='Run one method instead of the five-way comparison')
    parser.add_argument('--matrix-repeat', type=int, default=0, help='Enable the size/scale/kernel matrix with N measured repetitions')
    parser.add_argument('--filling-fallback', action=argparse.BooleanOptionalAction, help='Use local vessel evidence to recover or check global filling')
    parser.add_argument('--end-diastole', action=argparse.BooleanOptionalAction, help='Enable experimental cardiac phase evidence and ranking bonus')
    parser.add_argument('--sample-count', type=int, help='Select N sequences at evenly spaced frame-count quantiles')
    parser.add_argument('--annotations', type=Path, help='JSON file for one DICOM or directory of sample_NNN.json for a folder')
    args = parser.parse_args()
    if args.matrix_repeat < 0 or (args.sample_count is not None and args.sample_count < 1):
        parser.error('Invalid repeat or sample count')
    if not args.input.exists() or (args.output.exists() and any(args.output.iterdir())):
        parser.error('Input must exist and output must be empty')
    cfg = load_config(args.config)
    for key in ('filling_fallback', 'end_diastole'):
        value = getattr(args, key)
        if value is not None:
            cfg['round2'][key] = value
    if args.input.is_dir():
        files = sorted(p for p in args.input.rglob('*') if p.is_file() and not p.name.startswith('.')
                       and p.suffix.lower() in ('', '.dcm', '.dicom', '.ima'))
    else:
        files = [args.input]
    jobs, errors = [], []
    for i, path in enumerate(files, 1):
        sid = f'sample_{i:03}'
        try:
            header = dcmread(path, stop_before_pixels=True, specific_tags=['NumberOfFrames'])
            jobs.append((sid, path, int(header.get('NumberOfFrames', 1))))
        except Exception:
            errors.append({'dicom_id': sid, 'reason': 'Cannot read DICOM header'})
    valid_headers = len(jobs)
    if args.sample_count and args.sample_count < valid_headers:
        ranked = sorted(jobs, key=lambda job: job[2])
        jobs = [ranked[int((i+.5)*valid_headers/args.sample_count)] for i in range(args.sample_count)]
    args.output.mkdir(parents=True, exist_ok=True)
    reports, evaluations = [], {}
    start = perf_counter()
    for sid, path, _ in jobs:
        try:
            label = args.annotations
            if label and args.input.is_dir():
                label = label/f'{sid}.json'
                if not label.exists():
                    label = None
            annotation = json.loads(label.read_text()) if label else None
            analysis = analyze_round2(path, cfg, sid, annotation, args.matrix_repeat, method=args.method)
            export(analysis, args.output/sid)
            methods = analysis.result['methods']
            for name, result in methods.items():
                if annotation is not None:
                    evaluations.setdefault(name, []).append(result['evaluation'])
            reports.append({'dicom_id': sid, 'report': f'{sid}/report.html', 'frames': analysis.benchmark['frames'],
                            'candidate_intervals': analysis.result['candidate_intervals'],
                            'filling_source': analysis.result['filling_source'], 'cardiac_phase': analysis.result.get('cardiac_phase'),
                            'methods': methods,
                            'core_wall_seconds': analysis.benchmark['total_seconds'], 'core_cpu_seconds': analysis.benchmark['cpu_seconds']})
            del analysis
            print(f'{sid}: {reports[-1]["frames"]} frames, {reports[-1]["core_wall_seconds"]:.3f}s core', flush=True)
        except (ValueError, OSError, RuntimeError) as exc:
            errors.append({'dicom_id': sid, 'reason': str(exc) if isinstance(exc, DicomError) else type(exc).__name__})
            print(f'{sid}: failed', file=sys.stderr, flush=True)
    write_json(args.output/'summary.json', {'reports': reports, 'errors': errors, 'valid_headers': valid_headers,
               'total_wall_seconds': perf_counter()-start,
               'id_policy': 'sample_NNN uses sorted eligible file order before stratification; no source filenames exported.',
               'sampling': 'evenly spaced frame-count quantiles' if args.sample_count else 'all eligible files',
               'evaluation': {name: aggregate(rows) for name, rows in evaluations.items()} or None})
    return int(bool(errors or not reports))


if __name__ == '__main__':
    raise SystemExit(main())
