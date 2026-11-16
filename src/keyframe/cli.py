import argparse
import json
from pathlib import Path
import sys
import tempfile
from .config import load_config
from .dicom_reader import DicomError
from .pipeline import analyze
from .report import export, write_json
from .scoring import METHODS
from .benchmark import summarize, environment
from .evaluation import aggregate
from .synthetic import phantom, write_dicom


def parser(kind: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f'CPU coronary keyframe research: {kind}; zero-based frame indices')
    p.add_argument('--input', type=Path, required=kind != 'synthetic')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path)
    p.add_argument('--method', choices=METHODS, default='multi_feature')
    p.add_argument('--top-k', type=int)
    p.add_argument('--compare-all', action='store_true')
    p.add_argument('--sample-id', default='sample_001', help='Pseudonymous internal identifier')
    p.add_argument('--annotations', type=Path, help='Single JSON for analyze; directory of sample_NNN.json for batch')
    if kind == 'benchmark':
        p.add_argument('--repeat', type=int, default=3)
        p.add_argument('--warmup', type=int, default=1)
        p.add_argument('--sizes', type=int, nargs='+', default=[256,512,1024], help='Synthetic input sizes (use --input synthetic)')
        p.add_argument('--frame-counts', type=int, nargs='+', default=[50,100,200])
        p.add_argument('--analysis-size', type=int, help='Override analysis max dimension; default config 256')
    if kind == 'synthetic':
        p.add_argument('--size', type=int, default=256)
        p.add_argument('--frames', type=int, default=100)
        p.add_argument('--peaks', type=int, default=1)
        p.add_argument('--no-contrast', action='store_true')
    return p


def main(kind: str = 'analyze') -> int:
    p = parser(kind)
    args = p.parse_args()
    try:
        cfg = load_config(args.config)
        if args.top_k is not None:
            if args.top_k < 1:
                raise ValueError('top-k must be positive')
            cfg['selection']['top_k'] = args.top_k
        if kind == 'synthetic':
            if args.size < 32 or args.frames < 1 or args.peaks < 1:
                raise ValueError('Invalid phantom dimensions')
            if args.output.exists():
                raise ValueError('Synthetic output already exists')
            write_dicom(args.output, phantom(args.size,args.frames,contrast=not args.no_contrast,peaks=args.peaks))
            print('Synthetic phantom written (not clinical data).')
            return 0
        if kind == 'benchmark':
            return run_benchmark(args, cfg)
        if kind == 'analyze':
            annotation = json.loads(args.annotations.read_text()) if args.annotations else None
            a = analyze(args.input, cfg, args.method, args.compare_all, args.sample_id, annotation)
            print(export(a, args.output))
            return 0
        if not args.input.is_dir():
            raise ValueError('Batch input must be a directory')
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError('Output must be empty')
        args.output.mkdir(parents=True, exist_ok=True)
        rows, errors, references = [], [], []
        # All files are attempted; errors are explicit and source paths are not logged.
        for index, path in enumerate(sorted(p for p in args.input.rglob('*') if p.is_file()), 1):
            sid = f'sample_{index:03}'
            annotation = None
            try:
                if args.annotations:
                    label = args.annotations/f'{sid}.json'
                    if label.exists():
                        annotation = json.loads(label.read_text())
                a = analyze(path,cfg,args.method,args.compare_all,sid,annotation)
                export(a,args.output/sid)
                rows.append({'dicom_id':sid,'report':f'{sid}/report.html','status':a.result['methods'][args.method]['status']})
                if annotation:
                    references.append(a.result['methods'][args.method]['evaluation'])
            except (ValueError, OSError, RuntimeError) as exc:
                errors.append({'dicom_id':sid,'error_type':type(exc).__name__, 'reason': safe_error(exc)})
        write_json(args.output/'batch_summary.json', {'samples':rows,'errors':errors,'evaluation':aggregate(references) if references else None,
                                                    'id_policy':'sample_NNN assigned in sorted recursive file order; do not rename/reorder inputs between labelling and analysis'})
        print(f'Processed {len(rows)}; failed {len(errors)}. See batch_summary.json')
        return 1 if errors or not rows else 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(safe_error(exc), file=sys.stderr)
        return 2


def safe_error(exc: Exception) -> str:
    # OS/library exception text can contain patient-identifying paths.
    return str(exc) if isinstance(exc, (DicomError, ValueError)) and not isinstance(exc, json.JSONDecodeError) else f'{type(exc).__name__}: operation failed (source path suppressed)'


def run_benchmark(args, cfg: dict) -> int:
    if args.repeat < 1 or args.warmup < 0:
        raise ValueError('Invalid repetition count')
    if args.analysis_size:
        if args.analysis_size < 16:
            raise ValueError('analysis-size must be >=16')
        cfg['preprocessing']['max_size'] = args.analysis_size
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    groups = []
    with tempfile.TemporaryDirectory() as tmp:
        if str(args.input) == 'synthetic':
            jobs = []
            for size in args.sizes:
                for count in args.frame_counts:
                    if size < 32 or count < 1:
                        raise ValueError('Invalid phantom dimensions')
                    path = Path(tmp)/f'{size}_{count}.dcm'
                    write_dicom(path, phantom(size, count))
                    jobs.append((f'phantom_{size}_{count}', path))
        else:
            files = sorted(p for p in args.input.rglob('*') if p.is_file()) if args.input.is_dir() else [args.input]
            jobs = [(f'sample_{i:03}',path) for i,path in enumerate(files,1)]
        for sid, path in jobs:
            runs = []
            for iteration in range(args.warmup+args.repeat):
                a = analyze(path, cfg, args.method, args.compare_all, sid)
                if iteration >= args.warmup:
                    runs.append(a.benchmark)
                del a
            group = {'dicom_id':sid,'runs':runs,'summary':summarize(runs)}
            groups.append(group)
            write_json(args.output/'benchmark.json', {'environment':environment(),'config':cfg,'synthetic':str(args.input)=='synthetic','warmup':args.warmup,'groups':groups,
                        'notes':'Same-process repetitions; warm cache; 10ms sampled RSS includes retained allocator memory. Decode through selection, no report export. No accuracy claim.'})
            print(f'{sid}: mean {group["summary"]["mean_seconds"]:.3f}s', flush=True)
    if not groups:
        raise ValueError('No input files')
    return 0
