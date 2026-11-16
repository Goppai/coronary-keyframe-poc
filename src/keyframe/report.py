from html import escape
import json
from pathlib import Path
import time
import cv2
import numpy as np
import pandas as pd
from .pipeline import Analysis
from .algorithms.vessel import enhance
from .visualization import curves, comparison


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def export(analysis: Analysis, output: str | Path) -> Path:
    start = time.perf_counter()
    out = Path(output)
    # Never mix reports from different runs (stale candidates are misleading).
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be empty; use a new run directory')
    for folder in ('keyframes', 'visualizations'):
        (out/folder).mkdir(parents=True, exist_ok=True)
    r, features, prepared = analysis.result, analysis.features, analysis.prepared
    pd.DataFrame({'frame_index': np.arange(len(prepared.analysis)), **features}).to_csv(out/'scores.csv', index=False)
    indices = sorted({s['frame_index'] for m in r['methods'].values() for s in m['selected_frames']})
    preview_names = {}
    for i in indices:
        cv2.imwrite(str(out/'keyframes'/f'frame_{i:04}.png'), prepared.display[i])
        responses = (analysis.render_responses(i) if analysis.render_responses else
                     {name: enhance(prepared.analysis[i], name, r['config']['vessel']) for name in ('frangi', 'sato')})
        preview_names[i] = list(responses)
        for name, response in responses.items():
            display_response = np.clip(response/max(float(response.max()), 1e-8)*255, 0, 255).astype(np.uint8)
            cv2.imwrite(str(out/'visualizations'/f'{name}_{i:04}.png'), display_response)
    curves(features, r, out/'visualizations')
    comparison(prepared.display, r, out/'visualizations')
    cards = []
    for method, data in r['methods'].items():
        body = []
        for f in data['selected_frames']:
            i = f['frame_index']
            body.append(f'<article><h3>Frame {i} · rank {f["rank"]} · score {f["score"]:.4f}</h3><div class="images">'
                        f'<figure><img src="keyframes/frame_{i:04}.png"><figcaption>Original (sequence window)</figcaption></figure>'
                        + ''.join(f'<figure><img src="visualizations/{v}_{i:04}.png"><figcaption>{v} (display scaled)</figcaption></figure>' for v in preview_names[i])+'</div></article>')
        cards.append(f'<section><h2>{escape(method)}</h2><p>{escape(data["status"])}</p>'+(''.join(body) or '<p>No candidate. Manual review required.</p>')+'</section>')
    comparison_section = ''
    if analysis.render_responses is not None:
        comparison_section = round2_sections(r, analysis.benchmark)
    analysis.benchmark['export_seconds'] = time.perf_counter()-start
    analysis.benchmark['total_with_export_seconds'] = (analysis.benchmark['total_seconds']
                                                      + analysis.benchmark.get('matrix_seconds', 0.)
                                                      + analysis.benchmark.get('viewer_export_seconds', 0.)
                                                      + analysis.benchmark['export_seconds'])
    write_json(out/'results.json', r)
    write_json(out/'benchmark.json', analysis.benchmark)
    tech = escape(json.dumps(r['metadata'], indent=2))
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Keyframe research · {escape(r['dicom_id'])}</title>
<style>body{{max-width:1200px;margin:40px auto;padding:0 20px;font:16px system-ui;color:#203047;background:#f4f6f8}}section,article{{background:white;padding:18px;margin:16px 0;border-radius:12px}}img{{max-width:100%}}.images{{display:flex;gap:12px;flex-wrap:wrap}}figure{{margin:0;flex:1;min-width:120px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #e3e8ec}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}.notice{{background:#fff0cc;padding:20px;border-radius:10px}}h1,h2{{color:#134c65}}</style>
<h1>Coronary keyframe experiment · {escape(r['dicom_id'])}</h1>
<div class="notice">Research PoC, not clinically validated. All images may contain sensitive medical data or burned-in identity. Scores are relative rankings, not probabilities. Frame indices start at 0.</div>
<p>Filling intervals: {escape(str(r['candidate_intervals']))}{' · ' + escape(r['filling_source']) if 'filling_source' in r else ''}. Evaluation: {'annotated reference available' if r['evaluation_available'] else 'No reference labels: accuracy unavailable'}.</p>
<section><h2>Score timeline</h2><img src="visualizations/final_score_curve.png"><img src="visualizations/contrast_curve.png"><img src="visualizations/vessel_curve.png"></section>
{comparison_section}{''.join(cards)}<section><h2>Algorithm comparison</h2><img src="visualizations/algorithm_comparison.png"></section>
<section><h2>Technical metadata</h2><pre>{tech}</pre></section>
<section><h2>Performance</h2><pre>{escape(json.dumps(analysis.benchmark, indent=2))}</pre></section>
<section><h2>Parameters</h2><pre>{escape(json.dumps(r['config'], indent=2))}</pre></section>
<section><h2>Warnings and limitations</h2><pre>{escape(chr(10).join(r['warnings']))}</pre>
<p>Vessel filters respond to catheter, ribs and noise. Motion is sparse tracked-feature displacement, not isolated heart motion. Background selection can fail on already-filled sequences. Empty selections require review. Skipped vessel frames have computed=false in CSV.</p></section>
<section><h2>Reference evaluation</h2><pre>{escape(json.dumps({k: v.get('evaluation') for k,v in r['methods'].items()}, indent=2))}</pre></section></html>'''
    (out/'report.html').write_text(html, encoding='utf-8')
    (out/'SENSITIVE_MEDICAL_DATA.txt').write_text(r['privacy'])
    return out/'report.html'


def round2_sections(result: dict, benchmark: dict) -> str:
    rows = []
    for method, timings in benchmark['method_timings'].items():
        frames = ', '.join(str(frame['frame_index']) for frame in result['methods'][method]['selected_frames']) or 'Review'
        rows.append(f'<tr><td>{escape(method)}</td><td>{frames}</td><td>{timings["filter_cpu_seconds"]:.4f}</td>'
                    f'<td>{timings["cpu_seconds"]:.4f}</td><td>{timings["estimated_total_seconds"]:.4f}</td></tr>')
    section = ('<section><h2>Round 2 comparison</h2><p>Same decoded frames, normalization, filling intervals and selection windows. '
               'Black-hat responses are shared across its two methods. Estimated method total adds the shared stages once.</p>'
               f'<p>Selection windows: {escape(str(result["selection_windows"]))}. Motion states: {escape(str(result["motion_states"]))}.</p>'
               '<table><thead><tr><th>Method</th><th>Selected frames</th><th>Filter CPU (s)</th><th>Filter + ranking CPU (s)</th>'
               '<th>Estimated method wall time (s)</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
               '<img src="visualizations/final_scores_comparison.png" alt="Final score curves by method"></section>')
    if phase := result.get('cardiac_phase'):
        section += ('<section><h2>Cardiac phase references</h2><p>Repeated vascular expansion-to-contraction transitions provide '
                    'image-based end-diastolic estimates. Periodic low-motion markers are review cues.</p>'
                    f'<pre>{escape(json.dumps(phase, indent=2))}</pre></section>')
    matrix = result.get('experiment_matrix')
    if matrix:
        rows = []
        for name, variant in matrix['variants'].items():
            frames = ', '.join(str(f['frame_index']) for f in variant['selected_frames']) or 'Review'
            rows.append(f'<tr><td>{escape(name)}</td><td>{variant["computed_frames"]}</td><td>{frames}</td>'
                        f'<td>{variant["wall_mean_seconds"]:.4f}</td><td>{variant["cpu_mean_seconds"]:.4f}</td></tr>')
        section += (f'<section><h2>Frangi and Black-hat parameter matrix</h2><p>{escape(matrix["scope"])}</p>'
                    '<table><thead><tr><th>Configuration</th><th>Frames processed</th><th>Selected frames</th>'
                    '<th>Mean wall (s)</th><th>Mean CPU (s)</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
                    f'<pre>{escape(json.dumps(matrix["scale_reuse"], indent=2))}</pre></section>')
    return section
