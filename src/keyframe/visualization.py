from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .algorithms.contrast import normalize


def curves(features: dict, result: dict, destination: Path) -> None:
    filters = ('frangi', 'sato', 'frangi_fast', 'sobel', 'blackhat')
    vessels = {name: np.where(features[name+'_computed'], normalize(features[name+'_score']), np.nan)
               for name in filters if name+'_computed' in features}
    series = {'Contrast': features['contrast'], **vessels, 'Motion (proxy)': features['motion_normalized'],
              'Final': features['final_'+result['primary_method']]}
    if 'sharpness_normalized' in features:
        series['Sharpness'] = features['sharpness_normalized']
    filling_labels = ['Contrast']
    if result.get('filling_source') in ('local_ridges', 'local_ridges_early'):
        series['Global contrast'] = features['global_contrast']
        filling_labels.append('Global contrast')
    plots = [('contrast_curve.png', filling_labels), ('vessel_curve.png', list(vessels)), ('final_score_curve.png', list(series))]
    if 'selection_windows' in result:
        methods = list(result['methods'])
        series.update({method: np.where(features['candidate'], features['final_'+method], np.nan) for method in methods})
        plots.append(('final_scores_comparison.png', methods))
    for filename, labels in plots:
        fig, ax = plt.subplots(figsize=(11, 4))
        for label in labels:
            ax.plot(series[label], label=label)
        for a, b in result['candidate_intervals']:
            ax.axvspan(a, b, alpha=.08, color='green')
        for s in result['selected_frames']:
            ax.axvline(s['frame_index'], color='black', ls=':', alpha=.6)
        for index, frame in enumerate(result.get('cardiac_phase', {}).get('reference_frames', [])):
            ax.axvline(frame, color='tab:purple', ls='--', alpha=.5, label='Phase reference' if index == 0 else None)
        ax.set(xlabel='Frame index (zero based)', ylabel='Feature score / relative ranking', title=result['dicom_id'])
        if labels:
            ax.legend(loc='upper right', ncol=3)
        fig.tight_layout()
        fig.savefig(destination/filename, dpi=130)
        plt.close(fig)


def comparison(display: np.ndarray, result: dict, destination: Path) -> None:
    methods = list(result['methods'])
    parallel = 'selection_windows' in result
    if parallel:
        count = max(1, max(len(m['selected_frames']) for m in result['methods'].values()))
        fig, axes = plt.subplots(count, len(methods), figsize=(2.7*len(methods), 2.7*count), squeeze=False)
        axes = axes.T
    else:
        fig, axes = plt.subplots(len(methods), 3, figsize=(10, 3*len(methods)), squeeze=False)
    for row, method in enumerate(methods):
        frames = result['methods'][method]['selected_frames'][:len(axes[row])]
        for col, ax in enumerate(axes[row]):
            ax.axis('off')
            if col < len(frames):
                frame = frames[col]
                ax.imshow(display[frame['frame_index']], cmap='gray', vmin=0, vmax=255)
                ax.set_title(f"{method}\nFrame {frame['frame_index']} | {frame['score']:.3f}", fontsize=9)
            elif col == 0:
                ax.set_title(method+'\nRejected / review', fontsize=9)
    fig.tight_layout()
    fig.savefig(destination/'algorithm_comparison.png', dpi=100)
    plt.close(fig)
