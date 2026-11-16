import json

import cv2
import numpy as np
import pytest
from skimage.filters import frangi

from keyframe.algorithms.cheap_vessel import blackhat_vessel_score, sobel_vessel_score
from keyframe.algorithms.frangi_fast import fast_frangi
from keyframe.algorithms.motion import cycle_windows, temporal_stability
from keyframe.config import load_config
from keyframe.report import export
from keyframe.round2 import METHOD_FILTERS, analyze_round2
from keyframe.selector import select_frames
from keyframe.synthetic import phantom, write_dicom


@pytest.mark.parametrize('black_ridges', [True, False])
def test_optimized_frangi_matches_fixed_gamma_and_reuses_scales(black_ridges):
    rng = np.random.default_rng(4)
    image = (.7 + .01*rng.normal(size=(64, 80))).astype(np.float32)
    cv2.line(image, (12, 13), (65, 46), .2, 3)
    if not black_ridges:
        image = 1-image
    reference = frangi(image, sigmas=[1,2,3], gamma=.03, black_ridges=black_ridges)
    cache = {}
    two = fast_frangi(image, [1,2], .03, black_ridges, cache)
    three = fast_frangi(image, [1,2,3], .03, black_ridges, cache)
    assert three.dtype == np.float32
    assert three.max() > .05
    np.testing.assert_allclose(three, reference, atol=3e-6, rtol=1e-5)
    np.testing.assert_array_equal(three, np.maximum(two, cache[3]))
    np.testing.assert_array_equal(three, fast_frangi(image, [1,2,3], .03, black_ridges))


def test_optimized_frangi_rejects_opposite_polarity_on_a_straight_ridge():
    x = np.arange(129, dtype=np.float32) - 64
    bright = np.tile(.4 + .45 * np.exp(-x*x/8), (96, 1))
    dark_response = fast_frangi(bright, [1,2,3], .03, True)
    bright_response = fast_frangi(bright, [1,2,3], .03, False)
    assert dark_response[:, 64].max() == 0
    assert bright_response[:, 64].min() > .9


def test_cheap_filters_respect_mask_roi_and_polarity():
    background = np.full((60, 80), .8, np.float32)
    background[:, 60:] = .2  # Unchanged edge should be suppressed by the Sobel mask.
    frame = background.copy()
    frame[10:50, 29:32] = .15
    masked, stats = sobel_vessel_score(frame, background=background)
    assert masked[:, 58:62].max() == 0
    assert masked[10:50, 27:33].max() > .2 and stats['score'] > 0
    roi, _ = sobel_vessel_score(frame, roi=[24, 8, 15, 44], background=background)
    assert roi.shape == (44, 15)
    dark, values = blackhat_vessel_score(frame, size=9)
    bright, _ = blackhat_vessel_score(1-frame, size=9, black_ridges=False)
    np.testing.assert_allclose(dark, bright, atol=1e-7)
    assert dark[20:40, 29:32].mean() > .6 and values['area'] > 0


def test_temporal_stability_marks_filling_change_and_low_texture():
    rng = np.random.default_rng(6)
    texture = cv2.GaussianBlur(rng.uniform(.2,.8,(96,96)).astype(np.float32), (0,0), 1)
    shifted = cv2.warpAffine(texture, np.float32([[1,0,2],[0,1,0]]), (96,96), borderMode=cv2.BORDER_REFLECT)
    images = np.stack([texture, texture+.03, shifted+.03, shifted, np.full_like(texture,.4), np.full_like(texture,.4)])
    result = temporal_stability(images, [(0,5)], np.array([0,0,0,.04,.04,.04]), load_config()['motion'], 0)
    assert result['stability_status'][1] == 'estimated'
    assert result['stability_flow'][1] < .001
    assert result['stability_flow'][2] > .008
    assert result['stability_status'][3] == 'filling_change'
    assert result['stability_status'][5] == 'insufficient_tracks'
    assert np.isnan(result['stability_flow'][[3,5]]).all()


def test_periodic_windows_reserve_distinct_cycles():
    minima = np.zeros(31, bool)
    minima[[5, 15, 25]] = True
    windows = cycle_windows([(0,30)], minima, list(np.arange(31)*80.))
    assert windows == [(0,10),(11,20),(21,30)]
    assert cycle_windows([(0,30)], minima, None) == [(0,30)]
    scores = np.zeros(31)
    scores[[5,15,25]] = [1., .9, .8]
    identical = np.zeros((31, 8, 8))
    cfg = load_config()['selection']
    assert len(select_frames(scores, identical, [(0,30)], cfg)) == 1
    assert [x['frame_index'] for x in select_frames(scores, identical, [(0,30)], cfg, preferred_windows=windows)] == [5,15,25]


def test_round2_report_matrix_and_annotations(tmp_path):
    source = tmp_path/'sample.dcm'
    write_dicom(source, phantom(64,25))
    original = source.read_bytes()
    cfg = load_config()
    cfg['contrast']['min_dynamic'] = .003
    cfg['selection']['min_vessel_score'] = 0
    annotation = {'dicom_id': 'sample_001', 'acceptable_frames': list(range(9,17)), 'preferred_frame': 12}
    analysis = analyze_round2(source, cfg, annotation=annotation, matrix_repeat=1)
    assert list(analysis.result['methods']) == list(METHOD_FILTERS)
    assert all('top_k_hit' in method['evaluation'] for method in analysis.result['methods'].values())
    assert analysis.result['selected_frames']
    assert 'stability_status' in analysis.features
    variants = analysis.result['experiment_matrix']['variants']
    assert variants['frangi_256_s3_all']['computed_frames'] == 25
    assert variants['frangi_256_s3_candidate']['computed_frames'] < 25
    assert variants['frangi_fast_256_s3_candidate']['same_ranked_frames_as_reference']
    output = tmp_path/'report'
    report = export(analysis, output).read_text()
    assert 'Round 2 comparison' in report and 'SYNTHETIC^PHANTOM' not in report
    frame = analysis.result['selected_frames'][0]['frame_index']
    for name in ('frangi','frangi_fast','sobel','blackhat'):
        assert (output/'visualizations'/f'{name}_{frame:04}.png').is_file()
    assert json.loads((output/'benchmark.json').read_text())['method_timings']['contrast_sobel']['filter_cpu_seconds'] > 0
    assert source.read_bytes() == original


def test_no_candidate_retains_explicit_failure(tmp_path):
    source = tmp_path/'flat.dcm'
    write_dicom(source, np.full((6,32,32), 900, dtype=np.uint16))
    analysis = analyze_round2(source, load_config())
    assert all(m['reason'] == 'no_filling_interval' and not m['selected_frames'] for m in analysis.result['methods'].values())
    assert not analysis.result['evaluation_available']
