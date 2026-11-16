import cv2
import numpy as np
import pytest

from keyframe.algorithms.cardiac_phase import phase_evidence
from keyframe.algorithms.filling_fallback import recover_filling
from keyframe.algorithms.motion import vessel_area_change
from keyframe.config import load_config
from keyframe.round2 import ComparisonContext, analyze_round2, rank
from keyframe.synthetic import write_dicom


def faint_filling():
    images = np.full((36, 96, 96), .7, np.float32)
    for index, frame in enumerate(images):
        pulse = np.exp(-.5*((index-18)/4)**2)
        cv2.line(frame, (35, 32), (43, 67), .7-.31*pulse, 2)
        cv2.line(frame, (39, 47), (62, 60), .7-.25*pulse, 1)
    return images


@pytest.mark.parametrize('black_ridges', [True, False])
def test_fallback_recovers_faint_vessels_and_rejects_static_pattern(black_ridges):
    images = faint_filling()
    if not black_ridges:
        images = 1-images
    cfg = load_config()['contrast']
    intervals, _, _ = recover_filling(images, cfg, black_ridges=black_ridges)
    assert intervals and intervals[0][0] < 18 < intervals[0][1]
    static = np.repeat(images[18:19], len(images), axis=0)
    assert not recover_filling(static, cfg, black_ridges=black_ridges)[0]


def test_fallback_in_round2(tmp_path):
    source = tmp_path/'synthetic.dcm'
    write_dicom(source, (faint_filling()*4095).astype(np.uint16))
    config = load_config()
    config['contrast']['min_dynamic'] = .2
    config['preprocessing'].update(percentiles=[0, 100], denoise_sigma=0)
    old = analyze_round2(source, config, method='contrast_frangi_fast')
    assert old.result['candidate_intervals'] == []
    config['round2']['filling_fallback'] = True
    new = analyze_round2(source, config, method='contrast_frangi_fast')
    assert new.result['filling_source'] == 'local_ridges'
    assert new.result['selected_frames']
    assert 'fallback_ridge_score' in new.features
    assert 'filling_fallback' in new.benchmark['shared_stages']


def test_local_filling_precedes_unrelated_late_darkening(tmp_path):
    source = tmp_path/'synthetic.dcm'
    images = faint_filling()
    images[27:] -= np.linspace(.02, .2, len(images)-27, dtype=np.float32)[:, None, None]
    write_dicom(source, (images*4095).astype(np.uint16))
    config = load_config()
    config['preprocessing'].update(percentiles=[0, 100], denoise_sigma=0)
    baseline = analyze_round2(source, config, method='contrast_frangi_fast').result
    assert baseline['candidate_intervals'][0][0] >= 27
    config['round2']['filling_fallback'] = True
    checked = analyze_round2(source, config, method='contrast_frangi_fast').result
    assert checked['global_candidate_intervals'] == baseline['candidate_intervals']
    assert checked['filling_source'] == 'local_ridges_early'
    assert checked['selected_frames'][0]['frame_index'] in range(16, 22)


def test_image_phase_requires_repeated_signed_motion():
    times = np.arange(33)*66.67
    strain = np.zeros(33)
    for i in (10, 23):
        strain[i-1:i+3] = [.01, .02, -.02, -.01]
    minima = np.zeros(33, bool)
    minima[[7, 17, 27]] = True
    result, features = phase_evidence(times, minima, [(0,32)], [(0,11),(12,22),(23,32)], strain)
    assert result['source'] == 'image_geometry_estimate'
    assert result['reference_frames'] == [10, 23]
    np.testing.assert_array_equal(features['end_diastole_score'][[10,23]], [.5, .5])
    strain[22] = np.nan
    result, features = phase_evidence(times, minima, [(0,32)], [(0,11),(12,22),(23,32)], strain)
    assert result['source'] == 'image_low_motion'
    assert result['reference_frames'] == []
    assert np.isnan(features['end_diastole_score']).all()


def test_ridge_track_area_is_translation_invariant():
    image = np.full((100, 100), .8, np.float32)
    points = np.array([(x, y) for y in (20, 40, 60, 80) for x in (20, 40, 60, 80)], np.float32)
    for point in points:
        cv2.circle(image, tuple(point.astype(int)), 2, .2, -1)
    assert abs(vessel_area_change(image, points, points + 3, True)) < 1e-5
    assert vessel_area_change(image, points, points * 1.02, True) == pytest.approx(2*np.log(1.02), abs=1e-5)


def test_phase_bonus_promotes_a_near_tie_without_changing_filling_gate():
    cfg = load_config()
    cfg['round2']['score_smooth_sigma'] = 0
    cfg['selection']['min_frame_distance'] = 1
    vessels = np.array([0., 1., 0., .97, 0., 0.])
    mask = np.array([False, True, True, True, True, False])
    features = {'candidate': mask, 'contrast': np.where(mask, .8, 0.)}
    images = np.random.default_rng(2).random((6, 8, 8))
    context = ComparisonContext(images, images[0], features, [(1, 4)], [(1, 4)], cfg)
    raw, _, baseline = rank(context, vessels)
    features['end_diastole_bonus'] = np.array([.015, 0., 0., .015, 0., .015])
    new_raw, final, selected = rank(context, vessels)
    assert baseline['selected_frames'][0]['frame_index'] == 1
    assert selected['selected_frames'][0]['frame_index'] == 3
    assert selected['selected_frames'][0]['end_diastole_bonus'] == .015
    np.testing.assert_array_equal(raw, new_raw)
    np.testing.assert_array_equal(final[~mask], [0., 0.])
