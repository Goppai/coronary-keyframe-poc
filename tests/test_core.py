import json
import numpy as np
import pydicom
import pytest
from keyframe.config import load_config
from keyframe.dicom_reader import read_dicom, DicomError
from keyframe.preprocessing import preprocess
from keyframe.synthetic import phantom, write_dicom
from keyframe.algorithms.contrast import detect_contrast
from keyframe.algorithms.sharpness import sharpness
from keyframe.algorithms.motion import motion_features
from keyframe.algorithms.vessel import enhance
from keyframe.selector import select_frames
from keyframe.evaluation import evaluate, aggregate
from keyframe.pipeline import analyze
from keyframe.report import export

@pytest.mark.parametrize('dtype,signed', [('uint8',False),('uint16',False),('int16',True)])
@pytest.mark.parametrize('photo', ['MONOCHROME1','MONOCHROME2'])
def test_decode(tmp_path, dtype, signed, photo):
    x = np.arange(128).reshape(2,8,8).astype(dtype)
    if signed:
        x -= 64
    path = tmp_path/'a.dcm'
    write_dicom(path,x,photo,signed)
    ds = pydicom.dcmread(path)
    ds.RescaleSlope = 2
    ds.RescaleIntercept = -9
    ds.PatientName = 'PRIVATE_SENTINEL'
    ds.save_as(path)
    seq = read_dicom(path)
    expected = x.astype(float)*2-9
    if photo == 'MONOCHROME1':
        expected = expected.max()+expected.min()-expected
    np.testing.assert_allclose(seq.pixels, expected)
    np.testing.assert_array_equal(seq.raw_pixels, x)
    assert 'PRIVATE' not in json.dumps(seq.metadata)
    assert len(seq.metadata['frame_times_ms']) == 2


def test_single_and_timing(tmp_path):
    path = tmp_path/'a.dcm'
    write_dicom(path,np.ones((8,9),np.uint16))
    assert read_dicom(path).pixels.shape == (1,8,9)
    ds = pydicom.dcmread(path)
    ds.FrameTimeVector = [0]
    ds.save_as(path)
    assert read_dicom(path).metadata['frame_times_ms'] == [0]


def test_bad_inputs(tmp_path):
    path = tmp_path/'a.dcm'
    path.write_bytes(b'not a dicom')
    with pytest.raises(DicomError):
        read_dicom(path)
    write_dicom(path,np.ones((2,8,8),np.uint16))
    ds = pydicom.dcmread(path)
    ds.PhotometricInterpretation = 'RGB'
    ds.save_as(path)
    with pytest.raises(DicomError,match='MONOCHROME'):
        read_dicom(path)


def test_corrupt_pixel_data(tmp_path):
    path = tmp_path/'a.dcm'
    write_dicom(path,np.ones((2,8,8),np.uint16))
    ds = pydicom.dcmread(path)
    ds.PixelData = b'\0\0'
    ds.save_as(path)
    with pytest.raises(DicomError,match='decoding'):
        read_dicom(path)


def test_shared_scale_and_roi():
    cfg = load_config()['preprocessing']
    x = np.stack([np.full((20,30),2.),np.full((20,30),4.)])
    a = preprocess(x,cfg)
    assert a.analysis[0].mean() < a.analysis[1].mean()
    cfg['roi'] = [29,0,5,5]
    with pytest.raises(ValueError):
        preprocess(x,cfg)


@pytest.mark.parametrize('reference',['median','low_motion','adaptive'])
def test_two_peaks_and_absence(reference):
    cfg = load_config()['contrast']
    cfg['reference'] = reference
    images = np.full((80,32,32),.8,np.float32)
    for center in [22,58]:
        for t in range(80):
            images[t,8:24,8:24] -= .4*np.exp(-.5*((t-center)/4)**2)
    result = detect_contrast(images,cfg)
    assert len(result.candidate_intervals) == 2
    assert not detect_contrast(np.full_like(images,.8),cfg).candidate_intervals
    assert result.features['hist_00'].shape == (80,)


def test_filters_and_blur():
    import cv2
    image = np.full((64,64),.8,np.float32)
    image[8:56,29:34] = .15
    cfg = load_config()['vessel']
    for method in ['frangi','sato']:
        response = enhance(image,method,cfg)
        assert response[:,29:34].mean() > response[:,:10].mean()
    assert sharpness(image)['gradient'] > sharpness(cv2.GaussianBlur(image,(0,0),3))['gradient']


def test_motion_translation():
    import cv2
    rng = np.random.default_rng(3)
    a = cv2.GaussianBlur(rng.random((96,96)).astype(np.float32),(0,0),1)
    b = cv2.warpAffine(a,np.float32([[1,0,2],[0,1,0]]),(96,96))
    result = motion_features(np.stack([a,b]),load_config()['motion'])
    assert result['tracked_points'][1] > 0
    assert result['flow_displacement'][1] > .005


def test_selection_and_reference():
    scores = np.array([0,.3,1,.3,0,0,.2,.9,.2,0])
    images = np.random.default_rng(3).random((10,8,8)).astype(np.float32)
    cfg = load_config()['selection']
    selected = select_frames(scores,images,[(1,3),(6,8)],cfg)
    assert [r['frame_index'] for r in selected] == [2,7]
    images[7] = images[2]
    assert len(select_frames(scores,images,[(1,3),(6,8)],cfg)) == 1
    ref = evaluate(selected,{'acceptable_frames':[1,2,3], 'preferred_frame':2})
    assert ref['top1_hit'] and ref['preferred_frame_distance'] == 0
    assert evaluate([],{'acceptable_frames':[]})['correct_rejection']
    assert aggregate([ref])['top1_hit']['denominator'] == 1
    with pytest.raises(ValueError):
        evaluate([], {})


def test_end_to_end(tmp_path):
    path = tmp_path/'a.dcm'
    write_dicom(path,phantom(96,35))
    cfg = load_config()
    cfg['contrast']['min_dynamic'] = .003
    cfg['selection']['min_vessel_score'] = 0
    a = analyze(path,cfg,compare_all=True)
    assert len(a.result['methods']) == 8
    assert a.result['selected_frames']
    report = export(a,tmp_path/'output')
    assert report.exists()
    assert 'SYNTHETIC^PHANTOM' not in report.read_text()
    assert (tmp_path/'output'/'scores.csv').exists()
    assert a.benchmark['stages']['frangi'] > 0
    with pytest.raises(ValueError,match='empty'):
        export(a,tmp_path/'output')


def test_negative_pipeline(tmp_path):
    path = tmp_path/'a.dcm'
    write_dicom(path,np.full((10,32,32),1000,np.uint16))
    a = analyze(path,load_config(),compare_all=True)
    assert all(not m['selected_frames'] for m in a.result['methods'].values())
    assert a.result['contrast_confidence_heuristic'] == 0


def test_rle_compressed_and_missing_metadata(tmp_path):
    from pydicom.uid import RLELossless
    path = tmp_path/'rle.dcm'
    pixels = np.arange(2*32*32,dtype=np.uint16).reshape(2,32,32)
    write_dicom(path,pixels)
    ds = pydicom.dcmread(path)
    ds.compress(RLELossless)
    ds.save_as(path)
    np.testing.assert_array_equal(read_dicom(path).pixels,pixels)
    del ds.HighBit
    ds.save_as(path)
    with pytest.raises(DicomError, match='metadata'):
        read_dicom(path)


def test_fixed_gamma_monotonicity():
    image = np.full((64,64),.8,np.float32)
    image[8:56,29:34] = .2
    weaker = .8-(.8-image)*.3
    cfg = load_config()['vessel']
    assert enhance(image,'frangi',cfg).sum() > enhance(weaker,'frangi',cfg).sum()


def test_labelled_analysis(tmp_path):
    path=tmp_path/'a.dcm'
    write_dicom(path,np.ones((6,32,32),np.uint16)*1000)
    a=analyze(path,load_config(),annotation={'dicom_id':'sample_001','acceptable_frames':[]})
    assert a.result['methods']['multi_feature']['evaluation']['correct_rejection']
    with pytest.raises(ValueError,match='does not match'):
        analyze(path,load_config(),annotation={'dicom_id':'wrong','acceptable_frames':[]})
