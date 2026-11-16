from contextlib import contextmanager
from http.client import HTTPConnection
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlencode
from zipfile import ZipFile
import io
import cv2
import numpy as np
import pydicom
import pytest
from keyframe.config import load_config
from keyframe.synthetic import phantom, write_dicom
from keyframe.web_server import make_server


@contextmanager
def running_server(tmp_path):
    cfg=load_config()
    cfg['contrast']['min_dynamic']=.003
    server=make_server(tmp_path/'runs',cfg,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        server.app.close()
        thread.join()


def request(server, method, path, data=None, token=True, extra=None):
    client=HTTPConnection('127.0.0.1',server.server_port,timeout=30)
    headers={'X-Keyframe-Token':server.app.token} if token else {}
    headers.update(extra or {})
    client.request(method,path,body=data,headers=headers)
    response=client.getresponse()
    status,content,meta=response.status,response.read(),dict(response.getheaders())
    client.close()
    return status,content,meta


def wait_job(server, job_id):
    for _ in range(300):
        status,raw,_=request(server,'GET','/api/jobs/'+job_id)
        assert status==200
        job=json.loads(raw)
        if job['state'] in ('done','error'):
            return job
        time.sleep(.05)
    raise AssertionError('Analysis did not complete')


def test_web_analysis_and_downloads(tmp_path):
    source=tmp_path/'sample.dcm'
    write_dicom(source,phantom(64,25))
    with running_server(tmp_path) as server:
        status,html,headers=request(server,'GET','/')
        assert status==200 and b'__LOCAL_TOKEN__' not in html
        assert server.app.token.encode() in html
        assert headers['Cache-Control']=='no-store'
        health_status, health, _ = request(server, 'GET', '/api/health')
        assert health_status == 200 and 'round2' in json.loads(health)['workflows']
        assert 'filling_validation' in json.loads(health)['round2_features']
        assert b'id="fillingFallback" type="checkbox" checked' in html
        for asset in ('app.js','style.css'):
            assert request(server,'GET','/'+asset)[0]==200
        query=urlencode({'method':'multi_feature','compare':'true','size':128,'top_k':3,'repeat':3})
        status,raw,_=request(server,'POST','/api/jobs?'+query,source.read_bytes())
        assert status==202
        job_id=json.loads(raw)['id']
        job=wait_job(server,job_id)
        assert job['state']=='done',job
        assert job['result']['selected_frames']
        assert 'coronary_image' not in job['result']
        assert 'coronary_side' not in job['benchmark']['stages']
        assert len(job['result']['methods'])==8
        assert job['repeated']['summary']['repeat']==3
        assert job['repeated']['warmup']==1
        assert not (server.app.output/job_id/'input.dcm').exists()
        for file in ('report.html','results.json','scores.csv','benchmark.json','repeated_benchmark.json'):
            status,body,_=request(server,'GET',f'/artifacts/{job_id}/{file}')
            assert status==200
            assert b'SYNTHETIC^PHANTOM' not in body
        status,body,headers=request(server,'GET',f'/artifacts/{job_id}/all.zip')
        assert status==200 and 'attachment' in headers['Content-Disposition']
        with ZipFile(io.BytesIO(body)) as archive:
            assert 'report.html' in archive.namelist()
            assert 'repeated_benchmark.json' in archive.namelist()
            assert 'input.dcm' not in archive.namelist()
        assert request(server,'GET',f'/artifacts/{job_id}/../input.dcm')[0]==403
        assert request(server,'GET',f'/artifacts/{job_id}/%2e%2e/config.yaml')[0]==403
        assert request(server,'GET','/api/jobs/'+'0'*32)[0]==404


def test_web_invalid_input_and_request_guards(tmp_path):
    with running_server(tmp_path) as server:
        assert request(server,'POST','/api/jobs',b'bad',token=False)[0]==403
        assert request(server,'GET','/',extra={'Host':'evil.example'})[0]==403
        assert request(server,'POST','/api/jobs?size=42',b'bad')[0]==400
        assert request(server,'POST','/api/jobs?matrix_repeat=1',b'bad')[0]==400
        assert request(server,'POST','/api/jobs?workflow=round2&blackhat_size=7',b'bad')[0]==400
        assert request(server,'POST','/api/jobs?workflow=round2&end_diastole=bogus',b'bad')[0]==400
        assert request(server,'POST','/api/jobs',b'')[0]==413
        server.app.max_upload=2
        assert request(server,'POST','/api/jobs',b'too large')[0]==413
        server.app.max_upload=10000
        status,raw,_=request(server,'POST','/api/jobs',b'not a DICOM')
        assert status==202
        job=wait_job(server,json.loads(raw)['id'])
        assert job['state']=='error'
        assert 'Cannot parse DICOM' in job['message']
        assert not (server.app.output/job['id']/'input.dcm').exists()
        # After failure the next upload must not remain locked out.
        assert request(server,'POST','/api/jobs',b'another invalid file')[0]==202


@pytest.mark.parametrize('compare, matrix_repeat', [(True, 1), (False, 0)])
def test_round2_web_runs_requested_methods(tmp_path, compare, matrix_repeat):
    source = tmp_path/'sample.dcm'
    write_dicom(source, phantom(64, 25))
    ds = pydicom.dcmread(source)
    ds.SeriesDescription = 'RCA PRIVATE_SENTINEL'
    ds.save_as(source)
    with running_server(tmp_path) as server:
        query = urlencode({'workflow': 'round2', 'method': 'contrast_blackhat_motion',
                           'compare': str(compare).lower(), 'matrix_repeat': matrix_repeat,
                           'scale_count': 2, 'blackhat_size': 5, 'size': 128, 'repeat': 2,
                           'filling_fallback': 'true', 'end_diastole': 'true'})
        status, raw, _ = request(server, 'POST', '/api/jobs?'+query, source.read_bytes())
        assert status == 202
        job_id = json.loads(raw)['id']
        job = wait_job(server, job_id)
        assert job['state'] == 'done', job
        result = job['result']
        assert 'coronary_side' not in result['metadata']
        assert 'coronary_image' not in result
        assert 'coronary_side' not in job['benchmark']['stages']
        assert 'PRIVATE_SENTINEL' not in json.dumps(result)
        assert result['primary_method'] == 'contrast_blackhat_motion'
        assert len(result['methods']) == (5 if compare else 1)
        assert result['config']['round2']['blackhat_size'] == 5
        assert result['config']['round2']['optimized_sigmas'] == [1, 2]
        assert 'selection_windows' in result and 'motion_states' in result
        assert result['filling_source'] == 'global_darkening'
        assert result['cardiac_phase']['source'] in ('image_geometry_estimate', 'image_low_motion', 'unavailable')
        assert result['cardiac_phase']['max_score_bonus'] == .015
        assert 'cardiac_phase' in job['benchmark']['shared_stages']
        assert ('experiment_matrix' in result) == bool(matrix_repeat)
        summary = job['repeated']['summary']
        assert summary['method_mean_timings']['contrast_blackhat_motion']['filter_cpu_seconds'] > 0
        assert 'temporal_stability' in summary['stage_cpu_mean_seconds']
        assert all('matrix_seconds' not in run for run in job['repeated']['runs'])
        frame = result['selected_frames'][0]['frame_index']
        base = f'/artifacts/{job_id}/'
        status, report, _ = request(server, 'GET', base+'report.html')
        assert status == 200 and b'Score timeline' in report
        assert b'Cardiac phase references' in report
        assert b'coronary-side' not in report
        assert request(server, 'GET', base+f'visualizations/blackhat_{frame:04}.png')[0] == 200
        assert request(server, 'GET', base+'visualizations/final_scores_comparison.png')[0] == 200
        viewer = f'/api/jobs/{job_id}/frames/'
        assert job['viewer_frames'] == 25
        assert request(server, 'GET', viewer+'0')[0] == 200
        assert request(server, 'GET', viewer+'24')[0] == 200
        assert request(server, 'GET', viewer+'25')[0] == 404
        status, sequence_png, headers = request(server, 'GET', viewer+str(frame))
        assert status == 200 and headers['Content-Type'] == 'image/png'
        status, candidate_png, _ = request(server, 'GET', base+f'keyframes/frame_{frame:04}.png')
        assert status == 200
        sequence_image = cv2.imdecode(np.frombuffer(sequence_png, np.uint8), cv2.IMREAD_GRAYSCALE)
        candidate_image = cv2.imdecode(np.frombuffer(candidate_png, np.uint8), cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(sequence_image, candidate_image)


def test_web_serializes_processing(tmp_path,monkeypatch):
    import keyframe.web_server as module
    original=module.analyze
    entered,release=threading.Event(),threading.Event()
    def paused(*args,**kwargs):
        entered.set()
        release.wait(10)
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'analyze',paused)
    with running_server(tmp_path) as server:
        try:
            status,raw,_=request(server,'POST','/api/jobs',b'invalid DICOM')
            assert status==202 and entered.wait(5)
            assert request(server,'POST','/api/jobs',b'other')[0]==409
        finally:
            release.set()
        assert wait_job(server,json.loads(raw)['id'])['state']=='error'
