"""Loopback-only workbench. Files are transferred only to this local process."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import shutil
import threading
import time
from urllib.parse import parse_qs, unquote, urlsplit
import webbrowser

import cv2
import numpy as np

from .benchmark import summarize
from .config import load_config
from .dicom_reader import DicomError
from .pipeline import analyze
from .report import export, write_json
from .round2 import METHOD_FILTERS, analyze_round2
from .scoring import METHODS

ASSETS = Path(__file__).parent/'web'


class Workbench:
    def __init__(self, output: Path, config: dict, max_upload: int = 1024*2**20):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.max_upload = max_upload
        self.token = secrets.token_urlsafe(32)
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.gate = threading.Semaphore(1)
        # matplotlib and large DICOM processing are deliberately serialized.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='keyframe')

    def update(self, job_id: str, **values) -> None:
        with self.lock:
            self.jobs[job_id].update(values)

    def snapshot(self, job_id: str) -> dict | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return deepcopy(job) if job is not None else None

    def process(self, job_id: str, options: dict) -> None:
        root = self.output/job_id
        source = root/'input.dcm'
        cfg = deepcopy(self.config)
        cfg['preprocessing']['max_size'] = options['size']
        cfg['selection']['top_k'] = options['top_k']
        sample_id = 'sample_' + job_id[:12]
        if options['workflow'] == 'round2':
            if options['blackhat_size']:
                cfg['round2']['blackhat_size'] = options['blackhat_size']
            if options['scale_count']:
                cfg['round2']['optimized_sigmas'] = list(range(1, options['scale_count'] + 1))
            for key in ('filling_fallback', 'end_diastole'):
                if options[key] is not None:
                    cfg['round2'][key] = options[key] == 'true'

        def run_analysis(matrix_repeat: int = 0):
            if options['workflow'] == 'round2':
                return analyze_round2(source, cfg, sample_id, matrix_repeat=matrix_repeat,
                                      method=options['method'], compare_all=options['compare'])
            return analyze(source, cfg, options['method'], options['compare'], sample_id)

        start = time.perf_counter()
        try:
            message = ('正在分析并运行参数矩阵，矩阵可能需要较长时间…' if options['matrix_repeat']
                       else '正在解码并计算逐帧特征…')
            self.update(job_id, state='running', message=message)
            analysis = run_analysis(options['matrix_repeat'])
            result, single = analysis.result, analysis.benchmark
            self.update(job_id, message='正在准备序列阅片数据…')
            viewer_start = time.perf_counter()
            display = analysis.prepared.display
            viewer_frames = len(display)
            np.save(root/'viewer_frames.npy', display, allow_pickle=False)
            del display
            single['viewer_export_seconds'] = time.perf_counter() - viewer_start
            self.update(job_id, message='正在生成关键帧、曲线与 HTML 报告…')
            export(analysis, root/'artifacts')
            del analysis
            repeated = None
            if options['repeat'] > 1:
                runs = []
                for i in range(options['repeat']):
                    self.update(job_id, message=f'重复性能测试 {i+1}/{options["repeat"]}（首次分析作为预热）…')
                    measured = run_analysis()
                    runs.append(measured.benchmark)
                    del measured
                repeated = {'warmup': 1, 'runs': runs, 'summary': summarize(runs),
                            'scope': 'Decode through keyframe selection. First analysis, parameter matrix and export are excluded.'}
                write_json(root/'artifacts'/'repeated_benchmark.json', repeated)
            self.update(job_id, message='正在打包输出文件…')
            shutil.make_archive(str(root/'artifacts'), 'zip', root/'artifacts')
            self.update(job_id, state='done', message='分析完成', result=result, benchmark=single,
                        repeated=repeated, viewer_frames=viewer_frames, elapsed_seconds=time.perf_counter()-start)
        except Exception as exc:
            # Library exception text may contain patient tags/paths. Do not expose it.
            reason = str(exc) if isinstance(exc, DicomError) else '分析失败：输入可能不受支持或资源不足，请检查 DICOM、配置及可用内存。'
            self.update(job_id, state='error', message=reason, error_type=type(exc).__name__)
        finally:
            try:
                source.unlink(missing_ok=True)
                if self.jobs[job_id]['state'] != 'done':
                    (root/'viewer_frames.npy').unlink(missing_ok=True)
            finally:
                self.gate.release()

    def close(self) -> None:
        self.executor.shutdown(wait=True)


class Handler(BaseHTTPRequestHandler):
    server_version = 'KeyframeLocal/0.1'

    @property
    def app(self) -> Workbench:
        return self.server.app

    def log_message(self, *args) -> None:
        # No request paths, uploaded filenames or medical metadata in access logs.
        pass

    def send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('X-Frame-Options', 'SAMEORIGIN')
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, value: dict, status: int = 200) -> None:
        self.send_bytes(json.dumps(value, ensure_ascii=False, allow_nan=False).encode(), 'application/json; charset=utf-8', status)

    def valid_host(self) -> bool:
        return self.headers.get('Host', '').split(':')[0] in ('127.0.0.1', 'localhost')

    def do_GET(self) -> None:
        if not self.valid_host():
            self.send_json({'error': '仅允许本机访问'}, 403)
            return
        path = unquote(urlsplit(self.path).path)
        if path == '/':
            data = (ASSETS/'index.html').read_text().replace('__LOCAL_TOKEN__', self.app.token)
            self.send_bytes(data.encode(), 'text/html; charset=utf-8')
        elif path in ('/app.js', '/style.css'):
            self.send_bytes((ASSETS/path[1:]).read_bytes(), 'text/javascript; charset=utf-8' if path.endswith('.js') else 'text/css; charset=utf-8')
        elif path == '/api/health':
            self.send_json({'status': 'ok', 'local_only': True, 'workflows': ['legacy', 'round2'],
                            'round2_features': ['filling_fallback', 'filling_validation', 'end_diastole']})
        elif match := re.fullmatch(r'/api/jobs/([a-f0-9]{32})/frames/([0-9]{1,9})', path):
            self.send_viewer_frame(match[1], int(match[2]))
        elif match := re.fullmatch(r'/api/jobs/([a-f0-9]{32})', path):
            job = self.app.snapshot(match[1])
            self.send_json(job if job else {'error': '任务不存在'}, 200 if job else 404)
        elif match := re.fullmatch(r'/artifacts/([a-f0-9]{32})/(.+)', path):
            job_id, name = match.groups()
            if not self.app.snapshot(job_id):
                self.send_json({'error': '任务不存在'}, 404)
                return
            if name == 'all.zip':
                target = self.app.output/job_id/'artifacts.zip'
            else:
                base = self.app.output/job_id/'artifacts'
                target = (base/name).resolve()
                if not target.is_relative_to(base.resolve()):
                    self.send_json({'error': '无效路径'}, 403)
                    return
            types = {'.html':'text/html; charset=utf-8', '.json':'application/json; charset=utf-8',
                     '.csv':'text/csv; charset=utf-8', '.png':'image/png', '.zip':'application/zip', '.txt':'text/plain; charset=utf-8'}
            if not target.is_file() or target.suffix not in types:
                self.send_json({'error': '文件尚未生成或不存在'}, 404)
                return
            # Stream exports; never read a potentially large ZIP into RAM.
            self.send_response(200)
            self.send_header('Content-Type', types[target.suffix])
            self.send_header('Content-Length', str(target.stat().st_size))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'SAMEORIGIN')
            self.send_header('Referrer-Policy', 'no-referrer')
            if target.suffix == '.zip' or parse_qs(urlsplit(self.path).query).get('download'):
                self.send_header('Content-Disposition', f'attachment; filename="{target.name}"')
            self.end_headers()
            with target.open('rb') as stream:
                shutil.copyfileobj(stream, self.wfile)
        else:
            self.send_json({'error': '页面不存在'}, 404)

    def send_viewer_frame(self, job_id: str, index: int) -> None:
        with self.app.lock:
            job = self.app.jobs.get(job_id)
            count = job.get('viewer_frames', 0) if job and job['state'] == 'done' else 0
        if index >= count:
            self.send_json({'error': '序列帧不存在'}, 404)
            return
        try:
            frames = np.load(self.app.output/job_id/'viewer_frames.npy', mmap_mode='r', allow_pickle=False)
            ok, png = cv2.imencode('.png', frames[index], [cv2.IMWRITE_PNG_COMPRESSION, 1])
        except (OSError, ValueError, IndexError, cv2.error):
            self.send_json({'error': '序列帧读取失败'}, 500)
            return
        if not ok:
            self.send_json({'error': '序列帧编码失败'}, 500)
            return
        try:
            self.send_bytes(png.tobytes(), 'image/png')
        except (BrokenPipeError, ConnectionResetError):
            pass  # The browser may cancel a request when scrubbing to another frame.

    def do_POST(self) -> None:
        if not self.valid_host() or self.headers.get('X-Keyframe-Token') != self.app.token:
            self.send_json({'error': '请从本机首页操作；页面过期请刷新'}, 403)
            return
        if urlsplit(self.path).path != '/api/jobs':
            self.send_json({'error': '接口不存在'}, 404)
            return
        try:
            query = parse_qs(urlsplit(self.path).query)
            workflow = query.get('workflow', ['legacy'])[0]
            methods = {'legacy': METHODS, 'round2': METHOD_FILTERS}.get(workflow, ())
            default_method = 'contrast_frangi_fast' if workflow == 'round2' else 'multi_feature'
            options = {'workflow': workflow, 'method': query.get('method', [default_method])[0],
                       'top_k': int(query.get('top_k', ['3'])[0]),
                       'size': int(query.get('size', ['256'])[0]),
                       'compare': query.get('compare', ['false'])[0] == 'true',
                       'repeat': int(query.get('repeat', ['1'])[0]),
                       'matrix_repeat': int(query.get('matrix_repeat', ['0'])[0]),
                       'blackhat_size': int(query.get('blackhat_size', ['0'])[0]),
                       'scale_count': int(query.get('scale_count', ['0'])[0]),
                       'filling_fallback': query.get('filling_fallback', [None])[0],
                       'end_diastole': query.get('end_diastole', [None])[0]}
            length = int(self.headers.get('Content-Length', '0'))
            if (options['method'] not in methods or not 1 <= options['top_k'] <= 20
                    or options['size'] not in (128, 256, 512, 1024) or not 1 <= options['repeat'] <= 10
                    or options['matrix_repeat'] not in (0, 1, 2) or options['blackhat_size'] not in (0, 5, 9, 15)
                    or options['scale_count'] not in (0, 2, 3)
                    or any(options[key] not in (None, 'true', 'false') for key in ('filling_fallback', 'end_diastole'))
                    or (workflow == 'legacy' and any(options[key] == 'true' for key in ('filling_fallback', 'end_diastole')))
                    or (workflow == 'legacy' and any(options[key] for key in ('matrix_repeat', 'blackhat_size', 'scale_count')))):
                raise ValueError
            if not 0 < length <= self.app.max_upload:
                self.send_json({'error': f'空文件或文件过大（上限 {self.app.max_upload//2**20} MiB）'}, 413)
                self.close_connection = True
                return
        except ValueError:
            self.send_json({'error': '无效的分析参数'}, 400)
            self.close_connection = True
            return
        if not self.app.gate.acquire(blocking=False):
            self.send_json({'error': '已有分析正在运行，请等待完成后再点击文件'}, 409)
            self.close_connection = True
            return
        job_id = secrets.token_hex(16)
        root = self.app.output/job_id
        try:
            self.connection.settimeout(120)
            root.mkdir()
            remaining = length
            with (root/'input.dcm').open('wb') as stream:
                while remaining:
                    chunk = self.rfile.read(min(1024*1024, remaining))
                    if not chunk:
                        raise OSError('incomplete upload')
                    stream.write(chunk)
                    remaining -= len(chunk)
            with self.app.lock:
                self.app.jobs[job_id] = {'id': job_id, 'state': 'queued', 'message': '文件已接收，准备分析…', 'options': options}
            self.app.executor.submit(self.app.process, job_id, options)
        except Exception:
            self.app.gate.release()
            shutil.rmtree(root, ignore_errors=True)
            self.send_json({'error': '文件接收失败，请重试'}, 400)
            return
        self.send_json({'id': job_id, 'state': 'queued'}, 202)

def make_server(output: Path, config: dict | None = None, port: int = 8765) -> ThreadingHTTPServer:
    app = Workbench(output, config or load_config())
    try:
        server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    except Exception:
        app.close()
        raise
    server.app = app
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description='本机 DICOM 关键帧网页工作台（仅监听127.0.0.1）')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--output', type=Path, default=Path('outputs/web_runs'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--open', action='store_true', help='启动后打开默认浏览器')
    args = parser.parse_args()
    server = make_server(args.output, load_config(args.config), args.port)
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'本地关键帧工作台：{url}\n输出目录：{args.output.resolve()}\n按 Ctrl+C 停止服务。', flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n正在结束服务，等待当前分析完成…', flush=True)
    finally:
        server.server_close()
        server.app.close()
