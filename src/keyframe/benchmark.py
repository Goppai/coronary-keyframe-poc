from contextlib import contextmanager
import os
import platform
import threading
import time
from importlib.metadata import version
import cv2
import numpy as np
import psutil

class Monitor:
    """Process RSS sampled every 10 ms; includes native allocations, may miss spikes."""
    def __enter__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.baseline = self.peak
        self.stop = threading.Event()
        self.start = time.perf_counter()
        cpu = self.process.cpu_times()
        self.cpu_start = cpu.user+cpu.system
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stop.wait(.01):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.elapsed = time.perf_counter()-self.start
        cpu = self.process.cpu_times()
        self.cpu_seconds = cpu.user+cpu.system-self.cpu_start
        self.peak = max(self.peak, self.process.memory_info().rss)

    def result(self) -> dict:
        return {'total_seconds': self.elapsed, 'cpu_seconds': self.cpu_seconds,
                'cpu_percent_one_core_basis': self.cpu_seconds/max(self.elapsed, 1e-9)*100,
                'peak_rss_mb_sampled': self.peak/2**20, 'baseline_rss_mb': self.baseline/2**20,
                'memory_sampling_ms': 10}

@contextmanager
def stage(timings: dict, name: str, cpu_timings: dict | None = None):
    start = time.perf_counter()
    cpu_start = time.process_time() if cpu_timings is not None else None
    yield
    if cpu_timings is not None:
        cpu_timings[name] = time.process_time() - cpu_start
    timings[name] = time.perf_counter()-start


def environment() -> dict:
    return {'python': platform.python_version(), 'platform': platform.platform(), 'machine': platform.machine(),
            'opencv_threads': cv2.getNumThreads(), 'logical_cpus': os.cpu_count(), 'processor': platform.processor(),
            'libraries': {k: version(k) for k in ('numpy', 'opencv-python-headless', 'pydicom', 'scikit-image', 'scipy')}}


def summarize(runs: list[dict]) -> dict:
    times = [r['total_seconds'] for r in runs]
    summary = {'repeat': len(runs), 'mean_seconds': float(np.mean(times)), 'p50_seconds': float(np.percentile(times, 50)),
            'p95_seconds': float(np.percentile(times, 95)), 'mean_fps': float(np.mean([r['frames']/r['total_seconds'] for r in runs])),
            'mean_cpu_percent': float(np.mean([r['cpu_percent_one_core_basis'] for r in runs])),
            'peak_rss_mb_sampled': max(r['peak_rss_mb_sampled'] for r in runs),
            'stage_mean_seconds': {key: float(np.mean([r['stages'][key] for r in runs])) for key in runs[0]['stages']}}
    if 'stage_cpu_seconds' in runs[0]:
        summary['stage_cpu_mean_seconds'] = {
            stage: float(np.mean([run['stage_cpu_seconds'][stage] for run in runs]))
            for stage in runs[0]['stage_cpu_seconds']
        }
    if 'method_timings' in runs[0]:
        summary['method_mean_timings'] = {
            method: {key: float(np.mean([run['method_timings'][method][key] for run in runs])) for key in timings}
            for method, timings in runs[0]['method_timings'].items()
        }
    return summary
