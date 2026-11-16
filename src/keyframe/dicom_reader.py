"""Strict grayscale DICOM decoding. No identifying tags are returned."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut

class DicomError(ValueError):
    pass

@dataclass
class DicomSequence:
    pixels: np.ndarray
    metadata: dict[str, Any]
    raw_pixels: np.ndarray  # decoded stored values, before modality/polarity transforms


def read_dicom(path: str | Path) -> DicomSequence:
    try:
        ds = pydicom.dcmread(path)
    except Exception:
        raise DicomError('Cannot parse DICOM (invalid format or unreadable file)') from None
    required = ('Rows', 'Columns', 'BitsAllocated', 'BitsStored', 'HighBit', 'PixelRepresentation')
    if any(not hasattr(ds, k) for k in required):
        raise DicomError('Missing required pixel metadata')
    if int(ds.PixelRepresentation) not in (0, 1) or int(ds.HighBit) != int(ds.BitsStored)-1:
        raise DicomError('Unsupported PixelRepresentation/HighBit')
    photo = str(getattr(ds, 'PhotometricInterpretation', ''))
    if photo not in ('MONOCHROME1', 'MONOCHROME2') or int(getattr(ds, 'SamplesPerPixel', 1)) != 1:
        raise DicomError('Only single-sample MONOCHROME1/2 supported')
    if hasattr(ds, 'PerFrameFunctionalGroupsSequence') or hasattr(ds, 'SharedFunctionalGroupsSequence'):
        raise DicomError('Enhanced functional-group pixel transforms not supported')
    if int(getattr(ds, 'BitsAllocated', 0)) not in (8, 16) or not 1 <= int(getattr(ds, 'BitsStored', 0)) <= int(ds.BitsAllocated):
        raise DicomError('Unsupported BitsAllocated/BitsStored')
    if 'PixelData' not in ds or not hasattr(ds, 'file_meta') or 'TransferSyntaxUID' not in ds.file_meta:
        raise DicomError('Missing PixelData or TransferSyntaxUID')
    try:
        raw = ds.pixel_array
        frames = raw[None] if raw.ndim == 2 else raw
        if frames.ndim != 3 or len(frames) != int(getattr(ds, 'NumberOfFrames', 1)):
            raise DicomError('Inconsistent pixel dimensions/frame count')
        pixels = apply_modality_lut(frames, ds).astype(np.float32)
    except DicomError:
        raise
    except Exception:
        raise DicomError('Pixel decoding/transform failed; compressed data may require pylibjpeg or pyjpegls') from None
    if not np.isfinite(pixels).all():
        raise DicomError('Nonfinite transformed pixels')
    if photo == 'MONOCHROME1':
        pixels = pixels.max() + pixels.min() - pixels
    n = len(pixels)
    times = None
    if hasattr(ds, 'FrameTimeVector'):
        vector = np.atleast_1d(np.asarray(ds.FrameTimeVector, dtype=float))
        if len(vector) != n or not np.isfinite(vector).all() or np.any(vector < 0):
            raise DicomError('Invalid FrameTimeVector')
        times = (np.cumsum(vector) - vector[0]).tolist()
    else:
        dt = float(getattr(ds, 'FrameTime', 0))
        fps = float(getattr(ds, 'CineRate', getattr(ds, 'RecommendedDisplayFrameRate', 0)))
        dt = dt if dt > 0 else (1000 / fps if fps > 0 else 0)
        if not np.isfinite(dt) or not np.isfinite(fps):
            raise DicomError('Nonfinite frame timing')
        if dt:
            times = (np.arange(n) * dt).tolist()
    metadata = {'frames': n, 'rows': int(ds.Rows), 'columns': int(ds.Columns),
                'bits_allocated': int(ds.BitsAllocated), 'bits_stored': int(ds.BitsStored),
                'pixel_representation': int(ds.PixelRepresentation), 'photometric_interpretation': photo,
                'transfer_syntax': str(ds.file_meta.TransferSyntaxUID), 'modality': str(getattr(ds, 'Modality', '')),
                'frame_times_ms': times, 'index_base': 0}
    return DicomSequence(pixels, metadata, frames)
