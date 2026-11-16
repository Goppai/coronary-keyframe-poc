"""Deterministic branching phantom; tests mechanics, not coronary realism."""
from pathlib import Path
import cv2
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, XRayAngiographicImageStorage, generate_uid


def phantom(size: int = 256, frames: int = 100, seed: int = 7, contrast: bool = True, peaks: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)/size
    background = .72+.08*xx+.03*np.cos(yy*9)
    vessel = np.zeros((size, size), np.float32)
    paths = [[(.5,.1),(.47,.3),(.55,.5),(.48,.7),(.6,.9)],[(.49,.3),(.3,.45),(.2,.7)],[(.53,.48),(.73,.6),(.85,.8)]]
    for points in paths:
        coords = np.array([(round(x*size), round(y*size)) for x,y in points], np.int32)
        cv2.polylines(vessel, [coords], False, 1., max(2, size//90), cv2.LINE_AA)
    vessel = cv2.GaussianBlur(vessel, (0, 0), max(.6, size/512))
    output = np.empty((frames, size, size), np.uint16)
    for t in range(frames):
        centers = np.linspace(.35, .75, peaks) if peaks > 1 else [.5]
        filling = max(np.exp(-.5*((t/frames-c)/(.10 if peaks == 1 else .055))**2) for c in centers) if contrast else 0
        moved = cv2.warpAffine(vessel, np.float32([[1,0,np.sin(t*.32)*size*.004],[0,1,0]]), (size,size))
        image = background - .5*filling*moved + rng.normal(0, .003, (size,size))
        output[t] = np.clip(image*4095, 0, 4095).astype(np.uint16)
    return output


def write_dicom(path: str | Path, pixels: np.ndarray, photo: str = 'MONOCHROME2', signed: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = XRayAngiographicImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0'*128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = 'XA'
    ds.PatientName = 'SYNTHETIC^PHANTOM'
    ds.PatientID = 'SYNTHETIC_ONLY'
    ds.Rows, ds.Columns = pixels.shape[-2:]
    ds.NumberOfFrames = len(pixels) if pixels.ndim == 3 else 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photo
    ds.BitsAllocated = pixels.dtype.itemsize*8
    ds.BitsStored = 12 if ds.BitsAllocated == 16 else 8
    ds.HighBit = ds.BitsStored-1
    ds.PixelRepresentation = int(signed)
    ds.FrameTime = 66.6667
    ds.PixelData = pixels.tobytes()
    ds.save_as(path, enforce_file_format=True)
