import os
import argparse
import cv2
import numpy as np
import pydicom
import pandas as pd
from PIL import Image
from pydicom.dataset import Dataset, FileMetaDataset
from tempfile import TemporaryDirectory


def load_dicom_series(path: str) -> np.ndarray:
    """
    Load a multi-frame DICOM file and return a 4D numpy array of shape
    (num_frames, height, width, 3) as BGR images.
    Handles grayscale, RGB, and RGBA pixel data robustly, with forced syntax if needed.
    """
    from pydicom.dataset import FileMetaDataset
    from pydicom.uid import ImplicitVRLittleEndian

    try:
        ds = pydicom.dcmread(path)
    except pydicom.errors.InvalidDicomError:
        ds = pydicom.dcmread(path, force=True)

    if not hasattr(ds, 'file_meta') or ds.file_meta is None:
        ds.file_meta = FileMetaDataset()
    if not hasattr(ds.file_meta, 'TransferSyntaxUID'):
        ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

    if not hasattr(ds, 'PixelData'):
        raise ValueError("DICOM file has no pixel data.")

    arr = ds.pixel_array

    if arr.ndim == 2:
        frames = arr[np.newaxis, ...]
    elif arr.ndim == 3:
        if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.transpose(arr, (1, 2, 0))
            frames = arr[np.newaxis, ...]
        elif arr.shape[-1] in (3, 4):
            frames = arr[np.newaxis, ...]
        else:
            frames = arr
    elif arr.ndim == 4:
        frames = arr
    else:
        raise ValueError(f"Unsupported DICOM pixel array shape: {arr.shape}")

    out = []
    for frame in frames:
        img8 = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img8.ndim == 2:
            bgr = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
        elif img8.ndim == 3 and img8.shape[2] == 3:
            bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
        elif img8.ndim == 3 and img8.shape[2] == 4:
            rgb = img8[..., :3]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            gray = img8[..., 0] if img8.ndim == 3 else img8
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        out.append(bgr)
    return np.stack(out, axis=0)


def measure_pwv_frame(image: np.ndarray) -> float:
    """
    Process a single ultrasound frame (BGR image array) and return the time difference (ms)
    between ECG and Doppler upstroke initiations for that frame.
    """
    from pwv_visual_analysis import (
        get_calibration_from_image,
        create_masks,
        extract_ecg_trace,
        enhance_doppler_region,
        extract_doppler_trace,
        detect_upstroke_initiations
    )

    seconds_per_pixel, _ = get_calibration_from_image(image)
    ecg_mask, doppler_region, _ = create_masks(image)
    ecg_trace = extract_ecg_trace(ecg_mask, ecg_mask.shape[1])
    enhanced_mask, edges, enhanced_gray = enhance_doppler_region(doppler_region)
    doppler_trace = extract_doppler_trace(enhanced_mask, edges, enhanced_gray, enhanced_gray.shape[1])

    ecg_inits = detect_upstroke_initiations(ecg_trace, "ECG")
    doppler_inits = detect_upstroke_initiations(doppler_trace, "Doppler")

    diffs = []
    for e in ecg_inits:
        later = [d for d in doppler_inits if d > e]
        if later:
            dt = (min(later) - e) * seconds_per_pixel * 1000
            if 10 <= dt <= 300:
                diffs.append(dt)
    return float(np.mean(diffs)) if diffs else np.nan


def main():
    parser = argparse.ArgumentParser(description="Multi-frame PWV measurement with robust DICOM support")
    parser.add_argument("-p", "--proximal", required=True, help="Path to proximal DICOM file")
    parser.add_argument("-d", "--distal", required=True, help="Path to distal DICOM file")
    parser.add_argument("-s", "--site-distance", type=float, required=True, help="Distance between sites in mm")
    parser.add_argument("-o", "--output", default="pwv_multiframe_results", help="Output folder")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading proximal series...")
    prox_stack = load_dicom_series(args.proximal)
    print(f"Proximal frames: {len(prox_stack)}")

    print("Loading distal series...")
    dist_stack = load_dicom_series(args.distal)
    print(f"Distal frames: {len(dist_stack)}")

    n = min(len(prox_stack), len(dist_stack))
    results = []
    for i in range(n):
        print(f"Frame {i+1}/{n}")
        t_prox = measure_pwv_frame(prox_stack[i])
        t_dist = measure_pwv_frame(dist_stack[i])
        pwv = args.site_distance / ((t_dist - t_prox) / 1000) if (not np.isnan(t_prox) and not np.isnan(t_dist) and t_dist > t_prox) else np.nan
        results.append((i+1, t_prox, t_dist, pwv))

    csv_path = os.path.join(args.output, "pwv_results.csv")
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "TimeDiffProx_ms", "TimeDiffDist_ms", "PWV_mm_per_s"])
        writer.writerows(results)

    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()
