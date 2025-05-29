import tkinter as tk
from tkinter import filedialog, messagebox
import os
import cv2
import pydicom
import numpy as np
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid, MultiFrameTrueColorSecondaryCaptureImageStorage
import datetime

def convert_all_avi_to_rgb_dicom():
    folder = filedialog.askdirectory(title="Select Folder with AVI Files")
    if not folder:
        return

    output_dir = os.path.join(folder, "dicom_output")
    os.makedirs(output_dir, exist_ok=True)

    avi_files = [f for f in os.listdir(folder) if f.lower().endswith('.avi')]
    if not avi_files:
        messagebox.showwarning("No AVI files found.")
        return

    for avi_name in avi_files:
        avi_path = os.path.join(folder, avi_name)
        base_name = os.path.splitext(avi_name)[0]
        save_path = os.path.join(output_dir, f"{base_name}.dcm")

        cap = cv2.VideoCapture(avi_path)
        if not cap.isOpened():
            print(f"Failed to open {avi_name}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        frames = []
        timestamps = []

        for frame_index in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            timestamps.append(frame_index / fps)

        cap.release()
        if not frames:
            print(f"Skipping {avi_name}: no frames")
            continue

        array = np.stack(frames)  # (num_frames, height, width, 3)
        dicom = create_rgb_dicom(array, timestamps, fps, base_name, avi_path, width, height, duration)
        dicom.save_as(save_path)
        print(f"Saved {save_path}")

    messagebox.showinfo("Done", f"Converted {len(avi_files)} AVI files to DICOM in:\n{output_dir}")

def create_rgb_dicom(array, timestamps, fps, source_name, source_path, width, height, duration):
    num_frames, rows, cols, _ = array.shape
    now = datetime.datetime.now()

    meta = pydicom.Dataset()
    meta.MediaStorageSOPClassUID = MultiFrameTrueColorSecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()

    # Header
    ds.PatientName = "Converted^FromAVI"
    ds.PatientID = "000000"
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")
    ds.Manufacturer = "AVItoDICOMConverter"
    ds.StudyDescription = f"From {source_name}"
    ds.SeriesDescription = f"RGB DICOM video from {os.path.basename(source_path)}"
    ds.Modality = "OT"
    ds.StationName = os.uname().nodename if hasattr(os, 'uname') else "unknown"
    ds.ImageComments = f"Source: {source_path}; FPS: {fps:.2f}; Duration: {duration:.2f}s; Res: {width}x{height}"

    # Image properties
    ds.Rows = rows
    ds.Columns = cols
    ds.NumberOfFrames = str(num_frames)
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0  # Interleaved RGB
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0

    # Timing (FrameTime in ms)
    if fps > 0:
        ds.FrameTime = str(round(1000 / fps))  # ms per frame

    # Shared Functional Group
    shared_fg = Dataset()
    color = Dataset()
    color.SamplesPerPixel = 3
    color.PhotometricInterpretation = "RGB"
    shared_fg.FrameVOILUTSequence = Sequence([color])
    ds.SharedFunctionalGroupsSequence = Sequence([shared_fg])

    # Per-frame Functional Group
    per_frame_sequence = []
    for i, rel_time in enumerate(timestamps):
        fg = Dataset()
        frame_content = Dataset()
        frame_content.FrameAcquisitionNumber = i + 1
        acq_time = now + datetime.timedelta(seconds=rel_time)
        frame_content.FrameAcquisitionDateTime = acq_time.strftime("%Y%m%d%H%M%S.%f")[:22]
        fg.FrameContentSequence = Sequence([frame_content])
        per_frame_sequence.append(fg)

    ds.PerFrameFunctionalGroupsSequence = Sequence(per_frame_sequence)

    # PixelData
    ds.PixelData = array.tobytes()
    return ds

# GUI
root = tk.Tk()
root.title("AVI to RGB Multi-frame DICOM Converter")

btn = tk.Button(root, text="Batch Convert AVI to RGB DICOM", command=convert_all_avi_to_rgb_dicom, padx=20, pady=10)
btn.pack(pady=20)

root.mainloop()
