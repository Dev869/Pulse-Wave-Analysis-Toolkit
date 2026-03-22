# CLAUDE.md

## Project Overview

Pulse Wave Analysis Toolkit — a Streamlit-based application for extracting Pulse Wave Velocity (PWV) from VisualSonics Vevo ultrasound DICOM files. It processes multi-frame DICOM data to detect ECG and Doppler signals, measure transit times, and calculate PWV (distance / Δtime). Designed for biomedical researchers studying vascular function in small animal models.

**Compatibility**: VisualSonics Vevo equipment only (tested on Vevo 2100). Currently limited to straight arteries.

## Repository Structure

```
├── pwv_app.py                 # Main Streamlit app (entry point) — 3-tab UI
├── pwv_visual_analysis.py     # Core image processing & signal extraction
├── pwv_multiframe.py          # DICOM loading & per-frame PWV measurement
├── distance_tool.py           # Tkinter-based interactive distance picker
├── pyconvert.py               # AVI → DICOM conversion utility
├── pwv_settings.json          # Default calibration settings
├── requirements.txt           # Python dependencies
├── PWV.bat                    # Windows launcher script
├── run_instructions.txt       # Quick-start guide
├── ultrasound_results/        # Output directory for analysis results
└── README.md                  # User documentation
```

## Architecture

**Data flow**: DICOM upload → frame extraction (`pwv_multiframe.py`) → image segmentation & signal extraction (`pwv_visual_analysis.py`) → peak/upstroke detection (`pwv_app.py`) → transit time calculation → PWV result.

### Key modules

- **`pwv_app.py`**: Streamlit UI with Settings, Upload & Calibrate, and Results tabs. Contains `find_peaks_and_bases()` for peak detection and profile management (`load_all_settings()` / `save_all_settings()`).
- **`pwv_visual_analysis.py`**: Image processing pipeline — `create_masks()` segments ECG/Doppler regions, `extract_ecg_trace()` uses green-channel color masking, `enhance_doppler_region()` applies multi-stage denoising (NLM, CLAHE, bilateral filter, Otsu thresholding, morphological cleaning), `extract_doppler_trace()` tries three methods (mask/edges/intensity) with quality-based selection, `detect_upstroke_initiations()` finds wave onset via derivative analysis.
- **`pwv_multiframe.py`**: `load_dicom_series()` handles multi-frame DICOM with grayscale/RGB/RGBA conversion. `measure_pwv_frame()` orchestrates single-frame analysis.
- **`distance_tool.py`**: Tkinter sequential point picker (proximal → distal), returns pixel distance via stdout to the calling process.
- **`pyconvert.py`**: Batch AVI-to-DICOM converter with tkinter GUI, creates proper DICOM metadata.

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run
streamlit run pwv_app.py

# Windows shortcut
PWV.bat
```

## Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web UI framework |
| streamlit-drawable-canvas | Interactive image drawing |
| numpy | Numerical operations |
| pandas | Data tables, CSV export |
| matplotlib | Per-frame visualizations |
| pydicom | DICOM file I/O |
| scipy | Signal filtering (savgol_filter) |
| Pillow | Image format handling |
| opencv-python-headless | Edge detection, morphological ops, denoising |
| tkinter | GUI for distance picker (system package) |

## Configuration

Settings are stored in `pwv_settings.json` with named profiles. Key parameters:

- `pix_per_mm` / `pixel_distance` — spatial calibration
- `time_pixels` / `time_seconds` — temporal calibration
- `ec_peak_frac` / `dp_peak_frac` — ECG/Doppler peak detection thresholds
- `ec_diff_frac` / `dp_diff_frac` — amplitude minimum thresholds

## Code Conventions

- **Python 3**, snake_case naming throughout
- Function docstrings with Args/Returns format
- No type hints currently used
- Line length not strictly enforced (~100+ chars)
- No formal linter or formatter configured
- Warnings suppressed: RuntimeWarning, UserWarning (in signal processing code)

## Testing

No automated test suite. Testing is manual — run the app, upload DICOM files, verify signal extraction and PWV calculations visually.

## Development Notes

- No CI/CD pipeline or pre-commit hooks
- Commits are small and focused (e.g., "fixed naming", "added pyconvert")
- The signal extraction in `pwv_visual_analysis.py` is the most complex and sensitive code — changes to thresholds or processing pipeline stages should be verified against real DICOM data
- `enhance_doppler_region()` uses a carefully tuned multi-stage pipeline; order of operations matters
- The `extract_doppler_trace()` function tries three independent methods and selects the best by quality score — maintain this fallback approach when modifying
