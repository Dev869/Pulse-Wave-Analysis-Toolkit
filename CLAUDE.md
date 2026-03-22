# CLAUDE.md

## Project Overview

Pulse Wave Analysis Toolkit — a Streamlit-based application for extracting Pulse Wave Velocity (PWV) from VisualSonics Vevo ultrasound DICOM files. It processes multi-frame DICOM data to detect ECG and Doppler signals, measure transit times, and calculate PWV (distance / Δtime). Designed for biomedical researchers studying vascular function in small animal models.

**Compatibility**: VisualSonics Vevo equipment only (tested on Vevo 2100). Currently limited to straight arteries.

## Repository Structure

```
├── pwv_app.py                 # Main Streamlit app (entry point) — 3-tab UI
├── pwv_visual_analysis.py     # Core image processing & signal extraction
├── pwv_multiframe.py          # DICOM loading & per-frame PWV measurement (also has CLI mode)
├── distance_tool.py           # Tkinter-based interactive distance picker (subprocess)
├── pyconvert.py               # AVI → DICOM conversion utility
├── pwv_settings.json          # Legacy/default calibration values (NOT used by the app directly)
├── requirements.txt           # Python dependencies
├── PWV.bat                    # Windows launcher script
├── run_instructions.txt       # Quick-start guide
├── ultrasound_results/        # Output directory for analysis results
└── README.md                  # User documentation
```

## Architecture

**Data flow**: DICOM upload → frame extraction (`pwv_multiframe.py`) → image segmentation & signal extraction (`pwv_visual_analysis.py`) → peak/upstroke detection (`pwv_app.py`) → transit time calculation → PWV result.

**State management**: The Streamlit app uses `st.session_state` extensively to pass data between the three tabs (Settings, Upload & Calibrate, Results). All calibration parameters, DataFrames, images, and summary results are stored in session state.

### Key Modules

- **`pwv_app.py`**: Streamlit UI entry point. Settings tab manages named profiles persisted to `~/.pwv_settings.json` (user home, NOT the repo-level `pwv_settings.json`). Upload tab handles DICOM loading, distance measurement (spawns `distance_tool.py` as subprocess), and frame-by-frame analysis. Results tab displays transit times, PWV, and per-frame trace plots. Contains `find_peaks_and_bases()` for peak detection. Note: `sec_per_pix` is computed in the Settings tab scope but used in the Analysis tab — this works because Streamlit re-runs the entire script top-to-bottom on each interaction.
- **`pwv_visual_analysis.py`**: Image processing pipeline. `create_masks()` segments ECG/Doppler regions using hardcoded fractional boundaries (ECG: 80–95% height, Doppler: 40–70% height, left crop 20px, right crop 120px from right edge — these are Vevo-specific). `extract_ecg_trace()` uses green-channel HSV masking. `enhance_doppler_region()` applies a 10-step pipeline (NLM denoising → CLAHE → bilateral filter → Gaussian blur → horizontal line removal → Otsu threshold → adaptive threshold → combine → morphological open/close → Canny edges). `extract_doppler_trace()` tries three methods (mask/edges/intensity) and selects by highest variance. **Known dead code**: after `detect_upstroke_initiations()` returns on line 349, there is ~70 lines of unreachable code from an old implementation, plus a `main()` referencing a nonexistent `PWVUI` class.
- **`pwv_multiframe.py`**: `load_dicom_series()` handles multi-frame DICOM with grayscale/RGB/RGBA conversion, forced syntax for malformed files. `measure_pwv_frame()` orchestrates single-frame analysis. **CLI mode**: `python pwv_multiframe.py -p prox.dcm -d dist.dcm -s <distance_mm>` runs analysis without the Streamlit UI.
- **`distance_tool.py`**: Tkinter sequential point picker (proximal image → distal image), returns pixel distance via stdout. Called as a subprocess from `pwv_app.py`.
- **`pyconvert.py`**: Batch AVI-to-DICOM converter with tkinter file picker GUI.

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit UI
streamlit run pwv_app.py

# CLI mode (no UI)
python pwv_multiframe.py -p proximal.dcm -d distal.dcm -s 5.0

# Windows shortcut
PWV.bat
```

## Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web UI framework |
| streamlit-drawable-canvas | Interactive image drawing |
| numpy | Numerical operations, signal arrays |
| pandas | Data tables, CSV export |
| matplotlib | Per-frame trace visualizations |
| pydicom | DICOM file I/O |
| scipy | Signal filtering (savgol_filter) |
| Pillow | Image format handling |
| opencv-python-headless | Edge detection, morphological ops, denoising, color conversion |
| tkinter | GUI for distance picker (system package, not pip-installable) |

**Known issue in `requirements.txt`**: `opencv-python-headless` (imported as `cv2`) is missing — it must be installed separately (`pip install opencv-python-headless`). The entry `Image` on line 9 is invalid (PIL's Image comes from the `Pillow` package already listed).

## Configuration

User profiles are persisted to **`~/.pwv_settings.json`** (user home directory). The repo-level `pwv_settings.json` contains legacy default values and is not read by the app.

Key parameters (all accessible in the Settings tab):

- `pix_per_mm` — spatial calibration (pixels per millimeter)
- `time_pixels` / `time_seconds` — temporal calibration (pixels and seconds for the time scale bar)
- `ec_peak_frac` / `dp_peak_frac` — ECG/Doppler peak detection height threshold (fraction of max)
- `ec_diff_frac` / `dp_diff_frac` — minimum amplitude thresholds (fraction of max, filters noise)
- `tt_min_ms` / `tt_max_ms` — transit time inclusion bounds (ms), frames outside this range are excluded

## Code Conventions

- **Python 3**, snake_case naming throughout
- Function docstrings with Args/Returns format in key functions
- No type hints (except `load_dicom_series` in `pwv_multiframe.py`)
- No formal linter or formatter configured
- Warnings suppressed: RuntimeWarning, UserWarning (in signal processing code)
- OpenCV uses BGR color order internally; conversion happens at DICOM load time

## Testing

No automated test suite. Testing is manual — run the app, upload DICOM files, verify signal extraction and PWV calculations visually.

## Development Notes

- No CI/CD pipeline or pre-commit hooks
- **Image region boundaries are hardcoded** in `create_masks()` — ECG at 80–95% image height, Doppler at 40–70%, with fixed pixel crops. These values are calibrated for VisualSonics Vevo output format. Changing them affects all signal extraction downstream.
- `enhance_doppler_region()` is a carefully tuned 10-step pipeline; **order of operations matters** and parameter changes should be verified against real DICOM data
- `extract_doppler_trace()` tries three independent extraction methods and selects the best by variance — maintain this fallback approach when modifying
- `detect_upstroke_initiations()` in `pwv_visual_analysis.py` is marked as unused; the Streamlit app uses its own `find_peaks_and_bases()` approach instead
- The dead code block after line 349 in `pwv_visual_analysis.py` (old `measure_time_difference` body + `main()`) can be safely removed
- `pwv_multiframe.py`'s `measure_pwv_frame()` still uses `detect_upstroke_initiations()` and `get_calibration_from_image()` (which has hardcoded values) — this is the CLI path only, not used by the Streamlit app
