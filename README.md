# PWV (Pulse Wave Velocity) Visual Analysis Toolkit

## Compatibility Notes & Intended Usage

- This program is only designed for DICOM files output from a VisualSonics Vevo. The program has only been tested so far on a VisualSonics Vevo 2100.
- So far, the program is only able to calculate PWV on straight arteries due to the current methods of measuring distance & visual analysis. There are plans to adapt the program to accomodate a curved artery (like the aortic arch).

## Setup & Requirements (macOS)

> If you don’t have Git installed, download it from https://git-scm.com/.

> If you don't have Python 3 istalled, download it from http://python.org/downloads/

> If you don't have Homebrew installed, follow the steps at https://brew.sh/

### 1. Clone the Repository
```bash
# Clone the GitHub repo
git clone https://github.com/Dev869/pulsewave-visual.git
cd pulsewave-visual
```

### 2. Create & Activate a Python Virtual Environment
```bash
# Create a new virtual environment in `.venv`
python3 -m venv .venv

# Activate it
source .venv/bin/activate
```

### 3. Upgrade pip & Build Tools
```bash
pip install --upgrade pip setuptools wheel
```

### 4. Install System Dependencies (Homebrew)
```bash
# Install pkg-config & OpenCV & Tkinter
brew install pkg-config opencv tcl-tk
```

### 5. Install Python Requirements
```bash
# Install only the required PyPI packages
pip install -r requirements.txt
```

### 6. Run the Streamlit App
```bash
# Launch the PWV UI in your browser
streamlit run pwv_app.py
```

> Or equivalently:
> ```bash
> python -m streamlit run pwv_app.py
> ```

### 7. Configuration & Troubleshooting
- If you encounter **OpenCV** import errors, confirm you have installed `opencv-python-headless` or built OpenCV via Homebrew.
- For any broken image-to-canvas issues, ensure `streamlit-drawable-canvas` is up to date and matches your Streamlit version.
- Permissions errors? Try:
  ```bash
  chmod +x pwv_app.py
  ```
- Customizing ports or base URL? Edit `~/.streamlit/config.toml`.

---

## Setup & Requirements (Windows)

> Tested on Windows 10/11. Run commands in an **Administrator** PowerShell or CMD.

### 1. Clone the Repository
```powershell
git clone https://github.com/Dev869/pulsewave-visual.git
cd pulsewave-visual
```

### 2. Create & Activate a Virtual Environment
```powershell
python -m venv .venv
# Activate (PowerShell)
.\.venv\Scripts\Activate.ps1
# Or CMD
.\.venv\Scripts\activate.bat
```

### 3. Upgrade pip & Build Tools
```powershell
pip install --upgrade pip setuptools wheel
```

### 4. Install System Dependencies
```powershell
pip install opencv-python-headless
# Optional: Tkinter support
pip install tk
```

### 5. Install Python Requirements
```powershell
pip install -r requirements.txt
```

### 6. Run the Streamlit App
```powershell
streamlit run pwv_app.py
```

> Or:
> ```powershell
> python -m streamlit run pwv_app.py
> ```

---

## Setup & Requirements (Linux)

> Tested on Ubuntu 20.04 / Debian 11

### 1. Clone the Repository
```bash
git clone https://github.com/Dev869/pulsewave-visual.git
cd pulsewave-visual
```

### 2. Create & Activate a Python Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Upgrade pip & Build Tools
```bash
pip install --upgrade pip setuptools wheel
```

### 4. Install System Dependencies
```bash
sudo apt update
sudo apt install -y python3-tk libtcl8.6 libtk8.6 tcl-dev tk-dev pkg-config libopencv-dev
```

### 5. Install Python Requirements
```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit App
```bash
streamlit run pwv_app.py
```

> Or:
> ```bash
> python3 -m streamlit run pwv_app.py
> ```

---

## Usage
### Settings
- `Pixels per mm`: Your image’s spatial calibration. Enter how many pixels on the image correspond to 1 mm in the real world (e.g. from a known scale bar). This lets the app convert pixel distances into millimetres.

- `Time scale bar pixels`: The length of the on-image time bar in pixels (e.g. if the bar spans 839 pixels in the frame). Combined with the next setting, this gives your temporal calibration.

- `Time scale bar seconds`: The real‐world duration of that time bar (e.g. 0.7 s). Dividing this by the pixel length (Time scale bar seconds ÷ Time scale bar pixels) yields ms/pixel, used to convert pixel‐based delays into milliseconds.

- `ECG peak fraction`: A fraction (0–1) of the maximum ECG‐trace amplitude used to detect valid R‐wave peaks (e.g. 0.5 means only peaks ≥ 50% of the signal’s max are counted).

- `Doppler peak fraction`: Same idea for the Doppler signal: only count Doppler upstrokes whose peak amplitude is at least this fraction of its max.

- `Min ECG diff fraction`: A fraction (0–1) of the ECG’s peak‐to‐baseline amplitude that must be exceeded to consider a beat valid (filters out small noise blips).

- `Min Doppler diff fraction`: Like above, but for the Doppler trace’s peak‐to‐baseline amplitude.

- `Min TT to include (ms)` and `Max TT to include (ms)`: Exclude any transit‐time measurements (delay between ECG and Doppler upstrokes) that fall outside this millisecond window. Useful for ignoring physiologically implausible outliers.

- `Frames to process`: How many consecutive frames (from the start of your uploaded DICOM or TIFF stack) the app will analyze. More frames = more cycles = more robust average but longer compute time.

- `Measure Probe Separation`: Click once on the Proximal image and once on the Distal image to define the physical separation of your measurement points. The app converts your two clicks (in pixels) into millimetres using your Pixels per mm setting.

- **Profiles**: Save or load named profiles of all the above settings so you can quickly switch between different calibration or threshold configurations.

### Analysis

1. Upload your **DICOM** (.dcm) for the proximal and distal locations.
2. Measure the distance by clicking the **Measure Distance** button.

- **IMPORTANT**: Be as accurate as possible in this step. You will be *manually selecting* the points for the proximal and distal locations. 

This will open first the *proximal* image and second the *distal* image where you are meant to click the points on the image.

### Results

Here, you can see the traces for each frame analzyed as well as the data tables containing the TT (transit times) for both the proximal and distal frames.
