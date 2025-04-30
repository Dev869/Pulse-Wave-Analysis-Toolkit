# PWV (Pulse Wave Velocity) Visual Analysis Toolkit

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

