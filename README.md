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
