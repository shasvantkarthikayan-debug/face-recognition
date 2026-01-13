# Setup Instructions

Quick setup guide for the Face Recognition System.

## 🚀 Automated Setup (Recommended)

### Windows

Run the PowerShell setup script:

```powershell
# Option 1: Right-click and "Run with PowerShell"
.\scripts\setup_environment.ps1

# Option 2: From PowerShell terminal
cd path\to\face-recognition
.\scripts\setup_environment.ps1
```

**Note**: If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Cross-Platform (Windows/Linux/Mac)

Run the Python setup script:

```bash
cd path/to/face-recognition
python scripts/setup_environment.py
```

---

## 📋 What the Setup Script Does

1. ✅ Checks Python version (3.8+ required)
2. ✅ Creates virtual environment (optional)
3. ✅ Installs all Python packages from `requirements.txt`
4. ✅ Creates necessary directories (`models/`, `data/`, etc.)
5. ✅ Downloads ONNX models (det_10g.onnx, w600k_r50.onnx)
6. ✅ Validates the complete setup
7. ✅ Creates run scripts for easy startup

---

## 🔧 Manual Setup

If automated setup fails, follow these steps:

### 1. Install Python
- Download Python 3.8+ from [python.org](https://www.python.org/downloads/)
- Ensure "Add Python to PATH" is checked during installation

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Models

Download these ONNX models and place them in the `models/` folder:

- **det_10g.onnx** - Face detection (SCRFD)
- **w600k_r50.onnx** - Face recognition (ArcFace)

Download from:
- [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)
- Or run: `python scripts/download_models.py`

### 5. Verify Structure

Ensure these directories exist:
```
models/
data/
templates/
static/
```

---

## ▶️ Running the Application

### Option 1: Using Python directly

```bash
python app.py
```

### Option 2: Using run scripts

```bash
# Windows
run.bat

# Linux/Mac
./run.sh
```

### Option 3: Production mode

```bash
python run_prod.py
```

Then open your browser to: **http://127.0.0.1:5000**

---

## 🔒 HTTPS Setup (Optional)

For camera access on remote devices:

```bash
# Generate SSL certificate
python scripts/generate_cert.py

# Run with HTTPS
python scripts/run_https.py
```

Access via: **https://your-ip:5000**

---

## ✅ Troubleshooting

### "Python not found"
- Reinstall Python and ensure it's added to PATH
- Try `python3` instead of `python`

### "Permission denied" on Windows
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Models not downloading
- Check your internet connection
- Download manually from InsightFace model zoo
- See: [docs/MODEL_DOWNLOAD_INSTRUCTIONS.md](docs/MODEL_DOWNLOAD_INSTRUCTIONS.md)

### Camera not working
- Use HTTPS (browsers require secure context for camera)
- Check browser permissions
- Try a different browser (Chrome/Edge recommended)

### Package installation fails
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install packages one by one to identify the issue
pip install Flask
pip install opencv-python
# ... etc
```

---

## 📚 Next Steps

After setup:

1. **Capture Training Data**: Visit `/capture` to add faces
2. **Train Model**: Visit `/train` to compute embeddings
3. **Recognize Faces**: Visit `/recognize` to test the system
4. **Manage Data**: Visit `/manage` to view/delete entries

---

## 🆘 Getting Help

- Check [README.md](README.md) for detailed documentation
- Review [docs/](docs/) folder for technical details
- Check [scripts/README.md](scripts/README.md) for utility scripts
