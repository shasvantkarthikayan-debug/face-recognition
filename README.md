# FacePass — Face Recognition System

A real-time facial recognition system using **InsightFace ArcFace** (ONNX) for high-accuracy 512-D embeddings with Flask backend and browser-based camera interface.

---

## 🚀 Features

- 🔍 Real-time face detection and recognition
- 🎯 **512-D ArcFace embeddings** via InsightFace ONNX models
- 🌐 Browser-based webcam capture (no desktop app needed)
- 🧠 Backend embedding extraction and matching (Python)
- 💾 JSON-based embedding storage with centroid training
- ⚡ CPU-optimized (no GPU required)
- 📊 Multi-sample training for pose variation tolerance
- 🔐 Fully local processing (no cloud APIs)

---

## 🛠 Tech Stack

**Frontend**
- HTML5/CSS/JavaScript
- WebRTC for camera access
- Responsive UI

**Backend**
- Flask (Python web framework)
- InsightFace ONNX models (ArcFace w600k_r50)
- ONNX Runtime for inference
- OpenCV for image processing
- NumPy & scikit-learn for embeddings
- Cosine similarity for face matching

---

## 📁 Project Structure

```
face-recognition/
├── app.py                 # Main Flask application
├── run_prod.py           # Production runner
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── render.yaml          # Deployment config
│
├── templates/           # HTML pages
│   ├── index.html       # Home page
│   ├── capture.html     # Capture training data
│   ├── train.html       # Train model
│   ├── recognize.html   # Recognition page
│   └── manage.html      # Dataset management
│
├── static/              # CSS, JS, client-side assets
│   ├── css/
│   └── js/
│
├── models/              # ONNX model files
│   ├── det_10g.onnx     # Face detection (SCRFD)
│   └── w600k_r50.onnx   # Face recognition (ArcFace)
│
├── data/                # Training data (gitignored)
│   └── face_embeddings.json
│
├── scripts/             # Utility scripts
│   ├── download_models.py
│   ├── generate_cert.py
│   └── README.md
│
└── docs/               # Documentation
    ├── ALIGNMENT_FIX.md
    └── MODEL_DOWNLOAD_INSTRUCTIONS.md
```

---

## 🚀 Quick Start

### Automated Setup (Recommended)

**Windows:**
```powershell
.\scripts\setup_environment.ps1
```

**Cross-Platform:**
```bash
python scripts/setup_environment.py
```

The setup script will:
- ✅ Install all dependencies
- ✅ Download ONNX models
- ✅ Create necessary directories
- ✅ Validate the setup

### Manual Setup

See [SETUP.md](SETUP.md) for detailed installation instructions.

### Run the Application

```bash
python app.py
```

Visit: **http://127.0.0.1:5000**

---

## 📖 Usage

1. **Capture Training Data** (`/capture`)
   - Enter name and category
   - Capture 15-25 samples at various angles
   - Repeat for each person

2. **Train Model** (`/train`)
   - Click "Train Model"
   - Computes centroid embeddings from samples

3. **Recognize Faces** (`/recognize`)
   - Point camera at person
   - System shows identity and confidence score

4. **Manage Dataset** (`/manage`)
   - View all trained people
   - Delete entries
   - View statistics

---

## 🔧 Configuration

**Match Threshold** (in `app.py`):
```python
FACE_MATCH_THRESHOLD = 0.50  # Cosine similarity threshold
```

- **0.50-0.70**: Lenient (handles pose variations)
- **0.70-0.85**: Balanced (recommended)
- **0.85+**: Strict (requires high-quality frontal faces)

---

## 📊 Data Storage

- **Training data**: `data/face_embeddings.json`
  - Raw embeddings (all samples per person)
  - Computed centroids (for recognition)
  - Names and categories
  - Training status

⚠️ **Note**: The `data/` folder contains personal biometric data and is gitignored by default.

---

## 🔒 HTTPS (Optional)

For camera access on remote devices:

```bash
python scripts/generate_cert.py
python scripts/run_https.py
```

Visit: **https://your-ip:5000**

---

## 📚 Documentation

- **[SETUP.md](SETUP.md)** - Detailed setup and installation guide
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Production deployment guide
- **[docs/MODEL_DOWNLOAD_INSTRUCTIONS.md](docs/MODEL_DOWNLOAD_INSTRUCTIONS.md)** - Model setup
- **[scripts/README.md](scripts/README.md)** - Utility scripts reference
- **[data/README.md](data/README.md)** - Data structure and backup

---
