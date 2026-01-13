# Deployment Guide

Complete deployment guide for Face Recognition System on Windows environments.

## 📦 Quick Deployment

### For End Users (No Technical Knowledge Required)

1. **Download/Clone the project**
2. **Double-click `setup.bat`**
3. **Follow the prompts**
4. **Double-click `run.bat` to start**

That's it! 🎉

---

## 🛠️ Deployment Scripts

### 1. Automated Setup Scripts

#### Windows (PowerShell)
```powershell
.\scripts\setup_environment.ps1
```

Features:
- ✅ Python version check
- ✅ Virtual environment creation (optional)
- ✅ Package installation from requirements.txt
- ✅ Directory structure creation
- ✅ ONNX model download
- ✅ Setup validation
- ✅ Color-coded output

#### Cross-Platform (Python)
```bash
python scripts/setup_environment.py
```

Features:
- ✅ Works on Windows/Linux/Mac
- ✅ All features of PowerShell version
- ✅ Progress indicators
- ✅ Error handling

### 2. Quick Start Files

#### Windows Batch Files

**`setup.bat`** - One-click setup
```batch
@echo off
powershell -ExecutionPolicy Bypass -File "scripts\setup_environment.ps1"
```

**`run.bat`** - One-click start
```batch
@echo off
python app.py
```

---

## 📋 Prerequisites

### Required
- **Python 3.8+** 
- **pip** (comes with Python)
- **Internet connection** (for model downloads)

### Recommended
- **8GB RAM** minimum
- **2GB free disk space** (for models)
- **Modern browser** (Chrome/Edge/Firefox)

---

## 🚀 Step-by-Step Deployment

### Method 1: Automated (Recommended)

#### Windows Users:
1. Open project folder
2. Right-click `setup.bat`
3. Click "Run as Administrator" (optional)
4. Follow the prompts
5. Run `run.bat` to start

#### All Platforms:
```bash
cd face-recognition
python scripts/setup_environment.py
python app.py
```

### Method 2: Manual Deployment

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create directories
mkdir models data static\css static\js templates known_faces

# 6. Download models
python scripts/download_models.py

# 7. Run application
python app.py
```

---

## 🔧 Configuration

### Environment Variables (Optional)

Create `.env` file in project root:

```env
# Flask settings
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=your-secret-key-here

# Server settings
HOST=0.0.0.0
PORT=5000

# Face recognition settings
FACE_MATCH_THRESHOLD=0.50
```

### Production Deployment

#### Using Gunicorn (Linux/Mac)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### Using Waitress (Windows)
```bash
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

Or use the provided script:
```bash
python run_prod.py
```

---

## 🌐 Network Deployment

### Local Network Access

1. Find your local IP:
```bash
# Windows
ipconfig

# Linux/Mac
ifconfig
```

2. Update firewall rules (Windows):
```powershell
New-NetFirewallRule -DisplayName "Face Recognition" -Direction Inbound -LocalPort 5000 -Protocol TCP -Action Allow
```

3. Access from other devices:
```
http://YOUR_LOCAL_IP:5000
```

### HTTPS for Camera Access

Required for camera access on non-localhost:

```bash
# Generate certificate
python scripts/generate_cert.py

# Run with HTTPS
python scripts/run_https.py
```

Access via: `https://YOUR_IP:5000`

**Note**: Accept the self-signed certificate warning in your browser.

---

## 🐳 Docker Deployment (Advanced)

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directory
RUN mkdir -p data models

EXPOSE 5000

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t face-recognition .
docker run -p 5000:5000 -v $(pwd)/data:/app/data face-recognition
```

---

## ☁️ Cloud Deployment

### Render.com

Already configured with `render.yaml`:

1. Push to GitHub
2. Connect Render to your repository
3. Deploy automatically

### Heroku

```bash
# Login
heroku login

# Create app
heroku create your-app-name

# Add buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main
```

### AWS/Azure/GCP

Use the Docker deployment method with their container services.

---

## ✅ Validation Checklist

After deployment, verify:

- [ ] Python 3.8+ installed
- [ ] All packages installed (`pip list`)
- [ ] Models present in `models/` directory
  - [ ] det_10g.onnx (~17 MB)
  - [ ] w600k_r50.onnx (~166 MB)
- [ ] Directories created (`data/`, `templates/`, `static/`)
- [ ] Application starts without errors
- [ ] Web interface accessible
- [ ] Camera permissions work
- [ ] Face capture works
- [ ] Training completes successfully
- [ ] Recognition works

---

## 🔒 Security Considerations

### Production Checklist

- [ ] Change SECRET_KEY in app.py
- [ ] Use HTTPS (not HTTP)
- [ ] Disable debug mode
- [ ] Set up proper authentication
- [ ] Use environment variables for secrets
- [ ] Keep data/ folder secure
- [ ] Regular backups of face_embeddings.json
- [ ] Update dependencies regularly
- [ ] Use a reverse proxy (nginx/Apache)
- [ ] Set up rate limiting

### Data Privacy

- ⚠️ `data/face_embeddings.json` contains biometric data
- ⚠️ Ensure proper access controls
- ⚠️ Comply with GDPR/privacy regulations
- ⚠️ Regular data retention policy
- ⚠️ Secure deletion procedures

---

## 🆘 Troubleshooting

### Setup Fails

**Issue**: "Python not found"
```bash
# Verify Python installation
python --version

# Try python3
python3 --version

# Reinstall Python with PATH option checked
```

**Issue**: "Permission denied"
```powershell
# Windows: Allow script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Issue**: Package installation fails
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install with verbose output
pip install -r requirements.txt -v

# Try installing packages individually
pip install Flask opencv-python numpy
```

### Runtime Issues

**Issue**: Models not loading
- Check models/ directory has both .onnx files
- Verify file sizes (not 0 bytes)
- Re-download if corrupted

**Issue**: Camera not working
- Use HTTPS on remote devices
- Check browser permissions
- Try different browser
- Check camera not in use by other app

**Issue**: Poor recognition accuracy
- Capture more training samples (20-30 per person)
- Capture at multiple angles
- Ensure good lighting
- Retrain the model
- Adjust FACE_MATCH_THRESHOLD

---

## 📚 Additional Resources

- [Main README](README.md) - Project overview
- [SETUP.md](SETUP.md) - Detailed setup instructions
- [scripts/README.md](scripts/README.md) - Utility scripts documentation
- [docs/](docs/) - Technical documentation

---

## 🎯 Next Steps After Deployment

1. **Test the system**
   - Visit http://127.0.0.1:5000
   - Check all pages load
   - Test camera access

2. **Add training data**
   - Capture faces at `/capture`
   - Aim for 20-30 samples per person
   - Vary angles and expressions

3. **Train the model**
   - Visit `/train`
   - Click "Train Model"
   - Wait for completion

4. **Test recognition**
   - Visit `/recognize`
   - Test with known faces
   - Verify accuracy

5. **Configure for production**
   - Set environment variables
   - Enable HTTPS
   - Set up monitoring
   - Configure backups

---

## 📞 Support

For issues or questions:
- Review documentation in `docs/` folder
- Check troubleshooting section above
- Review application logs
- Verify all prerequisites are met
