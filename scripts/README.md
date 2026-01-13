# Utility Scripts

This folder contains utility and helper scripts for the face recognition project.

## 🚀 Setup & Deployment

- **`setup_environment.ps1`** - **Windows automated setup** (PowerShell)
- **`setup_environment.py`** - **Cross-platform automated setup** (Python)
  - Installs packages
  - Downloads models
  - Creates directories
  - Validates setup

## Model Management

- **`download_model.py`** - Download individual ONNX model files
- **`download_models.py`** - Download all required ONNX models  
- **`setup_models.py`** - Setup and extract model files
- **`extract_models.py`** - Extract models from archives

## Data Management

- **`clean_and_retrain.py`** - Clean old data and retrain from scratch (one-time migration script)
- **`fix_embeddings.py`** - Fix/inspect embedding data structure (one-time fix script)
- **`capture.py`** - Legacy command-line face capture utility

## Testing

- **`test_alignment.py`** - Test face alignment algorithms
- **`test_cert.py`** - Test SSL certificate configuration
- **`test_validation.py`** - Test face validation logic

## HTTPS/SSL

- **`generate_cert.py`** - Generate self-signed SSL certificates for HTTPS
- **`https_server.py`** - Run Flask app with HTTPS
- **`run_https.py`** - Alternative HTTPS runner

## Usage

Run scripts from the project root directory:

```bash
# Example: Download models
python scripts/download_models.py

# Example: Generate SSL certificate
python scripts/generate_cert.py

# Example: Test alignment
python scripts/test_alignment.py
```

## Note

Most of these are **one-time setup** or **diagnostic** tools. The main application is [`app.py`](../app.py) in the root directory.
