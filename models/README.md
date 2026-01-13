# Models Directory

This directory contains ONNX model files for face detection and recognition.

## Required Models

- **`det_10g.onnx`** (~17 MB) - SCRFD face detection model
- **`w600k_r50.onnx`** (~166 MB) - ArcFace face recognition model

## Download Models

Models are **not included in git** due to their large size.

### Automated Download

Run the setup script:
```bash
# Windows
.\scripts\setup_environment.ps1

# Cross-platform
python scripts/setup_environment.py
```

### Manual Download

```bash
python scripts/download_models.py
```

Or download manually from:
- [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)

## Additional Models (Optional)

Other models in this directory are optional and used for specific features:
- `1k3d68.onnx` - 3D face landmarks
- `2d106det.onnx` - 2D face detection
- `genderage.onnx` - Gender/age estimation
- `face_landmarker.task` - MediaPipe face landmarks

## Note

⚠️ Do not commit `.onnx` files to git - they are large binary files that should be downloaded during setup.
