# InsightFace Model Download Instructions

The models need to be downloaded from the InsightFace repository. The download script is running in the background.

## Option 1: Manual Download (Recommended)

Download these files from InsightFace GitHub releases:

**buffalo_l Model Pack:**
https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip

This package contains:
- `det_10g.onnx` - SCRFD detector (~17MB)
- `2d106det.onnx` - 5-point landmark detector (~500KB)  
- `w600k_r50.onnx` - ArcFace recognition model (~166MB)

## Extraction Steps:

1. Download buffalo_l.zip to `face-recognition/models/` folder

2. Extract the zip file

3. Rename/copy the files:
   ```
   models/buffalo_l/det_10g.onnx -> models/scrfd_10g_bnkps.onnx
   models/buffalo_l/2d106det.onnx -> models/coordinate_reg_mean.onnx
   models/buffalo_l/w600k_r50.onnx -> models/w600k_r50.onnx (or keep arcface_w600k_r50.onnx)
   ```

## Option 2: Use the Download Script

The `download_models.py` script in the face-recognition folder will automatically:
- Download buffalo_l.zip
- Extract it
- Rename models correctly
- Clean up temporary files

Run: `python download_models.py`

## Required Models:

After setup, you should have these files in `models/` folder:
- ✓ `scrfd_10g_bnkps.onnx` (SCRFD detector)
- ✓ `coordinate_reg_mean.onnx` (5-point landmarks)
- ✓ `arcface_w600k_r50.onnx` or `w600k_r50.onnx` (ArcFace embeddings)

## Update ArcFace Model Path:

If using `w600k_r50.onnx` instead of `arcface_w600k_r50.onnx`, update line 25 in app.py:
```python
ARCFACE_MODEL_PATH = "models/w600k_r50.onnx"
```
