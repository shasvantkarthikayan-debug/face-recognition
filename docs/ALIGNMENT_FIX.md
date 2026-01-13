# ArcFace Alignment Fix Summary

## Problem
Recognition similarity was extremely low (~0.05) because training and recognition pipelines used different preprocessing.

## Solution Implemented

### 1. Added MediaPipe for Landmark Detection
- Installed `mediapipe>=0.10.0`
- Uses MediaPipe Face Mesh for 5-point landmark detection
- Landmarks: left eye (33), right eye (263), nose (1), left mouth (61), right mouth (291)

### 2. Implemented Proper Face Alignment
```python
def align_face_5point(face_image, src_landmarks):
    # Detect 5-point landmarks using MediaPipe
    # Apply similarity transform to canonical ArcFace template
    # Output: 112x112 aligned face
```

**Canonical ArcFace 5-point template:**
- Left eye: (38.29, 51.70)
- Right eye: (73.53, 51.50)
- Nose: (56.03, 71.74)
- Left mouth: (41.55, 92.37)
- Right mouth: (70.73, 92.20)

### 3. Unified Preprocessing Pipeline
**Both training and recognition now use IDENTICAL preprocessing:**
1. Detect 5-point landmarks using MediaPipe
2. Apply similarity transform to align face to canonical template
3. Resize to 112x112
4. Convert BGR → RGB
5. Normalize to [-1, 1]: `(pixel - 127.5) / 127.5`
6. Transpose to CHW format (channels first)
7. Add batch dimension

### 4. Added Debug Logging
**During embedding generation:**
- Input image shape
- After alignment shape
- Normalized pixel range
- Embedding norm
- First 5 embedding values

**During recognition:**
- Query embedding details
- Each centroid comparison
- Cosine similarities
- Best match details

### 5. Updated Endpoints
- `/capture_photo`: Uses aligned preprocessing (debug=True)
- `/capture_dataset`: Uses aligned preprocessing (debug=False)
- `/recognize`: Uses aligned preprocessing (debug=True)

### 6. Cleaned Data
- Backed up old `face_data.json` → `face_data.json.backup_before_alignment`
- Backed up old `known_faces/` → `known_faces_backup_before_alignment/`
- Created fresh database ready for retraining

## Expected Results After Retraining

### With Proper Alignment:
- **Same person:** similarity > 0.75 (distance < 0.25)
- **Different person:** similarity < 0.4 (distance > 0.6)

### Previous (Without Alignment):
- Similarity ~0.05 (almost random)
- No meaningful recognition

## Next Steps

1. **Start server:**
   ```bash
   python run_prod.py
   ```

2. **Capture dataset:**
   - Go to Dataset tab
   - Capture 20-30 frames per person
   - Ensure good lighting and face visibility

3. **Train model:**
   - Go to Train tab
   - Click "Train Model"
   - Wait for training to complete

4. **Test recognition:**
   - Go to Recognition tab
   - Verify stable face detection
   - Check console for debug logs
   - Verify similarity scores > 0.75 for same person

## Debug Logs to Watch

### During Dataset Capture:
```
[CAPTURE] Processing face for John
  [PREPROCESS] Input shape: (480, 640, 3)
  [PREPROCESS] After alignment: (112, 112, 3)
  [PREPROCESS] Normalized range: [-1.000, 1.000]
  [PREPROCESS] Output shape: (1, 3, 112, 112)
  Embedding norm: 1.0000
  Embedding first 5 values: [0.123, -0.456, ...]
```

### During Recognition:
```
[RECOGNITION] Processing query image
  [PREPROCESS] Input shape: (480, 640, 3)
  [PREPROCESS] After alignment: (112, 112, 3)
  
[RECOGNITION DEBUG]
Query embedding norm: 1.0000
Query embedding first 5 values: [0.123, -0.456, ...]

  Person 0 (John): Using centroid (dim=512)
    Centroid norm: 1.0000
    Cosine similarity: 0.8234  ← Should be > 0.75 for match

Best match:
  Name: John
  Similarity: 0.8234
  Distance: 0.1766
  Threshold: 0.45
  Match: True
```

## Files Modified

1. **app.py**
   - Added MediaPipe import and initialization
   - Implemented `get_face_landmarks_5point()`
   - Implemented `align_face_5point()` with similarity transform
   - Updated `preprocess_face_for_arcface()` with debug logging
   - Updated `generate_arcface_embedding()` with debug logging
   - Updated all endpoints to use consistent preprocessing

2. **requirements.txt**
   - Added `mediapipe>=0.10.0`

3. **clean_and_retrain.py** (NEW)
   - Backup and clean existing data
   - Prepare for retraining

## Troubleshooting

### If similarity is still low after retraining:
1. Check debug logs for embedding consistency
2. Verify landmarks are detected (check for "⚠️ Landmark detection failed")
3. Ensure good lighting during capture
4. Capture more frames (30-40) per person
5. Verify face is frontal and clearly visible

### If landmarks fail to detect:
- Falls back to center crop + resize
- Still maintains 112x112, RGB, [-1,1] normalization
- Less accurate but functional
