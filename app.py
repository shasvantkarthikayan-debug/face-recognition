from flask import Flask, render_template, Response, request, jsonify, send_file
import numpy as np
import os
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import onnxruntime as ort
import cv2
from PIL import Image
import io
import base64

print(f"✓ OpenCV imported: {cv2.__version__ if hasattr(cv2, '__version__') else 'version unknown'}")

# Initialize ONNX models for face detection and recognition
MODELS_DIR = 'models'
detection_model_path = os.path.join(MODELS_DIR, 'det_10g.onnx')
recognition_model_path = os.path.join(MODELS_DIR, 'w600k_r50.onnx')

# Load ONNX models
print("🔄 Loading InsightFace models...")
try:
    detection_session = ort.InferenceSession(detection_model_path, providers=['CPUExecutionProvider'])
    recognition_session = ort.InferenceSession(recognition_model_path, providers=['CPUExecutionProvider'])
    INSIGHTFACE_AVAILABLE = True
    print("✓ InsightFace models loaded successfully (512-D embeddings)")
except Exception as e:
    INSIGHTFACE_AVAILABLE = False
    print(f"❌ Failed to load InsightFace models: {e}")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'facepass-secret-key'

# MediaPipe: Disabled for now (0.10+ API changed, and landmarks don't significantly improve ArcFace)
# The align_face_5point function will fall back to center crop + resize, which works well
MEDIAPIPE_AVAILABLE = False
print("⚠️ MediaPipe disabled - using center crop alignment (sufficient for ArcFace)")

# Face recognition threshold for verification (cosine similarity)
# 0.50 = More lenient to handle pose variations
# IMPORTANT: Capture training data at multiple angles (frontal, left, right, up, down)
FACE_MATCH_THRESHOLD = 0.50  # Lowered to 0.50 for pose tolerance

def preprocess_image_for_detection(img, input_size=(640, 640)):
    """Preprocess image for SCRFD face detection"""
    img_resized = cv2.resize(img, input_size)
    img_normalized = (img_resized.astype(np.float32) - 127.5) / 128.0
    img_transposed = img_normalized.transpose(2, 0, 1)
    img_batch = np.expand_dims(img_transposed, axis=0)
    return img_batch

def get_face_landmarks_5point(image):
    """Detect 5-point facial landmarks using MediaPipe
    Returns: dict with keys: left_eye, right_eye, nose, left_mouth, right_mouth
             Each value is (x, y) tuple
             Returns None if no face detected
    """
    if not MEDIAPIPE_AVAILABLE:
        return None
    
    try:
        # MediaPipe expects RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume it's already RGB from our pipeline
            image_rgb = image
        else:
            return None
        
        # Note: face_mesh is only defined when MEDIAPIPE_AVAILABLE is True
        # This code won't run if MediaPipe is disabled
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh if hasattr(mp, 'solutions') else None
        if not mp_face_mesh:
            return None
            
        face_mesh_local = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        h, w = image.shape[:2]
        results = face_mesh_local.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # MediaPipe Face Mesh landmark indices for 5-point:
        # Left eye: 33, Right eye: 263, Nose tip: 1, 
        # Left mouth corner: 61, Right mouth corner: 291
        landmarks_5pt = {
            'left_eye': (int(landmarks[33].x * w), int(landmarks[33].y * h)),
            'right_eye': (int(landmarks[263].x * w), int(landmarks[263].y * h)),
            'nose': (int(landmarks[1].x * w), int(landmarks[1].y * h)),
            'left_mouth': (int(landmarks[61].x * w), int(landmarks[61].y * h)),
            'right_mouth': (int(landmarks[291].x * w), int(landmarks[291].y * h))
        }
        
        return landmarks_5pt
        
    except Exception as e:
        print(f"⚠️ Landmark detection failed: {e}")
        return None

def align_face_5point(face_image, src_landmarks=None):
    """Align face using 5-point landmarks to ArcFace canonical template
    
    Args:
        face_image: RGB image (numpy array)
        src_landmarks: dict with 5 points or None (will detect automatically)
    
    Returns:
        Aligned 112x112 RGB face image
    """
    # ArcFace canonical 5-point template for 112x112 image
    dst_landmarks = np.array([
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # left mouth
        [70.7299, 92.2041]   # right mouth
    ], dtype=np.float32)
    
    # Detect landmarks if not provided
    if src_landmarks is None:
        src_landmarks = get_face_landmarks_5point(face_image)
    
    if src_landmarks is None:
        # Fallback: center crop with padding for pose tolerance
        h, w = face_image.shape[:2]
        
        # Use 80% of min dimension to include more context
        size = int(min(h, w) * 0.8)
        y1 = max(0, (h - size) // 2)
        x1 = max(0, (w - size) // 2)
        y2 = min(h, y1 + size)
        x2 = min(w, x1 + size)
        
        face_crop = face_image[y1:y2, x1:x2]
        
        from PIL import Image as PILImage
        face_pil = PILImage.fromarray(face_crop.astype('uint8'), 'RGB')
        face_aligned_pil = face_pil.resize((112, 112), PILImage.LANCZOS)  # Better quality
        return np.array(face_aligned_pil)
    
    # Convert landmarks dict to array
    src_points = np.array([
        src_landmarks['left_eye'],
        src_landmarks['right_eye'],
        src_landmarks['nose'],
        src_landmarks['left_mouth'],
        src_landmarks['right_mouth']
    ], dtype=np.float32)
    
    # Estimate similarity transform (scale, rotation, translation)
    from skimage import transform as trans
    tform = trans.SimilarityTransform()
    tform.estimate(src_points, dst_landmarks)
    
    # Apply transformation
    aligned_face = trans.warp(
        face_image,
        tform.inverse,
        output_shape=(112, 112),
        mode='edge',
        preserve_range=True
    ).astype(np.uint8)
    
    return aligned_face

def detect_faces_insightface(image):
    """Detect faces using InsightFace SCRFD model"""
    try:
        if not INSIGHTFACE_AVAILABLE:
            return []
        
        # Preprocess
        input_data = preprocess_image_for_detection(image)
        
        # Run detection
        input_name = detection_session.get_inputs()[0].name
        outputs = detection_session.run(None, {input_name: input_data})
        
        # Parse detections (simplified - you may need to adjust based on model output)
        # This is a placeholder - actual parsing depends on SCRFD output format
        faces = []
        # TODO: Parse detection outputs properly
        return faces
    except Exception as e:
        print(f"Detection error: {e}")
        return []

def extract_embedding_insightface(image, bbox=None, debug=False):
    """Extract 512-D face embedding using InsightFace ArcFace model with proper alignment"""
    try:
        if not INSIGHTFACE_AVAILABLE:
            return None
        
        if debug:
            print(f"  [PREPROCESS] Input shape: {image.shape}")
        
        # If bbox provided, crop face
        if bbox:
            x, y, w, h = bbox
            face_img = image[y:y+h, x:x+w]
        else:
            face_img = image
        
        # CRITICAL: Apply face alignment using 5-point landmarks
        face_aligned = align_face_5point(face_img)
        
        if debug:
            print(f"  [PREPROCESS] After alignment: {face_aligned.shape}")
        
        # Normalize: convert to float32, transpose to CHW, normalize to [-1, 1]
        face_normalized = face_aligned.astype(np.float32)
        face_normalized = (face_normalized - 127.5) / 127.5
        face_transposed = face_normalized.transpose(2, 0, 1)
        face_batch = np.expand_dims(face_transposed, axis=0).astype(np.float32)
        
        if debug:
            print(f"  [PREPROCESS] Normalized range: [{face_normalized.min():.3f}, {face_normalized.max():.3f}]")
            print(f"  [PREPROCESS] Output shape: {face_batch.shape}")
        
        # Extract embedding
        input_name = recognition_session.get_inputs()[0].name
        embedding = recognition_session.run(None, {input_name: face_batch})[0][0]
        
        # Normalize embedding (L2 normalization)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        if debug:
            print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
            print(f"  Embedding first 5 values: {embedding[:5].tolist()}")
        else:
            print(f"✓ Extracted embedding: shape={embedding.shape}, norm={np.linalg.norm(embedding):.4f}")
        
        return embedding.tolist()
    except Exception as e:
        print(f"❌ Embedding extraction error: {e}")
        import traceback
        traceback.print_exc()
        return None

# Global variables - FIXED STRUCTURE
embeddings_db = {
    'raw_embeddings': [],  # List of lists of embeddings per person (for training)
    'centroids': [],       # Computed centroids (for recognition)
    'names': [],           # List of names
    'categories': [],      # List of categories
    'is_trained': False    # Whether centroids have been computed
}

# Ensure data directory exists
os.makedirs('data', exist_ok=True)
TRAINING_DATA_FILE = 'data/face_embeddings.json'

def save_training_data():
    """Save embeddings database to disk"""
    try:
        with open(TRAINING_DATA_FILE, 'w') as f:
            json.dump(embeddings_db, f, indent=2)
        print(f"💾 Saved to {TRAINING_DATA_FILE} ({len(embeddings_db['names'])} people)")
    except Exception as e:
        print(f"❌ Failed to save training data: {e}")
        import traceback
        traceback.print_exc()

def load_training_data():
    """Load embeddings database from disk"""
    global embeddings_db
    try:
        if os.path.exists(TRAINING_DATA_FILE):
            with open(TRAINING_DATA_FILE, 'r') as f:
                loaded_data = json.load(f)
            
            # Migrate old format to new format
            if 'raw_embeddings' not in loaded_data:
                print("⚠️ Migrating old format to new format...")
                embeddings_db = {
                    'raw_embeddings': loaded_data.get('embeddings', []),
                    'centroids': [],
                    'names': loaded_data.get('names', []),
                    'categories': loaded_data.get('categories', []),
                    'is_trained': False
                }
                save_training_data()
            else:
                embeddings_db = loaded_data
            
            print(f"✓ Loaded {len(embeddings_db.get('names', []))} people from database")
        else:
            print(f"ℹ No existing training data found")
    except Exception as e:
        print(f"❌ Failed to load training data: {e}")

# Initialize at startup
print("="*60)
print("🚀 Initializing Face Recognition System")
print("="*60)
load_training_data()
print(f"✓ Loaded {len(embeddings_db['names'])} people from database")
print("="*60)

# Error handlers to ensure JSON responses
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"Unhandled exception: {str(e)}")
    import traceback
    traceback.print_exc()
    return jsonify({'error': str(e)}), 500

# Disable caching for all routes
@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera_test')
def camera_test():
    """Simple camera test page"""
    return render_template('camera_test.html')

@app.route('/capture')
def capture_page():
    return render_template('capture.html')

@app.route('/train')
def train_page():
    return render_template('train.html')

@app.route('/recognize')
def recognize_page():
    return render_template('recognize.html')

@app.route('/manage')
def manage_page():
    return render_template('manage.html')

@app.route('/get_trained_data', methods=['GET'])
def get_trained_data():
    """Return trained face data to client for recognition"""
    response_data = {
        'embeddings': embeddings_db.get('centroids', []),
        'names': embeddings_db.get('names', []),
        'categories': embeddings_db.get('categories', []),
        'is_trained': embeddings_db.get('is_trained', False),
        'threshold': FACE_MATCH_THRESHOLD,
        'embedding_dim': 512,  # Now using 512-D embeddings
        'use_backend': True    # Flag to indicate backend processing
    }
    return jsonify(response_data)

@app.route('/extract_embedding', methods=['POST'])
def extract_embedding():
    """
    NEW ENDPOINT: Extract 512-D embedding from image using InsightFace
    Input: Base64 image from client
    Output: 512-D embedding array
    """
    print("\n[BACKEND] Extract embedding request received")
    
    if not INSIGHTFACE_AVAILABLE:
        print("❌ InsightFace not available")
        return jsonify({
            'error': 'InsightFace models not available',
            'success': False
        }), 503
    
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'Image required', 'success': False}), 400
        
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        print(f"[BACKEND] Image decoded: {image.size}, mode={image.mode}")
        
        # Convert PIL image to numpy array (RGB format)
        frame_rgb = np.array(image)
        
        print(f"[BACKEND] Frame prepared: shape={frame_rgb.shape}")
        
        # Extract embedding (pass RGB numpy array) with debug logging
        embedding = extract_embedding_insightface(frame_rgb, debug=True)
        
        if embedding is None:
            print("❌ Failed to extract embedding")
            return jsonify({
                'error': 'Failed to extract embedding',
                'success': False
            }), 500
        
        print(f"✓ Successfully extracted {len(embedding)}-D embedding")
        
        return jsonify({
            'success': True,
            'embedding': embedding,
            'dimension': len(embedding)
        })
        
    except Exception as e:
        print(f"❌ Embedding extraction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/detect_landmarks', methods=['POST'])
def detect_landmarks():
    """
    MediaPipe endpoint for face detection and landmarking ONLY
    Input: Base64 image from client
    Output: Face bounding box and 468 landmarks
    """
    if not MEDIAPIPE_AVAILABLE:
        return jsonify({
            'error': 'MediaPipe not available',
            'face_detected': False
        }), 503
    
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'Image required'}), 400
        
        # Decode base64 image
        import base64
        import io
        from PIL import Image
        
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe face detection
        if not MEDIAPIPE_AVAILABLE:
            return jsonify({
                'face_detected': False,
                'error': 'MediaPipe not available',
                'landmarks': None,
                'bbox': None
            })
        
        # Use a local face_mesh instance since global one is disabled
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh if hasattr(mp, 'solutions') else None
        if not mp_face_mesh:
            return jsonify({
                'face_detected': False,
                'error': 'MediaPipe solutions not available',
                'landmarks': None,
                'bbox': None
            })
            
        face_mesh_local = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        results = face_mesh_local.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return jsonify({
                'face_detected': False,
                'landmarks': None,
                'bbox': None
            })
        
        # Get first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract landmarks
        h, w, _ = frame.shape
        landmarks = []
        x_coords = []
        y_coords = []
        
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append({'x': x, 'y': y})
            x_coords.append(x)
            y_coords.append(y)
        
        # Calculate bounding box
        bbox = {
            'x': min(x_coords),
            'y': min(y_coords),
            'width': max(x_coords) - min(x_coords),
            'height': max(y_coords) - min(y_coords)
        }
        
        return jsonify({
            'face_detected': True,
            'landmarks': landmarks,
            'bbox': bbox,
            'landmark_count': len(landmarks)
        })
        
    except Exception as e:
        print(f"MediaPipe detection error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/recognize', methods=['POST'])
def recognize():
    """
    Recognize a face using embedding from client (face-api.js)
    Input: JSON with 'embedding' field containing 128-D array from face-api.js
    Output: JSON with 'identity' and 'similarity' fields
    """
    try:
        data = request.json
        query_embedding = data.get('embedding')
        
        if not query_embedding:
            return jsonify({'error': 'Embedding required'}), 400
        
        # Check if database is trained
        if not embeddings_db.get('is_trained', False):
            return jsonify({
                'identity': 'Unknown',
                'similarity': 0.0,
                'category': 'N/A',
                'error': 'Model not trained. Please run training first.'
            }), 200
        
        if len(embeddings_db['names']) == 0:
            return jsonify({
                'identity': 'Unknown',
                'similarity': 0.0,
                'category': 'N/A',
                'error': 'No trained data available'
            }), 200
        
        # Validate embedding dimension (128-D from face-api.js OR 512-D from InsightFace)
        expected_dim = 512 if INSIGHTFACE_AVAILABLE else 128
        if len(query_embedding) not in [128, 512]:
            return jsonify({
                'identity': 'Unknown',
                'similarity': 0.0,
                'category': 'N/A',
                'error': f'Invalid embedding dimension: {len(query_embedding)}, expected {expected_dim}'
            }), 200
        
        query_embedding_np = np.array([query_embedding])
        
        print(f"\n[RECOGNITION] Query embedding shape: {query_embedding_np.shape}")
        print(f"[RECOGNITION] Database people: {embeddings_db['names']}")
        print(f"[RECOGNITION] Number of centroids: {len(embeddings_db.get('centroids', []))}")
        
        # Compare against all centroid embeddings using cosine similarity
        best_similarity = -1.0
        best_match_idx = -1
        
        for i, centroid_emb in enumerate(embeddings_db.get('centroids', [])):
            centroid_np = np.array([centroid_emb])
            centroid_norm = np.linalg.norm(centroid_np)
            
            # Compute cosine similarity
            similarity = cosine_similarity(query_embedding_np, centroid_np)[0][0]
            print(f"  Person {i} ({embeddings_db['names'][i]}): similarity={similarity:.4f}, centroid_norm={centroid_norm:.4f}")
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = i
        
        print(f"\n[RECOGNITION] Best match: {embeddings_db['names'][best_match_idx] if best_match_idx >= 0 else 'N/A'}")
        print(f"[RECOGNITION] Similarity: {best_similarity:.4f}, Threshold: {FACE_MATCH_THRESHOLD:.2f}")
        
        # Determine identity based on threshold
        if best_match_idx >= 0 and best_similarity >= FACE_MATCH_THRESHOLD:
            identity = embeddings_db['names'][best_match_idx]
            category = embeddings_db['categories'][best_match_idx]
            # Flag if similarity is medium-range (possible pose variation or lookalike)
            needs_verification = bool(best_similarity >= 0.50 and best_similarity < 0.75)
            if needs_verification:
                print("  ⚠️ Medium similarity - possible pose variation or lighting difference")
        else:
            identity = 'Unknown'
            category = 'N/A'
            needs_verification = False
            if best_match_idx >= 0:
                print(f"  ❌ Below threshold: {best_similarity:.3f} < {FACE_MATCH_THRESHOLD:.2f}")
                print(f"  💡 TIP: Retrain with multiple face angles (frontal, left, right, up, down)")
        
        response_data = {
            'identity': identity,
            'similarity': float(best_similarity),
            'category': category,
            'needs_verification': bool(needs_verification),
            'warning': 'Similarity in gray zone - verify identity' if needs_verification else None
        }
        
        print(f"[RECOGNITION] Sending response: {response_data}")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in recognize: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/capture_dataset', methods=['POST'])
def capture_dataset():
    """
    Capture face embedding from client (face-api.js)
    Input: JSON with 'embedding' (128-D array), 'name', 'category'
    """
    try:
        data = request.get_json()
        embedding = data.get('embedding')
        name = data.get('name', '').strip()
        category = data.get('category', 'student').strip()
        
        print(f"\n[CAPTURE] Request for: {name} ({category})")
        
        if not name:
            return jsonify({'status': 'error', 'message': 'Name is required'}), 400
        
        if not embedding:
            return jsonify({'status': 'error', 'message': 'Embedding is required'}), 400
        
        # Validate embedding dimension (128-D from face-api.js OR 512-D from InsightFace)
        if len(embedding) not in [128, 512]:
            return jsonify({
                'status': 'error',
                'message': f'Invalid embedding dimension: {len(embedding)}, expected 128 or 512'
            }), 400
        
        print(f"[CAPTURE] Embedding received: {len(embedding)}-D")
        
        # Find if person already exists
        person_idx = None
        for idx, existing_name in enumerate(embeddings_db['names']):
            if existing_name.lower() == name.lower():
                person_idx = idx
                break
        
        if person_idx is not None:
            # Person exists - append to their raw embeddings
            embeddings_db['raw_embeddings'][person_idx].append(embedding)
            sample_count = len(embeddings_db['raw_embeddings'][person_idx])
            print(f"✅ {name}: Added sample #{sample_count}")
            
            # Give helpful tips for better accuracy
            if sample_count == 5:
                print(f"💡 TIP: Rotate your head slightly left for next captures")
            elif sample_count == 10:
                print(f"💡 TIP: Rotate your head slightly right for next captures")
            elif sample_count == 15:
                print(f"💡 TIP: Tilt your head up slightly for next captures")
            elif sample_count == 20:
                print(f"💡 TIP: Tilt your head down slightly for next captures")
        else:
            # New person - create new entry
            embeddings_db['names'].append(name)
            embeddings_db['categories'].append(category)
            embeddings_db['raw_embeddings'].append([embedding])
            sample_count = 1
            print(f"✅ {name}: New person created with sample #1")
            print(f"💡 IMPORTANT: Capture at multiple angles (frontal, left, right, up, down) for best accuracy!")
            print(f"📊 Total people in database: {len(embeddings_db['names'])}")
        
        # Mark as needing retraining
        embeddings_db['is_trained'] = False
        
        # Save to disk
        save_training_data()
        
        return jsonify({
            'status': 'success',
            'message': f'Captured sample #{sample_count} for {name}',
            'sample_count': sample_count,
            'total_people': len(embeddings_db['names'])
        })
        
    except Exception as e:
        print(f"❌ Capture error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/capture_photo', methods=['POST'])
def capture_photo():
    """ALIAS for capture_dataset - used by capture.html"""
    return capture_dataset()

@app.route('/train_model', methods=['POST'])
def train_model():
    """
    Train model by computing centroid embeddings (mean of all samples per person)
    FIXED: Now preserves raw_embeddings and only updates centroids
    """
    try:
        global embeddings_db
        
        print("="*60)
        print("🔄 Starting Training Process")
        print("="*60)
        
        if len(embeddings_db.get('names', [])) == 0:
            return jsonify({
                'status': 'error',
                'message': 'No training data. Capture faces first.'
            }), 400
        
        names = embeddings_db['names']
        categories = embeddings_db['categories']
        all_raw_embeddings = embeddings_db['raw_embeddings']
        
        print(f"📊 Found {len(names)} people with raw embeddings")
        
        # Compute centroids (mean embedding per person)
        centroid_embeddings = []
        
        for i, name in enumerate(names):
            emb_list = all_raw_embeddings[i]
            
            # Validate structure
            if not isinstance(emb_list, list) or len(emb_list) == 0:
                print(f"  ⚠️ {name}: No embeddings, skipping")
                continue
            
            # Check all embeddings are 512-D (or 128-D for legacy)
            valid_embeddings = []
            for emb in emb_list:
                if isinstance(emb, list) and len(emb) in [128, 512]:
                    valid_embeddings.append(emb)
            
            if len(valid_embeddings) == 0:
                print(f"  ⚠️ {name}: No valid embeddings")
                continue
            
            # Compute centroid (mean)
            embeddings_array = np.array(valid_embeddings)
            centroid = np.mean(embeddings_array, axis=0)
            
            # CRITICAL: L2-normalize the centroid (same as query embeddings)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid = centroid / centroid_norm
            
            centroid_embeddings.append(centroid.tolist())
            
            print(f"  ✓ {name}: {len(valid_embeddings)} samples → centroid (norm={np.linalg.norm(centroid):.4f})")
        
        if len(centroid_embeddings) != len(names):
            return jsonify({
                'status': 'error',
                'message': 'Some people have no valid embeddings'
            }), 400
        
        # FIXED: Update only centroids, keep raw_embeddings intact
        embeddings_db['centroids'] = centroid_embeddings
        embeddings_db['is_trained'] = True
        
        # Save
        save_training_data()
        
        print("="*60)
        print(f"✅ Training Complete: {len(names)} people")
        print(f"📊 Raw samples preserved: {sum(len(e) for e in all_raw_embeddings)} total")
        print("="*60)
        
        return jsonify({
            'status': 'success',
            'trained_identities': len(names),
            'message': f'Trained on {len(names)} people'
        })
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/statistics', methods=['GET'])
def get_statistics():
    """Get statistics about trained faces"""
    try:
        total_photos = 0
        for emb_list in embeddings_db.get('raw_embeddings', []):
            if isinstance(emb_list, list):
                total_photos += len(emb_list)
        
        stats = {
            'total_people': len(embeddings_db.get('names', [])),
            'total_photos': total_photos,
            'is_trained': embeddings_db.get('is_trained', False),
            'people_details': []
        }
        
        for i, (name, category) in enumerate(zip(
            embeddings_db.get('names', []), 
            embeddings_db.get('categories', [])
        )):
            emb_list = embeddings_db['raw_embeddings'][i] if i < len(embeddings_db.get('raw_embeddings', [])) else []
            photo_count = len(emb_list) if isinstance(emb_list, list) else 0
            
            stats['people_details'].append({
                'id': i,
                'name': name,
                'category': category,
                'photo_count': photo_count
            })
        
        return jsonify(stats)
    except Exception as e:
        print(f"Error in statistics: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/delete_person/<int:person_id>', methods=['DELETE'])
def delete_person(person_id):
    """Delete a person's data - FIXED with bounds checking"""
    try:
        if person_id < 0 or person_id >= len(embeddings_db['names']):
            return jsonify({'error': 'Invalid person ID'}), 400
        
        name = embeddings_db['names'][person_id]
        
        # Use pop() to safely remove by index
        embeddings_db['names'].pop(person_id)
        embeddings_db['categories'].pop(person_id)
        embeddings_db['raw_embeddings'].pop(person_id)
        
        # Also remove centroid if trained
        if len(embeddings_db.get('centroids', [])) > person_id:
            embeddings_db['centroids'].pop(person_id)
        
        # Mark as needing retraining
        embeddings_db['is_trained'] = False
        
        save_training_data()
        
        print(f"✅ Deleted {name} (ID: {person_id})")
        print(f"📊 Remaining people: {len(embeddings_db['names'])}")
        
        return jsonify({'status': 'success', 'message': f'Deleted {name}'})
    except Exception as e:
        print(f"❌ Delete error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/export_data', methods=['GET'])
def export_data():
    """Export face data as JSON file"""
    return jsonify(embeddings_db)

@app.route('/debug_data', methods=['GET'])
def debug_data():
    """Debug endpoint to inspect data structure"""
    debug_info = []
    for i, name in enumerate(embeddings_db.get('names', [])):
        raw_emb = embeddings_db['raw_embeddings'][i] if i < len(embeddings_db.get('raw_embeddings', [])) else []
        sample_count = len(raw_emb) if isinstance(raw_emb, list) else 0
        
        debug_info.append({
            'id': i,
            'name': name,
            'category': embeddings_db['categories'][i],
            'samples': sample_count,
            'has_centroid': i < len(embeddings_db.get('centroids', []))
        })
    
    return jsonify({
        'total_people': len(embeddings_db['names']),
        'is_trained': embeddings_db.get('is_trained', False),
        'people': debug_info,
        'mediapipe_available': MEDIAPIPE_AVAILABLE
    })

if __name__ == '__main__':
    print("\n🌐 Server starting at: http://127.0.0.1:5000")
    print("   Open this URL in your browser\n")
    app.run(debug=True, host='127.0.0.1', port=5000)