from flask import Flask, render_template, Response, request, jsonify, send_file, session, redirect, url_for
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
import time
from functools import wraps
from werkzeug.security import check_password_hash, generate_password_hash
from urllib.parse import urlparse

print(f"✓ OpenCV imported: {cv2.__version__ if hasattr(cv2, '__version__') else 'version unknown'}")

# Initialize ONNX models for face detection and recognition
MODELS_DIR = 'models'

# Try better models first (higher quality recognition)
model_configs = [
    {
        'name': 'adaface',
        'detection': 'det_10g.onnx',
        'recognition': 'adaface_ir50_webface4m.onnx',
        'description': '512-D AdaFace (Better quality than ArcFace)'
    },
    {
        'name': 'buffalo_l',
        'detection': 'det_10g.onnx',
        'recognition': 'w600k_r50.onnx',
        'description': '512-D ArcFace (ResNet50)'
    },
    {
        'name': 'glint360k',
        'detection': 'det_10g.onnx',
        'recognition': 'glintr100.onnx',
        'description': '512-D ArcFace (ResNet100 - Better)'
    },
    {
        'name': 'current',
        'detection': 'det_10g.onnx', 
        'recognition': 'w600k_r50.onnx',
        'description': '512-D ArcFace (ResNet50)'
    }
]

# Load ONNX models
print("🔄 Loading InsightFace models...")
INSIGHTFACE_AVAILABLE = False
MODEL_NAME = 'Unknown'
MODEL_DESCRIPTION = 'None'

for config in model_configs:
    detection_model_path = os.path.join(MODELS_DIR, config['detection'])
    recognition_model_path = os.path.join(MODELS_DIR, config['recognition'])
    
    if os.path.exists(detection_model_path) and os.path.exists(recognition_model_path):
        try:
            detection_session = ort.InferenceSession(detection_model_path, providers=['CPUExecutionProvider'])
            recognition_session = ort.InferenceSession(recognition_model_path, providers=['CPUExecutionProvider'])
            INSIGHTFACE_AVAILABLE = True
            MODEL_NAME = config['name']
            MODEL_DESCRIPTION = config['description']
            print(f"✓ InsightFace models loaded: {MODEL_NAME}")
            print(f"  → Detection: {config['detection']}")
            print(f"  → Recognition: {config['recognition']} ({MODEL_DESCRIPTION})")
            break
        except Exception as e:
            print(f"  ⚠️ Failed to load {config['name']}: {e}")
            continue

if not INSIGHTFACE_AVAILABLE:
    print(f"❌ No InsightFace models could be loaded")
    print(f"💡 TIP: Run 'python scripts/download_models.py' to download models")

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FACEPASS_SECRET_KEY', 'facepass-secret-key')

# Session cookie defaults (tune via env if needed)
# Chrome can be stricter about cookies + redirects across hosts.
FACEPASS_COOKIE_SAMESITE = os.getenv('FACEPASS_COOKIE_SAMESITE', 'Lax')
FACEPASS_COOKIE_SECURE = os.getenv('FACEPASS_COOKIE_SECURE', '0').strip().lower() in ('1', 'true', 'yes', 'on')
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = FACEPASS_COOKIE_SAMESITE
app.config['SESSION_COOKIE_SECURE'] = FACEPASS_COOKIE_SECURE
app.config['SESSION_COOKIE_PATH'] = '/'

# Auth settings
# Preferred: manage users via scripts/user_admin.py (users stored in FACEPASS_USERS_FILE).
# Back-compat: you can still use a single env-based login by setting FACEPASS_USER/FACEPASS_PASS.
FACEPASS_USERS_FILE = os.getenv('FACEPASS_USERS_FILE', os.path.join('data', 'users.json'))
FACEPASS_ALLOW_ENV_LOGIN = os.getenv('FACEPASS_ALLOW_ENV_LOGIN', '1').strip().lower() in ('1', 'true', 'yes', 'on')

FACEPASS_USER = os.getenv('FACEPASS_USER', 'admin')
FACEPASS_PASS = os.getenv('FACEPASS_PASS', 'admin')

if FACEPASS_USER == 'admin' and FACEPASS_PASS == 'admin':
    print("⚠️ SECURITY: FACEPASS_USER/PASS are default 'admin'/'admin'. Create users via scripts/user_admin.py or set FACEPASS_USER/PASS.")


def _load_users_from_file(path: str):
    """Load users from a JSON file.

    Expected format:
      {"users": {"alice": {"password_hash": "...", "role": "admin|user"}}}
    """
    try:
        if not path or not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        users = payload.get('users')
        if not isinstance(users, dict):
            return {}
        normalized = {}
        for username, info in users.items():
            if not isinstance(username, str) or not username.strip():
                continue
            if not isinstance(info, dict):
                continue
            ph = info.get('password_hash')
            if not isinstance(ph, str) or not ph:
                continue
            normalized[username.strip()] = {
                'password_hash': ph,
                'role': (info.get('role') if isinstance(info.get('role'), str) else 'user')
            }
        return normalized
    except Exception as e:
        print(f"⚠️ Failed to load users file '{path}': {e}")
        return {}


def _authenticate_user(username: str, password: str):
    """Return (ok, role) for credentials."""
    username = (username or '').strip()
    password = password or ''

    if not username:
        return False, None

    users = _load_users_from_file(FACEPASS_USERS_FILE)
    info = users.get(username)
    if info is not None:
        try:
            if check_password_hash(info.get('password_hash', ''), password):
                role = info.get('role') or 'user'
                return True, role
        except Exception:
            return False, None

    # Back-compat single-user env credentials (optional)
    if FACEPASS_ALLOW_ENV_LOGIN and username == FACEPASS_USER and password == FACEPASS_PASS:
        return True, 'admin'

    return False, None


def _is_api_request() -> bool:
    """Heuristic: treat JSON/fetch calls as API requests (return 401 JSON instead of redirect)."""
    try:
        if request.is_json:
            return True
        accept = (request.headers.get('Accept') or '').lower()
        xrw = (request.headers.get('X-Requested-With') or '').lower()
        return ('application/json' in accept) or (xrw == 'xmlhttprequest')
    except Exception:
        return False


def _safe_next_url(next_url: str) -> str:
    """Return a safe relative redirect target.

    Prevents absolute redirects (e.g. http://127.0.0.1/...) which can cause
    the session cookie to be set on one host and then not sent to another.
    """
    try:
        next_url = (next_url or '').strip()
        if not next_url:
            return '/'

        parsed = urlparse(next_url)
        if parsed.scheme or parsed.netloc:
            # Absolute URL: only allow the path portion.
            path = parsed.path or '/'
            return path if path.startswith('/') else '/'

        # Relative path only
        if not next_url.startswith('/'):
            return '/'

        return next_url
    except Exception:
        return '/'


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if session.get('authenticated'):
            return view_func(*args, **kwargs)

        if _is_api_request() or request.path.startswith('/system_info'):
            return jsonify({'error': 'Authentication required'}), 401

        return redirect(url_for('login', next=request.path))

    return wrapper


def admin_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get('authenticated'):
            if _is_api_request() or request.path.startswith('/system_info'):
                return jsonify({'error': 'Authentication required'}), 401
            return redirect(url_for('login', next=request.path))

        if (session.get('role') or 'user') != 'admin':
            if _is_api_request():
                return jsonify({'error': 'Admin access required'}), 403
            return "Admin access required", 403

        return view_func(*args, **kwargs)

    return wrapper


def _load_users_payload(path: str) -> dict:
    """Load raw users payload; always returns {'users': {...}}."""
    if not path or not os.path.exists(path):
        return {'users': {}}

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {'users': {}}
        users = data.get('users')
        if not isinstance(users, dict):
            users = {}
        return {'users': users}
    except Exception as e:
        print(f"⚠️ Failed to read users file '{path}': {e}")
        return {'users': {}}


def _atomic_write_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write('\n')
    os.replace(tmp_path, path)


def _is_valid_username(username: str) -> bool:
    username = (username or '').strip()
    if not (3 <= len(username) <= 32):
        return False
    allowed = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-')
    return all(ch in allowed for ch in username)


def _count_admins(users: dict) -> int:
    try:
        count = 0
        for _uname, info in (users or {}).items():
            if isinstance(info, dict) and (info.get('role') or 'user') == 'admin':
                count += 1
        return count
    except Exception:
        return 0


@app.route('/users', methods=['GET', 'POST'])
@admin_required
def users_page():
    """Admin UI to create/list users."""
    message = None
    error = None

    path = FACEPASS_USERS_FILE
    payload = _load_users_payload(path)
    users = payload.get('users', {})

    current_user = session.get('user')

    if request.method == 'POST':
        action = (request.form.get('action') or 'create_update').strip().lower()
        username = (request.form.get('username') or '').strip()

        if action in ('create_update', 'create', 'update'):
            password = request.form.get('password') or ''
            role = (request.form.get('role') or 'user').strip().lower()
            if role not in ('user', 'admin'):
                role = 'user'

            if not _is_valid_username(username):
                error = "Username must be 3-32 chars and use only letters/numbers/._-"
            else:
                exists = isinstance(users.get(username), dict)
                if (not exists) and len(password) < 4:
                    error = 'Password must be at least 4 characters'
                elif exists and password and len(password) < 4:
                    error = 'Password must be at least 4 characters'
                else:
                    now = datetime.utcnow().isoformat() + 'Z'
                    prev = users.get(username) if isinstance(users.get(username), dict) else {}
                    created_at = prev.get('created_at') if isinstance(prev, dict) else None
                    password_hash = prev.get('password_hash') if isinstance(prev, dict) else None
                    if password:
                        password_hash = generate_password_hash(password)
                    if not isinstance(password_hash, str) or not password_hash:
                        error = 'Password is required for new users'
                    else:
                        users[username] = {
                            'password_hash': password_hash,
                            'role': role,
                            'created_at': created_at or now,
                            'updated_at': now,
                        }
                        payload['users'] = users
                        _atomic_write_json(path, payload)
                        message = f"User '{username}' created/updated"

        elif action in ('set_role', 'promote', 'demote'):
            role = (request.form.get('role') or '').strip().lower()
            if action == 'promote':
                role = 'admin'
            elif action == 'demote':
                role = 'user'

            if username not in users or not isinstance(users.get(username), dict):
                error = f"User '{username}' not found"
            elif role not in ('user', 'admin'):
                error = 'Invalid role'
            elif username == current_user and role != 'admin':
                error = 'You cannot demote your own admin account'
            else:
                # Prevent removing last admin
                admin_count = _count_admins(users)
                prev_role = (users.get(username) or {}).get('role') or 'user'
                if prev_role == 'admin' and role != 'admin' and admin_count <= 1:
                    error = 'Cannot demote the last admin'
                else:
                    now = datetime.utcnow().isoformat() + 'Z'
                    users[username]['role'] = role
                    users[username]['updated_at'] = now
                    payload['users'] = users
                    _atomic_write_json(path, payload)
                    message = f"Updated role for '{username}'"

        elif action == 'reset_password':
            new_password = request.form.get('password') or ''
            if username not in users or not isinstance(users.get(username), dict):
                error = f"User '{username}' not found"
            elif len(new_password) < 4:
                error = 'Password must be at least 4 characters'
            else:
                now = datetime.utcnow().isoformat() + 'Z'
                users[username]['password_hash'] = generate_password_hash(new_password)
                users[username]['updated_at'] = now
                payload['users'] = users
                _atomic_write_json(path, payload)
                message = f"Password reset for '{username}'"

        elif action == 'delete':
            if username not in users or not isinstance(users.get(username), dict):
                error = f"User '{username}' not found"
            elif username == current_user:
                error = 'You cannot delete your own logged-in account'
            else:
                # Prevent deleting last admin
                admin_count = _count_admins(users)
                prev_role = (users.get(username) or {}).get('role') or 'user'
                if prev_role == 'admin' and admin_count <= 1:
                    error = 'Cannot delete the last admin'
                else:
                    del users[username]
                    payload['users'] = users
                    _atomic_write_json(path, payload)
                    message = f"Deleted user '{username}'"

        else:
            error = 'Unknown action'

        # reload after any POST attempt
        payload = _load_users_payload(path)
        users = payload.get('users', {})

    # Build a safe view-model (no hashes)
    user_rows = []
    for uname in sorted(users.keys()):
        info = users.get(uname) if isinstance(users.get(uname), dict) else {}
        user_rows.append({
            'username': uname,
            'role': info.get('role', 'user'),
            'created_at': info.get('created_at', ''),
            'updated_at': info.get('updated_at', ''),
        })

    return render_template('users.html', users=user_rows, message=message, error=error, current_user=current_user)


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Simple username/password login."""
    error = None
    next_url = _safe_next_url(request.args.get('next') or '/')

    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = (request.form.get('password') or '')

        ok, role = _authenticate_user(username, password)
        if ok:
            session['authenticated'] = True
            session['user'] = username
            session['role'] = role or 'user'
            return redirect(next_url)

        error = 'Invalid username or password'

    return render_template('login.html', error=error, next_url=next_url)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Logging controls
# Set env var VERBOSE_LOGS=1 to enable detailed per-frame debug prints.
# You can also pass ?debug=1 on specific endpoints that support it.
VERBOSE_LOGS = os.getenv('VERBOSE_LOGS', '0').strip().lower() in ('1', 'true', 'yes', 'on')

_LAST_LOG_TS_BY_KEY = {}


def _throttled_print(key: str, message: str, min_interval_s: float = 1.5):
    """Print at most once per key per min_interval_s to avoid terminal spam."""
    now = time.monotonic()
    last = _LAST_LOG_TS_BY_KEY.get(key, 0.0)
    if now - last >= min_interval_s:
        _LAST_LOG_TS_BY_KEY[key] = now
        print(message)

# MediaPipe: Disabled for now (0.10+ API changed, and landmarks don't significantly improve ArcFace)
# The align_face_5point function will fall back to center crop + resize, which works well
MEDIAPIPE_AVAILABLE = False
print("⚠️ MediaPipe disabled - using center crop alignment (sufficient for ArcFace)")

# Face recognition threshold for verification (cosine similarity)
# IMPORTANT: Capture training data at multiple angles (frontal, left, right, up, down)
# You can tune these without code changes:
#   - set FACE_MATCH_THRESHOLD (e.g. 0.70)
#   - set SECOND_BEST_MARGIN (e.g. 0.10)
try:
    FACE_MATCH_THRESHOLD = float(os.getenv('FACE_MATCH_THRESHOLD', '0.70'))
except Exception:
    FACE_MATCH_THRESHOLD = 0.70

try:
    SECOND_BEST_MARGIN = float(os.getenv('SECOND_BEST_MARGIN', '0.10'))
except Exception:
    SECOND_BEST_MARGIN = 0.10

MIN_SAMPLES_FOR_TRAINING = 5  # Minimum samples per person for reliable recognition

# Lightweight face detector for cropping (prevents "center-crop" embeddings matching random people)
try:
    _cascade_path = os.path.join(getattr(cv2.data, 'haarcascades', ''), 'haarcascade_frontalface_default.xml')
    FACE_CASCADE = cv2.CascadeClassifier(_cascade_path) if _cascade_path else None
    if FACE_CASCADE is None or FACE_CASCADE.empty():
        FACE_CASCADE = None
        print("⚠️ OpenCV Haar cascade not available - face crop fallback disabled")
except Exception:
    FACE_CASCADE = None
    print("⚠️ Failed to initialize Haar cascade - face crop fallback disabled")


def detect_largest_face_bbox_opencv(image_rgb):
    """Return (x, y, w, h) for the largest detected face in an RGB frame, or None."""
    if FACE_CASCADE is None:
        return None

    try:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if faces is None or len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda r: int(r[2]) * int(r[3]))
        h_img, w_img = image_rgb.shape[:2]

        pad = int(0.25 * max(w, h))
        x0 = max(0, int(x) - pad)
        y0 = max(0, int(y) - pad)
        x1 = min(w_img, int(x + w) + pad)
        y1 = min(h_img, int(y + h) + pad)

        return (x0, y0, x1 - x0, y1 - y0)
    except Exception:
        return None

def preprocess_image_for_detection(img, input_size=(640, 640)):
    """Preprocess image for SCRFD face detection"""
    img_resized = cv2.resize(img, input_size)
    img_normalized = (img_resized.astype(np.float32) - 127.5) / 128.0
    img_transposed = img_normalized.transpose(2, 0, 1)
    img_batch = np.expand_dims(img_transposed, axis=0)
    return img_batch


def _nms_xyxy(boxes, scores, iou_thresh=0.4):
    """Non-maximum suppression for boxes in (x1, y1, x2, y2)."""
    if boxes.size == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def _distance2bbox(points, distance):
    """Decode distance predictions to bbox in xyxy."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def detect_largest_face_bbox_scrfd(image_rgb, score_thresh=0.55, nms_thresh=0.4, input_size=(640, 640), debug=False):
    """Detect faces via SCRFD (det_10g.onnx). Returns (x, y, w, h) in original image coords, or None."""
    if not INSIGHTFACE_AVAILABLE:
        return None

    try:
        h_img, w_img = image_rgb.shape[:2]

        input_data = preprocess_image_for_detection(image_rgb, input_size=input_size)
        input_name = detection_session.get_inputs()[0].name
        net_outs = detection_session.run(None, {input_name: input_data})

        # SCRFD exports commonly have 6 (scores+bbox) or 9 (scores+bbox+kps) outputs.
        if len(net_outs) not in (6, 9):
            if debug:
                shapes = [getattr(o, 'shape', None) for o in net_outs]
                _throttled_print('scrfd_unexpected_outputs', f"[SCRFD] Unexpected outputs={len(net_outs)} shapes={shapes}", 3.0)
            return None

        fmc = 3
        use_kps = (len(net_outs) == 9)
        scores_list = net_outs[0:fmc]
        bbox_list = net_outs[fmc:fmc * 2]
        # kps_list = net_outs[fmc * 2:fmc * 3] if use_kps else None

        feat_strides = [8, 16, 32]
        all_boxes = []
        all_scores = []

        for idx, stride in enumerate(feat_strides):
            scores = scores_list[idx]
            bbox_preds = bbox_list[idx]

            # Flatten shapes to (N,) and (N,4)
            scores = scores.reshape(-1)
            bbox_preds = bbox_preds.reshape(-1, 4) * float(stride)

            # Build anchor centers
            height = int(input_size[1] // stride)  # input_size is (w,h)?? We used cv2.resize(img, (w,h))
            width = int(input_size[0] // stride)
            # Note: preprocess_image_for_detection uses cv2.resize(img, input_size) where input_size=(w,h)
            # So feature map is (h/stride, w/stride)
            fm_h = int(input_size[1] // stride)
            fm_w = int(input_size[0] // stride)

            anchor_centers = np.stack(np.mgrid[0:fm_h, 0:fm_w][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * float(stride)).reshape((-1, 2))

            # Some exports have 2 anchors per location. Infer from output length.
            num_locs = anchor_centers.shape[0]
            if bbox_preds.shape[0] == num_locs * 2 and scores.shape[0] == num_locs * 2:
                anchor_centers = np.repeat(anchor_centers, 2, axis=0)
            elif bbox_preds.shape[0] != num_locs or scores.shape[0] != num_locs:
                # If we can't align anchors, skip this level.
                if debug:
                    _throttled_print(
                        f"scrfd_anchor_mismatch_{stride}",
                        f"[SCRFD] Anchor mismatch stride={stride} anchors={num_locs} bboxN={bbox_preds.shape[0]} scoreN={scores.shape[0]}",
                        3.0,
                    )
                continue

            pos = np.where(scores >= float(score_thresh))[0]
            if pos.size == 0:
                continue

            boxes = _distance2bbox(anchor_centers, bbox_preds)
            boxes = boxes[pos]
            sc = scores[pos]

            all_boxes.append(boxes)
            all_scores.append(sc)

        if not all_boxes:
            return None

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)

        keep = _nms_xyxy(boxes, scores, iou_thresh=float(nms_thresh))
        if not keep:
            return None

        boxes = boxes[keep]
        scores = scores[keep]

        # Choose the largest face (by area) among remaining boxes.
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best_i = int(np.argmax(areas))
        best = boxes[best_i]

        # Map from resized (input_size) coords back to original image coords.
        scale_x = float(w_img) / float(input_size[0])
        scale_y = float(h_img) / float(input_size[1])

        x1 = int(max(0, min(w_img - 1, round(best[0] * scale_x))))
        y1 = int(max(0, min(h_img - 1, round(best[1] * scale_y))))
        x2 = int(max(0, min(w_img - 1, round(best[2] * scale_x))))
        y2 = int(max(0, min(h_img - 1, round(best[3] * scale_y))))

        if x2 <= x1 or y2 <= y1:
            return None

        # Add small padding in original coordinates
        pad = int(0.15 * max(x2 - x1, y2 - y1))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w_img, x2 + pad)
        y2 = min(h_img, y2 + pad)

        return (x1, y1, x2 - x1, y2 - y1)
    except Exception as e:
        if debug:
            _throttled_print('scrfd_exception', f"[SCRFD] Detection error: {e}", 2.0)
        return None


def detect_largest_face_bbox(image_rgb, debug=False):
    """Prefer SCRFD detector; fall back to Haar if available."""
    bbox = detect_largest_face_bbox_scrfd(image_rgb, debug=debug)
    if bbox is not None:
        return bbox
    return detect_largest_face_bbox_opencv(image_rgb)

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

def validate_face_quality(image, min_brightness=30, max_brightness=235, blur_threshold=50.0):
    """Validate face image quality to prevent poor quality captures
    Returns: (is_valid, reason)
    """
    try:
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 1. Check blur using Laplacian variance (RELAXED for better acceptance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < blur_threshold:
            return False, f"Image too blurry (score: {laplacian_var:.1f}, need {blur_threshold})"
        
        # 2. Check brightness (WIDER RANGE for varied lighting)
        mean_brightness = np.mean(gray)
        if mean_brightness < min_brightness:
            return False, f"Image too dark (brightness: {mean_brightness:.1f}, need {min_brightness}+)"
        if mean_brightness > max_brightness:
            return False, f"Image too bright (brightness: {mean_brightness:.1f}, max {max_brightness})"
        
        # 3. Check contrast (RELAXED from 20 to 15)
        contrast = gray.std()
        if contrast < 15:
            return False, f"Image has low contrast (contrast: {contrast:.1f}, need 15+)"
        
        return True, "Quality OK"
        
    except Exception as e:
        print(f"Quality validation error: {e}")
        return True, "Validation skipped"  # Fail-open

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
        # Fallback: center crop with MORE padding for better pose tolerance
        h, w = face_image.shape[:2]
        
        # Use 90% of min dimension to include more facial context (improved from 80%)
        size = int(min(h, w) * 0.9)
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
        
        # Always operate on a detected face crop (never the full frame).
        if bbox is None:
            bbox = detect_largest_face_bbox(image, debug=debug)
            if bbox is None:
                if debug:
                    print("  [PREPROCESS] No face bbox detected")
                return None

        x, y, w, h = bbox
        h_img, w_img = image.shape[:2]
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(w_img, int(x + w))
        y1 = min(h_img, int(y + h))

        if x1 <= x0 or y1 <= y0:
            if debug:
                print(f"  [PREPROCESS] Invalid bbox after clamp: {(x0, y0, x1, y1)}")
            return None

        face_img = image[y0:y1, x0:x1]

        # Too-small crops yield junk embeddings
        if face_img.shape[0] < 60 or face_img.shape[1] < 60:
            if debug:
                print(f"  [PREPROCESS] Face crop too small: {face_img.shape}")
            return None
        
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
        else:
            print("❌ Zero-norm embedding detected")
            return None
        
        # Validate embedding quality (RELAXED from 0.01 to 0.005)
        embedding_std = np.std(embedding)
        if embedding_std < 0.005:  # Too uniform, likely invalid
            print(f"❌ Low-variance embedding (std: {embedding_std:.6f}, need 0.005+)")
            return None
        
        if debug:
            print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
            print(f"  Embedding std: {embedding_std:.4f}")
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
@login_required
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

@app.route('/system_info_page')
@login_required
def system_info_page():
    """System information page"""
    return render_template('system_info.html')

@app.route('/capture_tips')
def capture_tips():
    """Capture tips and guidelines page"""
    return render_template('capture_tips.html')

@app.route('/emotion')
def emotion_page():
    """Real-time emotion recognition page"""
    return render_template('emotion.html')

@app.route('/age_gender')
def age_gender_page():
    """Real-time age and gender detection page"""
    return render_template('age_gender.html')

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
    debug = VERBOSE_LOGS or (str(request.args.get('debug', '0')).strip() == '1')

    if debug:
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
        
        if debug:
            print(f"[BACKEND] Image decoded: {image.size}, mode={image.mode}")
        
        # Convert PIL image to numpy array (RGB format)
        frame_rgb = np.array(image)
        
        if debug:
            print(f"[BACKEND] Frame prepared: shape={frame_rgb.shape}")

        # Detect face bbox for proper cropping (prevents background/center-crop embeddings)
        # Prefer SCRFD (det_10g.onnx); fall back to Haar if available.
        bbox = detect_largest_face_bbox(frame_rgb, debug=debug)
        if bbox is None:
            if debug:
                _throttled_print('extract_embedding_no_face', "[BACKEND] No face detected (throttled)")
            return jsonify({
                'error': 'No face detected. Move closer and face the camera.',
                'success': False
            }), 400

        x, y, w, h = bbox
        face_roi = frame_rgb[y:y + h, x:x + w]

        # Validate face quality on the face region (not the whole frame)
        is_valid, quality_reason = validate_face_quality(face_roi)
        if not is_valid:
            if debug:
                print(f"⚠️ Quality check: {quality_reason}")
            return jsonify({
                'error': f'Poor image quality: {quality_reason}',
                'success': False,
                'quality_issue': quality_reason
            }), 400
        
        if debug:
            print(f"✓ Quality check passed: {quality_reason}")
        
        # Extract embedding from detected face crop. Enable per-frame debug only when requested.
        embedding = extract_embedding_insightface(frame_rgb, bbox=bbox, debug=debug)
        
        if embedding is None:
            print("❌ Failed to extract embedding")
            return jsonify({
                'error': 'Failed to extract embedding',
                'success': False
            }), 500
        
        if debug:
            print(f"✓ Successfully extracted {len(embedding)}-D embedding")
        
        return jsonify({
            'success': True,
            'embedding': embedding,
            'dimension': len(embedding),
            'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
        })
        
    except Exception as e:
        print(f"❌ Embedding extraction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500

# Motion tracking for emotion detection
previous_frame = None
motion_history = []

@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    """
    Analyze facial expression and emotion from image
    Input: Base64 image from client
    Output: Emotion, facial features, and landmark positions
    """
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'Image required'}), 400
        
        # Decode base64 image
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            
            # Convert directly to numpy array via OpenCV
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame_bgr is None:
                raise ValueError("Failed to decode image")
                
        except Exception as e:
            print(f"Image decode error: {e}")
            return jsonify({'error': 'Invalid image data', 'face_detected': False}), 400
        
        # Convert to numpy array
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Calculate motion/movement intensity
        global previous_frame, motion_history
        motion_score = 0.0
        
        if previous_frame is not None:
            frame_diff = cv2.absdiff(previous_frame, gray)
            motion_score = float(np.mean(frame_diff))
            motion_history.append(motion_score)
            if len(motion_history) > 10:
                motion_history.pop(0)
        
        previous_frame = gray.copy()
        
        # Calculate movement patterns
        is_still = motion_score < 5.0  # Very little movement
        is_erratic = motion_score > 20.0  # Wild erratic movement
        
        if len(motion_history) > 5:
            motion_variance = float(np.std(motion_history))
            is_erratic = motion_variance > 15.0 or motion_score > 25.0
        
        # Convert to Python bool immediately
        is_still = bool(is_still)
        is_erratic = bool(is_erratic)
        
        # Try using DeepFace AI model for better emotion detection
        try:
            from deepface import DeepFace

            # Focus analysis on the largest detected face crop to avoid background/noise
            bbox = detect_largest_face_bbox(frame_rgb, debug=False)
            if bbox is None:
                return jsonify({
                    'face_detected': False,
                    'error': 'No face detected'
                })

            x, y, w, h = bbox
            face_bgr = frame_bgr[y:y + h, x:x + w]
            if face_bgr is None or face_bgr.size == 0:
                return jsonify({
                    'face_detected': False,
                    'error': 'Invalid face crop'
                })

            # Preprocessing: Enhance image quality for better detection
            enhanced_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
            enhanced_face[:, :, 0] = cv2.equalizeHist(enhanced_face[:, :, 0])
            enhanced_face = cv2.cvtColor(enhanced_face, cv2.COLOR_YCrCb2BGR)
            enhanced_face = cv2.GaussianBlur(enhanced_face, (3, 3), 0)

            # Analyze emotions using DeepFace.
            # Use detector_backend='skip' because we already cropped to the face.
            # If DeepFace version doesn't support 'skip', fall back to 'retinaface'.
            try:
                result = DeepFace.analyze(
                    img_path=enhanced_face,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='skip',
                    silent=True
                )
            except Exception:
                result = DeepFace.analyze(
                    img_path=enhanced_face,
                    actions=['emotion'],
                    enforce_detection=True,
                    detector_backend='retinaface',
                    silent=True
                )
            
            # DeepFace returns a list, get first face
            if isinstance(result, list):
                result = result[0]
            
            # Get emotion with highest confidence
            emotions = result.get('emotion', {}) or {}

            # DeepFace commonly returns emotion scores as percentages (0..100)
            ranked = sorted(((str(k).lower(), float(v)) for k, v in emotions.items()), key=lambda kv: kv[1], reverse=True)
            if not ranked:
                return jsonify({'face_detected': False, 'error': 'No emotion scores returned'})

            dominant_emotion, dominant_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else 0.0

            # Guardrail: if confidence is low or ambiguous, avoid over-confident "Happy"
            # This dramatically reduces the "everything is happy" failure mode.
            if dominant_score < 45.0 or (dominant_score - second_score) < 10.0:
                dominant_emotion = 'neutral'
                dominant_score = float(emotions.get('neutral', dominant_score))

            confidence = float(dominant_score) / 100.0
            
            # Map DeepFace emotions to our custom emotions
            emotion_mapping = {
                'happy': 'Happy',
                'sad': 'Sad',
                'angry': 'Angry',
                'surprise': 'Surprised',
                'fear': 'Fearful',
                'disgust': 'Disgusted',
                'neutral': 'Locked In' if emotions.get('angry', 0) > 15 or emotions.get('neutral', 0) > 70 else 'Neutral'
            }
            
            mapped_emotion = emotion_mapping.get(dominant_emotion, 'Neutral')
            
            # Detect eyebrows raised (surprise/anger indicator)
            eyebrows_raised = bool(dominant_emotion in ['surprise', 'fear'] or emotions.get('surprise', 0) > 25)
            
            # Enhanced emotion detection with motion and expression analysis
            # (Keep these rules non-offensive and based on signals we can justify.)
            # "Goofy": strong disgust with some happy signal (often tongue-out / playful faces)
            if dominant_emotion == 'disgust' and float(emotions.get('happy', 0)) > 25.0 and float(emotions.get('disgust', 0)) > 40.0:
                mapped_emotion = 'Goofy'
                confidence = max(confidence, 0.85)
            
            # RAMPAGING: Erratic wild movement + aggressive emotions (DEADLY)
            elif is_erratic and (emotions.get('angry', 0) > 30 or emotions.get('fear', 0) > 30):
                mapped_emotion = 'Rampaging'
                confidence = 0.95
            
            # ANGRY: Eyebrows raised + angry emotion
            elif eyebrows_raised and emotions.get('angry', 0) > 30:
                mapped_emotion = 'Angry'
                confidence = 0.90
            elif dominant_emotion == 'angry' or emotions.get('angry', 0) > 50:
                mapped_emotion = 'Angry'
                confidence = float(emotions.get('angry', 50) / 100.0)
            
            # LOCKED IN: Still/steady + focused neutral expression
            elif is_still and dominant_emotion == 'neutral' and emotions.get('neutral', 0) > 40:
                mapped_emotion = 'Locked In'
                confidence = 0.90
            
            # Other custom emotions
            elif dominant_emotion == 'happy' and float(emotions.get('angry', 0)) > 20.0 and float(emotions.get('happy', 0)) > 60.0:
                mapped_emotion = 'Psychotic'
            elif dominant_emotion == 'sad' and confidence > 0.7:
                mapped_emotion = 'Sobbing'
            elif dominant_emotion == 'surprise' and confidence > 0.7:
                mapped_emotion = 'Shocked'
            elif emotions.get('neutral', 0) > 80:
                mapped_emotion = 'Calm'
            
            emotion_result = {
                'face_detected': True,
                'emotion': mapped_emotion,
                'confidence': float(confidence),
                'features': {
                    'smile_detected': bool(float(emotions.get('happy', 0)) > 60.0),
                    'eyes_open': True,
                    'eyebrows_raised': bool(eyebrows_raised),
                    'mouth_open': bool(dominant_emotion in ['surprise', 'fear'])
                },
                'motion': {
                    'score': float(motion_score),
                    'is_still': bool(is_still),
                    'is_erratic': bool(is_erratic)
                },
                'raw_emotions': {k: float(v) for k, v in emotions.items()},
                'face_quality': {
                    'brightness': float(np.mean(gray)),
                    'sharpness': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                    'face_size': 'Good'
                }
            }
            
            motion_text = ""
            if is_erratic:
                motion_text = " [ERRATIC MOVEMENT - DEADLY]"
            elif is_still:
                motion_text = " [STILL/STEADY]"
            
            print(f"[DEEPFACE] Detected: {mapped_emotion} ({confidence:.2f}) - Raw: {dominant_emotion} ({emotions[dominant_emotion]:.1f}%){motion_text} Motion: {motion_score:.1f}")
            
            return jsonify(emotion_result)
            
        except Exception as deepface_error:
            print(f"DeepFace error, falling back to Haar Cascade: {deepface_error}")
            # Fallback to original Haar Cascade method
        
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Enhanced emotion detection based on facial analysis
        emotion_result = {
            'face_detected': True,
            'emotion': 'Neutral',
            'confidence': 0.0,
            'features': {
                'smile_detected': False,
                'eyes_open': True,
                'eyebrows_raised': False,
                'mouth_open': False
            },
            'face_quality': {
                'brightness': 0,
                'sharpness': 0,
                'face_size': 'Good'
            }
        }
        
        # Calculate quality metrics
        brightness = float(np.mean(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        emotion_result['face_quality']['brightness'] = brightness
        emotion_result['face_quality']['sharpness'] = sharpness
        
        # Try to detect face using Haar Cascade with improved parameters
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            
            # Better face detection parameters
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            
            if len(faces) > 0:
                # Get the largest face
                face = max(faces, key=lambda f: f[2] * f[3])
                (x, y, w, h) = face
                
                face_roi_gray = gray[y:y+h, x:x+w]
                face_roi_color = frame_bgr[y:y+h, x:x+w]
                
                # Simple metrics
                face_std = np.std(face_roi_gray)
                face_mean = np.mean(face_roi_gray)
                
                # Detect features - STRICT smile detection to avoid false positives
                eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
                smiles = smile_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.4, minNeighbors=18, minSize=(25, 25))
                
                has_eyes = len(eyes) >= 2
                has_smile = len(smiles) > 0
                
                emotion_result['features']['eyes_open'] = bool(has_eyes)
                emotion_result['features']['smile_detected'] = bool(has_smile)
                
                # Analyze face regions
                h_third = h // 3
                lower_face = face_roi_gray[2*h_third:, :]
                upper_face = face_roi_gray[:h_third, :]
                
                lower_mean = np.mean(lower_face)
                upper_mean = np.mean(upper_face)
                
                # Mouth open: dark lower region
                mouth_open = lower_mean < (face_mean * 0.8)
                emotion_result['features']['mouth_open'] = bool(mouth_open)
                
                # Eyebrows raised: bright upper region
                brows_raised = upper_mean > (face_mean * 1.05)
                emotion_result['features']['eyebrows_raised'] = bool(brows_raised)
                
                # SIMPLE EMOTION DETECTION - Clear rules
                
                # Priority 1: Happy (smile detected)
                if has_smile:
                    if face_std > 50:
                        emotion_result['emotion'] = 'Psychotic'
                        emotion_result['confidence'] = 0.85
                    else:
                        emotion_result['emotion'] = 'Happy'
                        emotion_result['confidence'] = 0.90
                
                # Priority 2: Surprised/Shocked (mouth open + eyebrows)
                elif mouth_open and brows_raised:
                    emotion_result['emotion'] = 'Shocked'
                    emotion_result['confidence'] = 0.85
                
                elif mouth_open:
                    emotion_result['emotion'] = 'Surprised'
                    emotion_result['confidence'] = 0.80
                
                # Priority 3: Angry (high tension, no smile)
                elif face_std > 45 and not has_smile:
                    if face_std > 55:
                        emotion_result['emotion'] = 'Rampaging'
                        emotion_result['confidence'] = 0.80
                    else:
                        emotion_result['emotion'] = 'Angry'
                        emotion_result['confidence'] = 0.75
                
                # Priority 4: Sad/Sobbing (no smile, low energy)
                elif not has_smile and not has_eyes:
                    emotion_result['emotion'] = 'Sobbing'
                    emotion_result['confidence'] = 0.75
                
                elif not has_smile and face_mean < 100:
                    emotion_result['emotion'] = 'Sad'
                    emotion_result['confidence'] = 0.70
                
                # Priority 5: Disgusted (eyebrows raised, no smile)
                elif brows_raised and not has_smile:
                    emotion_result['emotion'] = 'Disgusted'
                    emotion_result['confidence'] = 0.70
                
                # Priority 6: Locked In (serious, focused, steady face with eyes)
                elif has_eyes and not has_smile and face_std > 28 and face_std < 42:
                    emotion_result['emotion'] = 'Locked In'
                    emotion_result['confidence'] = 0.85
                
                # Priority 7: Thinking (pondering)
                elif face_std > 30:
                    emotion_result['emotion'] = 'Thinking'
                    emotion_result['confidence'] = 0.65
                
                # Default: Neutral or Calm
                else:
                    if face_std < 25:
                        emotion_result['emotion'] = 'Calm'
                        emotion_result['confidence'] = 0.70
                    else:
                        emotion_result['emotion'] = 'Neutral'
                        emotion_result['confidence'] = 0.65
                
                print(f"[EMOTION] Detected: {emotion_result['emotion']} ({emotion_result['confidence']:.2f}) - Smile:{has_smile}, Eyes:{has_eyes}, Mouth:{mouth_open}, Std:{face_std:.1f}")
                
                # Convert all numpy types to Python types for JSON serialization
                emotion_result['confidence'] = float(emotion_result['confidence'])
                emotion_result['face_detected'] = True
                
        except Exception as e:
            print(f"Face detection error: {e}")
            # Fallback to varied analysis
            import random
            emotions = ['Happy', 'Calm', 'Surprised', 'Thinking', 'Angry', 'Neutral']
            weights = [0.25, 0.2, 0.15, 0.15, 0.15, 0.1]
            emotion_result['emotion'] = random.choices(emotions, weights=weights)[0]
            emotion_result['confidence'] = 0.5 + random.random() * 0.3
            
            # Random features based on emotion
            emotion_result['features']['smile_detected'] = emotion_result['emotion'] in ['Happy', 'Psychotic']
            emotion_result['features']['eyebrows_raised'] = emotion_result['emotion'] in ['Surprised', 'Shocked', 'Angry', 'Rampaging']
            emotion_result['features']['mouth_open'] = emotion_result['emotion'] in ['Surprised', 'Shocked', 'Sobbing', 'Rampaging']
        
        return jsonify(emotion_result)
        
    except Exception as e:
        print(f"Emotion analysis error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'face_detected': False}), 500

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
        debug = VERBOSE_LOGS or (str(request.args.get('debug', '0')).strip() == '1')
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
        
        # Validate embedding dimension against the trained centroids.
        centroids = embeddings_db.get('centroids', [])
        if not centroids:
            return jsonify({
                'identity': 'Unknown',
                'similarity': 0.0,
                'category': 'N/A',
                'error': 'No trained centroids available'
            }), 200

        centroid_dim = len(centroids[0])
        if len(query_embedding) != centroid_dim:
            return jsonify({
                'identity': 'Unknown',
                'similarity': 0.0,
                'category': 'N/A',
                'error': f'Embedding dimension mismatch: got {len(query_embedding)}, expected {centroid_dim}'
            }), 200
        
        query_embedding_np = np.array([query_embedding])

        if debug:
            print(f"\n[RECOGNITION] Query embedding shape: {query_embedding_np.shape}")
            print(f"[RECOGNITION] Database people: {embeddings_db['names']}")
            print(f"[RECOGNITION] Number of centroids: {len(embeddings_db.get('centroids', []))}")
        
        # Compare against all centroid embeddings using cosine similarity
        similarities = []
        
        for i, centroid_emb in enumerate(centroids):
            centroid_np = np.array([centroid_emb])
            centroid_norm = np.linalg.norm(centroid_np)
            
            # Compute cosine similarity
            similarity = cosine_similarity(query_embedding_np, centroid_np)[0][0]
            similarities.append((i, similarity))
            if debug:
                print(f"  Person {i} ({embeddings_db['names'][i]}): similarity={similarity:.4f}, centroid_norm={centroid_norm:.4f}")
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        best_match_idx = similarities[0][0] if similarities else -1
        best_similarity = similarities[0][1] if similarities else -1.0
        second_best_similarity = similarities[1][1] if len(similarities) > 1 else -1.0
        
        if debug:
            print(f"\n[RECOGNITION] Best match: {embeddings_db['names'][best_match_idx] if best_match_idx >= 0 else 'N/A'}")
            print(f"[RECOGNITION] Best similarity: {best_similarity:.4f}, Threshold: {FACE_MATCH_THRESHOLD:.2f}")
            if len(similarities) > 1:
                print(f"[RECOGNITION] 2nd best: {embeddings_db['names'][similarities[1][0]]} ({second_best_similarity:.4f})")
                print(f"[RECOGNITION] Margin: {best_similarity - second_best_similarity:.4f} (min: {SECOND_BEST_MARGIN:.2f})")
        
        # Strict verification: must pass threshold AND have sufficient margin from 2nd best
        margin = best_similarity - second_best_similarity
        
        if best_match_idx >= 0 and best_similarity >= FACE_MATCH_THRESHOLD:
            # Additional check: ensure clear winner (avoid ambiguous matches)
            if len(similarities) > 1 and margin < SECOND_BEST_MARGIN:
                identity = 'Unknown'
                category = 'N/A'
                needs_verification = True
                if debug:
                    print(f"  ⚠️ REJECTED: Ambiguous match - too close to 2nd best (margin: {margin:.3f})")
                    print(f"  💡 TIP: Could be confused with {embeddings_db['names'][similarities[1][0]]}")
            else:
                identity = embeddings_db['names'][best_match_idx]
                category = embeddings_db['categories'][best_match_idx]
                # Flag if similarity is medium-range
                needs_verification = bool(best_similarity >= 0.65 and best_similarity < 0.75)
                if debug:
                    if needs_verification:
                        print("  ℹ️ Medium confidence - acceptable match")
                    else:
                        print(f"  ✅ High confidence match: {identity} ({best_similarity:.3f})")
        else:
            identity = 'Unknown'
            category = 'N/A'
            needs_verification = False
            if best_match_idx >= 0:
                if debug:
                    print(f"  ❌ Below threshold: {best_similarity:.3f} < {FACE_MATCH_THRESHOLD:.2f}")
                    print(f"  💡 TIP: Retrain with multiple face angles (frontal, left, right, up, down)")
        
        response_data = {
            'identity': identity,
            'similarity': float(best_similarity),
            'category': category,
            'needs_verification': bool(needs_verification),
            'warning': 'Similarity in gray zone - verify identity' if needs_verification else None,
            'second_best_similarity': float(second_best_similarity) if second_best_similarity > 0 else None,
            'margin': float(margin) if len(similarities) > 1 else None
        }
        
        if debug:
            print(f"[RECOGNITION] Sending response: {response_data}")
        else:
            # Minimal summary, throttled to avoid spamming when running realtime loops.
            _throttled_print(
                'recognize_summary',
                f"[RECOGNITION] {response_data['identity']} sim={response_data['similarity']:.3f}",
                min_interval_s=2.0,
            )
        
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
        image_data = data.get('image')
        name = data.get('name', '').strip()
        category = data.get('category', 'student').strip()
        
        print(f"\n[CAPTURE] Request for: {name} ({category})")
        
        if not name:
            return jsonify({'status': 'error', 'message': 'Name is required'}), 400
        
        # Prefer server-side embedding extraction from the submitted image.
        # This guarantees we only embed the detected face region (not the full frame)
        # and keeps training embeddings consistent.
        if image_data:
            try:
                if ',' in image_data:
                    image_data = image_data.split(',')[1]

                image_bytes = base64.b64decode(image_data)
                pil_img = Image.open(io.BytesIO(image_bytes))
                frame_rgb = np.array(pil_img)

                if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
                    return jsonify({'status': 'error', 'message': 'Invalid image format'}), 400

                bbox = detect_largest_face_bbox(frame_rgb, debug=False)
                if bbox is None:
                    return jsonify({'status': 'error', 'message': 'No face detected. Move closer and face the camera.'}), 400

                x, y, w, h = bbox
                face_roi = frame_rgb[y:y + h, x:x + w]
                is_valid, quality_reason = validate_face_quality(face_roi)
                if not is_valid:
                    return jsonify({'status': 'error', 'message': f'Poor image quality: {quality_reason}'}), 400

                if not INSIGHTFACE_AVAILABLE:
                    return jsonify({'status': 'error', 'message': 'InsightFace models not available on server'}), 503

                embedding = extract_embedding_insightface(frame_rgb, bbox=bbox, debug=False)
                if not embedding:
                    return jsonify({'status': 'error', 'message': 'Failed to extract face embedding. Try better lighting and a closer face.'}), 500

                print(f"[CAPTURE] Server embedding extracted: {len(embedding)}-D")

            except Exception as e:
                print(f"❌ Capture image decode/extract error: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'status': 'error', 'message': 'Invalid image data'}), 400
        else:
            # No image provided: only allow embedding-only capture when InsightFace is unavailable.
            if INSIGHTFACE_AVAILABLE:
                return jsonify({'status': 'error', 'message': 'Image is required for capture (server-side embedding enabled).'}), 400

            if not embedding:
                return jsonify({'status': 'error', 'message': 'Embedding is required'}), 400

            if len(embedding) != 128:
                return jsonify({'status': 'error', 'message': f'Invalid embedding dimension: {len(embedding)}, expected 128'}), 400

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
        
        # Compute centroids (mean embedding per person) with outlier removal
        centroid_embeddings = []
        skipped_people = []
        
        for i, name in enumerate(names):
            emb_list = all_raw_embeddings[i]
            
            # Validate structure
            if not isinstance(emb_list, list) or len(emb_list) == 0:
                print(f"  ⚠️ {name}: No embeddings, skipping")
                skipped_people.append(name)
                continue
            
            # Check all embeddings and ensure consistent dimensions
            valid_embeddings = []
            embedding_dims = {}
            
            for emb in emb_list:
                if isinstance(emb, list) and len(emb) in [128, 512]:
                    dim = len(emb)
                    if dim not in embedding_dims:
                        embedding_dims[dim] = []
                    embedding_dims[dim].append(emb)
            
            # Use the dimension with the most samples
            if not embedding_dims:
                print(f"  ⚠️ {name}: No valid embeddings")
                skipped_people.append(name)
                continue
            
            # Select the most common dimension
            primary_dim = max(embedding_dims.keys(), key=lambda d: len(embedding_dims[d]))
            valid_embeddings = embedding_dims[primary_dim]
            
            if len(embedding_dims) > 1:
                other_dims = [d for d in embedding_dims.keys() if d != primary_dim]
                print(f"  ℹ️ {name}: Using {len(valid_embeddings)} samples of {primary_dim}D (ignored {sum(len(embedding_dims[d]) for d in other_dims)} samples with different dimensions)")
            
            if len(valid_embeddings) == 0:
                print(f"  ⚠️ {name}: No valid embeddings")
                skipped_people.append(name)
                continue
            
            # Check minimum samples requirement
            if len(valid_embeddings) < MIN_SAMPLES_FOR_TRAINING:
                print(f"  ⚠️ {name}: Only {len(valid_embeddings)} samples (need {MIN_SAMPLES_FOR_TRAINING}), skipping")
                skipped_people.append(name)
                continue
            
            # Remove outliers using median-based method (more robust than mean)
            # Now safe to convert to array since all embeddings have same dimension
            embeddings_array = np.array(valid_embeddings, dtype=np.float32)
            
            if len(valid_embeddings) >= 5:
                # Compute pairwise similarities
                median_embedding = np.median(embeddings_array, axis=0)
                median_norm = np.linalg.norm(median_embedding)
                if median_norm > 0:
                    median_embedding = median_embedding / median_norm
                
                # Filter out embeddings too far from median (RELAXED from 0.85 to 0.75)
                filtered_embeddings = []
                for emb in valid_embeddings:
                    emb_np = np.array(emb, dtype=np.float32)
                    emb_np = emb_np / np.linalg.norm(emb_np) if np.linalg.norm(emb_np) > 0 else emb_np
                    similarity_to_median = np.dot(emb_np, median_embedding)
                    
                    # Keep embeddings with similarity > 0.75 to median (more lenient)
                    if similarity_to_median > 0.75:
                        filtered_embeddings.append(emb)
                
                if len(filtered_embeddings) >= MIN_SAMPLES_FOR_TRAINING:
                    embeddings_array = np.array(filtered_embeddings, dtype=np.float32)
                    outliers_removed = len(valid_embeddings) - len(filtered_embeddings)
                    if outliers_removed > 0:
                        print(f"  🧹 {name}: Removed {outliers_removed} outlier(s)")
            
            # Compute centroid (mean of filtered embeddings)
            centroid = np.mean(embeddings_array, axis=0)
            
            # CRITICAL: L2-normalize the centroid (same as query embeddings)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid = centroid / centroid_norm
            else:
                print(f"  ⚠️ {name}: Zero-norm centroid, skipping")
                skipped_people.append(name)
                continue
            
            centroid_embeddings.append(centroid.tolist())
            
            print(f"  ✓ {name}: {len(embeddings_array)} samples → centroid (norm={np.linalg.norm(centroid):.4f})")
        
        if len(centroid_embeddings) == 0:
            issues_msg = ', '.join(skipped_people)
            return jsonify({
                'status': 'error',
                'message': f'No people could be trained. Issues: {issues_msg}'
            }), 400
        
        if len(skipped_people) > 0:
            skipped_msg = ', '.join(skipped_people)
            print(f"⚠️ Warning: {len(skipped_people)} people skipped: {skipped_msg}")
        
        # FIXED: Update only centroids, keep raw_embeddings intact
        embeddings_db['centroids'] = centroid_embeddings
        embeddings_db['is_trained'] = True
        
        # Save
        save_training_data()
        
        print("="*60)
        print(f"✅ Training Complete: {len(centroid_embeddings)} people trained")
        if len(skipped_people) > 0:
            print(f"⚠️ Skipped {len(skipped_people)} people (insufficient samples)")
        print(f"📊 Raw samples preserved: {sum(len(e) for e in all_raw_embeddings)} total")
        print(f"🔒 Security: Threshold={FACE_MATCH_THRESHOLD:.2f}, Margin={SECOND_BEST_MARGIN:.2f}")
        print("="*60)
        
        message = f'Trained on {len(centroid_embeddings)} people'
        if len(skipped_people) > 0:
            message += f' (Skipped {len(skipped_people)}: need {MIN_SAMPLES_FOR_TRAINING}+ samples)'
        
        return jsonify({
            'status': 'success',
            'trained_identities': len(centroid_embeddings),
            'skipped_identities': len(skipped_people),
            'skipped_names': skipped_people,
            'message': message
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
        'mediapipe_available': MEDIAPIPE_AVAILABLE,
        'insightface_available': INSIGHTFACE_AVAILABLE,
        'model_name': MODEL_NAME,
        'model_description': MODEL_DESCRIPTION,
        'threshold': FACE_MATCH_THRESHOLD,
        'second_best_margin': SECOND_BEST_MARGIN,
        'min_samples': MIN_SAMPLES_FOR_TRAINING
    })

@app.route('/system_info', methods=['GET'])
@login_required
def system_info():
    """System information endpoint"""
    import platform
    
    return jsonify({
        'system': {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'opencv_version': cv2.__version__ if hasattr(cv2, '__version__') else 'unknown'
        },
        'models': {
            'insightface_available': INSIGHTFACE_AVAILABLE,
            'model_name': MODEL_NAME,
            'model_description': MODEL_DESCRIPTION,
            'mediapipe_available': MEDIAPIPE_AVAILABLE
        },
        'config': {
            'threshold': FACE_MATCH_THRESHOLD,
            'second_best_margin': SECOND_BEST_MARGIN,
            'min_samples_for_training': MIN_SAMPLES_FOR_TRAINING,
            'embedding_dimension': 512
        },
        'database': {
            'total_people': len(embeddings_db['names']),
            'is_trained': embeddings_db.get('is_trained', False),
            'total_samples': sum(len(e) for e in embeddings_db.get('raw_embeddings', []) if isinstance(e, list))
        }
    })

@app.route('/analyze_age_gender', methods=['POST'])
def analyze_age_gender():
    """Analyze age and gender using DeepFace"""
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'Image required'}), 400
        
        # Decode base64 image
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame_bgr is None:
                raise ValueError("Failed to decode image")
                
        except Exception as e:
            print(f"Image decode error: {e}")
            return jsonify({'error': 'Invalid image data', 'face_detected': False}), 400
        
        # Use DeepFace for gender detection only
        try:
            from deepface import DeepFace
            
            # Analyze gender with multiple backends for better detection
            result = None
            face_region = None
            
            # Try opencv first (faster)
            try:
                result = DeepFace.analyze(
                    img_path=frame_bgr,
                    actions=['gender'],
                    enforce_detection=True,
                    detector_backend='opencv',
                    silent=True
                )
                if isinstance(result, list):
                    result = result[0]
                face_region = result.get('region', None)
                print(f"[DEBUG] OpenCV detection successful, region: {face_region}")
            except:
                print(f"[DEBUG] OpenCV failed, trying retinaface...")
                # Try retinaface (more accurate)
                try:
                    result = DeepFace.analyze(
                        img_path=frame_bgr,
                        actions=['gender'],
                        enforce_detection=True,
                        detector_backend='retinaface',
                        silent=True
                    )
                    if isinstance(result, list):
                        result = result[0]
                    face_region = result.get('region', None)
                    print(f"[DEBUG] RetinaFace detection successful")
                except:
                    print(f"[DEBUG] RetinaFace failed, trying ssd...")
                    # Try SSD as last resort
                    result = DeepFace.analyze(
                        img_path=frame_bgr,
                        actions=['gender'],
                        enforce_detection=True,
                        detector_backend='ssd',
                        silent=True
                    )
                    if isinstance(result, list):
                        result = result[0]
                    face_region = result.get('region', None)
                    print(f"[DEBUG] SSD detection successful")
            
            gender = result['dominant_gender']
            gender_confidence = result['gender'][gender]
            
            # Get bounding box coordinates
            bbox = None
            if face_region:
                bbox = {
                    'x': int(face_region['x']),
                    'y': int(face_region['y']),
                    'w': int(face_region['w']),
                    'h': int(face_region['h'])
                }
            
            response_data = {
                'face_detected': True,
                'gender': gender.capitalize(),
                'gender_confidence': float(gender_confidence),
                'raw_gender': {k: float(v) for k, v in result['gender'].items()},
                'bbox': bbox
            }
            
            print(f"[GENDER] Detected: {gender.capitalize()} ({gender_confidence:.1f}%) BBox: {bbox}")
            
            return jsonify(response_data)
            
        except Exception as e:
            print(f"DeepFace gender error: {e}")
            return jsonify({'error': str(e), 'face_detected': False}), 500
            
    except Exception as e:
        print(f"Gender analysis error: {str(e)}")
        return jsonify({'error': str(e), 'face_detected': False}), 500

if __name__ == '__main__':
    print("\n🌐 Server starting at: http://127.0.0.1:5000")
    print("   Open this URL in your browser\n")
    app.run(debug=True, host='127.0.0.1', port=5000)