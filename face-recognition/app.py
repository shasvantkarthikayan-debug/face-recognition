from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import numpy as np
import mediapipe as mp
import os
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'facepass-secret-key'

# Global variables
camera = None
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
known_encodings = []
known_names = []
known_categories = []
is_recognizing = False
capture_count = 0

# Load trained data
def load_training_data():
    global known_encodings, known_names, known_categories
    try:
        known_encodings = np.load("encodings.npy", allow_pickle=True)
        known_names = np.load("names.npy", allow_pickle=True)
        known_categories = np.load("categories.npy", allow_pickle=True)
    except:
        known_encodings = []
        known_names = []
        known_categories = []

load_training_data()

def get_face_embedding(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark
    embedding = np.array([[p.x, p.y, p.z] for p in landmarks]).flatten()
    return embedding

def l2_distance(a, b):
    return np.linalg.norm(a - b)

def generate_frames():
    global camera, is_recognizing
    camera = cv2.VideoCapture(0)
    
    correct = 0
    total = 0
    
    while camera and is_recognizing:
        success, frame = camera.read()
        if not success:
            break
        
        if len(known_encodings) > 0:
            embedding = get_face_embedding(frame)
            
            if embedding is not None:
                distances = [l2_distance(embedding, enc) for enc in known_encodings]
                min_idx = np.argmin(distances)
                
                h, w, _ = frame.shape
                xs = [p[0] for p in embedding.reshape(-1, 3)]
                ys = [p[1] for p in embedding.reshape(-1, 3)]
                left, right = int(min(xs) * w), int(max(xs) * w)
                top, bottom = int(min(ys) * h), int(max(ys) * h)
                
                if distances[min_idx] < 5:
                    name = known_names[min_idx]
                    category = known_categories[min_idx]
                    correct += 1
                    box_color = (0, 255, 0)
                else:
                    name = "Unknown"
                    category = ""
                    box_color = (0, 0, 255)
                
                total += 1
                accuracy = (correct / total) * 100
                
                cv2.rectangle(frame, (left, top), (right, bottom), box_color, 3)
                cv2.putText(frame, f"{name} ({category})", (left, top-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                cv2.putText(frame, f"Accuracy: {accuracy:.1f}%", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    if camera:
        camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture')
def capture_page():
    return render_template('capture.html')

@app.route('/train')
def train_page():
    return render_template('train.html')

@app.route('/recognize')
def recognize_page():
    return render_template('recognize.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recognition', methods=['POST'])
def start_recognition():
    global is_recognizing
    is_recognizing = True
    return jsonify({'status': 'started'})

@app.route('/stop_recognition', methods=['POST'])
def stop_recognition():
    global is_recognizing, camera
    is_recognizing = False
    if camera:
        camera.release()
        camera = None
    return jsonify({'status': 'stopped'})

@app.route('/capture_photo', methods=['POST'])
def capture_photo():
    global capture_count
    data = request.json
    name = data.get('name')
    category = data.get('category')
    image_data = data.get('image')
    
    if not name or not category:
        return jsonify({'error': 'Name and category required'}), 400
    
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    save_dir = f"known_faces/{category}/{name}"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/{name}_{capture_count}.jpg"
    cv2.imwrite(filename, img)
    capture_count += 1
    
    return jsonify({'status': 'success', 'count': capture_count})

@app.route('/train_model', methods=['POST'])
def train_model():
    averaged_encodings = []
    names_list = []
    categories_list = []
    
    base_dir = "known_faces"
    processed = 0
    
    if not os.path.exists(base_dir):
        return jsonify({'error': 'No training data found'}), 400
    
    for category in os.listdir(base_dir):
        cat_path = os.path.join(base_dir, category)
        if not os.path.isdir(cat_path):
            continue
        
        for person in os.listdir(cat_path):
            person_path = os.path.join(cat_path, person)
            if not os.path.isdir(person_path):
                continue
            
            person_embeddings = []
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                with mp.solutions.face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
                    embedding = get_face_embedding(img)
                    if embedding is not None:
                        person_embeddings.append(embedding)
                        processed += 1
            
            if person_embeddings:
                avg_embedding = np.mean(person_embeddings, axis=0)
                averaged_encodings.append(avg_embedding)
                names_list.append(person)
                categories_list.append(category)
    
    np.save("encodings.npy", averaged_encodings)
    np.save("names.npy", names_list)
    np.save("categories.npy", categories_list)
    
    load_training_data()
    
    return jsonify({
        'status': 'success',
        'processed': processed,
        'people': len(names_list)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)