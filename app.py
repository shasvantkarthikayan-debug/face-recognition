from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import numpy as np
import os
import base64
import json
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'facepass-secret-key'

# Global variables
capture_count = 0
embeddings_db = {
    'embeddings': [],
    'names': [],
    'categories': []
}

# Load trained data from JSON file
def load_training_data():
    global embeddings_db
    try:
        if os.path.exists("face_data.json"):
            with open("face_data.json", "r") as f:
                embeddings_db = json.load(f)
    except:
        embeddings_db = {
            'embeddings': [],
            'names': [],
            'categories': []
        }

def save_training_data():
    with open("face_data.json", "w") as f:
        json.dump(embeddings_db, f)

load_training_data()

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

@app.route('/manage')
def manage_page():
    return render_template('manage.html')

@app.route('/get_trained_data', methods=['GET'])
def get_trained_data():
    """Return trained face data to client for recognition"""
    return jsonify(embeddings_db)

@app.route('/capture_photo', methods=['POST'])
def capture_photo():
    global capture_count
    try:
        data = request.json
        name = data.get('name')
        category = data.get('category')
        image_data = data.get('image')
        embedding = data.get('embedding')  # Face embedding from client
        
        if not name or not category:
            return jsonify({'error': 'Name and category required'}), 400
        
        if not image_data:
            return jsonify({'error': 'Image data required'}), 400
        
        # If no embedding provided, use a placeholder
        if not embedding:
            embedding = [0.0] * 128
        
        # Decode base64 image
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return jsonify({'error': 'Failed to decode image'}), 400
        except Exception as e:
            return jsonify({'error': f'Image decode error: {str(e)}'}), 400
        
        # Save image
        save_dir = f"known_faces/{category}/{name}"
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/{name}_{capture_count}.jpg"
        cv2.imwrite(filename, img)
        
        # Store embedding in database
        # Check if person already exists
        found = False
        for i, (n, c) in enumerate(zip(embeddings_db['names'], embeddings_db['categories'])):
            if n == name and c == category:
                embeddings_db['embeddings'][i].append(embedding)
                found = True
                break
        
        if not found:
            embeddings_db['embeddings'].append([embedding])
            embeddings_db['names'].append(name)
            embeddings_db['categories'].append(category)
        
        save_training_data()
        capture_count += 1
        
        return jsonify({'status': 'success', 'count': capture_count})
    except Exception as e:
        print(f"Error in capture_photo: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    """Training is now done on client-side, this just returns current stats"""
    total_embeddings = sum(len(emb_list) for emb_list in embeddings_db['embeddings'])
    num_people = len(embeddings_db['names'])
    
    if num_people == 0:
        return jsonify({'error': 'No training data found'}), 400
    
    return jsonify({
        'status': 'success',
        'processed': total_embeddings,
        'people': num_people
    })

@app.route('/statistics', methods=['GET'])
def get_statistics():
    """Get detailed statistics about trained faces"""
    stats = {
        'total_people': len(embeddings_db['names']),
        'total_photos': sum(len(emb_list) for emb_list in embeddings_db['embeddings']),
        'people_details': []
    }
    
    for i, (name, category, embeddings) in enumerate(zip(
        embeddings_db['names'], 
        embeddings_db['categories'], 
        embeddings_db['embeddings']
    )):
        stats['people_details'].append({
            'id': i,
            'name': name,
            'category': category,
            'photo_count': len(embeddings)
        })
    
    return jsonify(stats)

@app.route('/delete_person/<int:person_id>', methods=['DELETE'])
def delete_person(person_id):
    """Delete a person's data"""
    if person_id < 0 or person_id >= len(embeddings_db['names']):
        return jsonify({'error': 'Invalid person ID'}), 400
    
    name = embeddings_db['names'][person_id]
    category = embeddings_db['categories'][person_id]
    
    # Remove from database
    del embeddings_db['names'][person_id]
    del embeddings_db['categories'][person_id]
    del embeddings_db['embeddings'][person_id]
    
    save_training_data()
    
    return jsonify({'status': 'success', 'message': f'Deleted {name}'})

@app.route('/export_data', methods=['GET'])
def export_data():
    """Export face data as JSON file"""
    return jsonify(embeddings_db)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)