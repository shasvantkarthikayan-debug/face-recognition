import cv2
import numpy as np
import mediapipe as mp
import webbrowser
import os
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
# This script opens your webcam,
# detects faces with Mediapipe Face Mesh,
# computes embeddings like in training,
# compares with saved face data,
# and labels recognized people in real-time.
# Basically, it’s your face ID squad in action.

# --- Load trained face data ---
print("Loading trained face data...")
known_encodings = np.load("encodings.npy", allow_pickle=True)
known_names = np.load("names.npy", allow_pickle=True)
known_categories = np.load("categories.npy", allow_pickle=True)

# --- Mediapipe Face Mesh setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# --- Helper functions ---
def l2_distance(a, b):
    """Compute distance between two embeddings (small number = match)"""
    return np.linalg.norm(a - b)

def get_face_embedding(image):
    """Return a 468-point Mediapipe face mesh embedding (flattened)"""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark
    embedding = np.array([[p.x, p.y, p.z] for p in landmarks]).flatten()
    return embedding

def save_unknown_face(frame, left, top, right, bottom):
    """Save unknown face crop for manual search"""
    face_crop = frame[top:bottom, left:right]
    filename = f"unknown_face_{np.random.randint(1000)}.jpg"
    filepath = os.path.abspath(filename)
    cv2.imwrite(filepath, face_crop)
    return filepath

def open_reverse_search(image_path):
    """Open Google Images reverse search in browser"""
    # Convert to absolute path and URL encode
    abs_path = os.path.abspath(image_path)
    
    # Option 1: Open folder so user can drag-drop to Google Lens
    folder_path = os.path.dirname(abs_path)
    os.startfile(folder_path)
    
    # Option 2: Open Google Lens upload page
    webbrowser.open("https://lens.google.com/")
    
    print(f"📂 Opened folder: {folder_path}")
    print(f"📷 Image saved as: {os.path.basename(image_path)}")
    print("👉 Drag and drop the image to Google Lens in your browser!")

def search_face_selenium(image_path):
    """Use headless Selenium to search face and get first result"""
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        
        # Auto-install ChromeDriver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        driver.get("https://images.google.com/")
        
        # Upload image
        upload_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "input[type='file']"))
        )
        upload_btn.send_keys(image_path)
        
        # Wait for results
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.rg_i"))
        )
        
        # Click on first result
        first_result = driver.find_element(By.CSS_SELECTOR, "div.rg_i")
        first_result.click()
        
        # Wait for new tab
        WebDriverWait(driver, 10).until(
            EC.number_of_windows_to_be(2)
        )
        
        # Switch to new tab
        driver.switch_to.window(driver.window_handles[1])
        
        # Get URL of the result
        result_url = driver.current_url
        
        driver.quit()
        
        return result_url
    except Exception as e:
        print(f"❌ Error during search: {e}")
        return None

# --- Main recognition loop ---
def recognize_faces():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    correct = 0
    total = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera error!")
            break

        embedding = get_face_embedding(frame)
        if embedding is None:
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Compare with known encodings
        distances = [l2_distance(embedding, enc) for enc in known_encodings]
        min_idx = np.argmin(distances)
        name = "Unknown"
        category = ""

        # Draw bounding box using Mediapipe landmarks
        h, w, _ = frame.shape
        xs = [p[0] for p in embedding.reshape(-1, 3)]
        ys = [p[1] for p in embedding.reshape(-1, 3)]
        left, right = int(min(xs) * w), int(max(xs) * w)
        top, bottom = int(min(ys) * h), int(max(ys) * h)

        # If closest match is close enough, label it
        if distances[min_idx] < 5:  # tweak threshold as needed
            name = known_names[min_idx]
            category = known_categories[min_idx]
            correct += 1  # counts as correct match
            box_color = (0, 255, 0)  # Green for recognized
        else:
            box_color = (0, 0, 255)  # Red for unknown

        total += 1
        accuracy = (correct / total) * 100  # rolling accuracy

        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
        cv2.putText(frame, f"{name} ({category}) Accuracy: {accuracy:.2f}%", 
                    (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        cv2.imshow("Face Recognition", frame)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        
        # Press 's' to search unknown face
        if key == ord('s') and name == "Unknown":
            img_file = save_unknown_face(frame, left, top, right, bottom)
            print(f"✅ Saved unknown face: {img_file}")
            print("🔍 Opening reverse image search...")
            result_url = search_face_selenium(img_file)
            if result_url:
                print(f"🔗 First result URL: {result_url}")
            else:
                print("❌ Failed to get search results.")
        # Quit on 'q'
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# --- Run ---
if __name__ == "__main__":
    recognize_faces()
