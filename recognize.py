import cv2
import numpy as np
import mediapipe as mp
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

        # If closest match is close enough, label it
        if distances[min_idx] < 5:  # tweak threshold as needed
            name = known_names[min_idx]
            category = known_categories[min_idx]
            correct += 1  # counts as correct match

        total += 1
        accuracy = (correct / total) * 100  # rolling accuracy

        # Draw bounding box using Mediapipe landmarks
        h, w, _ = frame.shape
        xs = [p[0] for p in embedding.reshape(-1, 3)]
        ys = [p[1] for p in embedding.reshape(-1, 3)]
        left, right = int(min(xs) * w), int(max(xs) * w)
        top, bottom = int(min(ys) * h), int(max(ys) * h)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({category}) Accuracy: {accuracy:.2f}%", 
                    (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# --- Run ---
if __name__ == "__main__":
    recognize_faces()
