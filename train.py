import cv2
import numpy as np
import mediapipe as mp
import os

# -----------------------------
# Teacher-friendly comments 😎
# -----------------------------
# This script reads all face images for each person,
# computes embeddings using Mediapipe Face Mesh,
# then averages them per person so recognition is more reliable.

mp_face_mesh = mp.solutions.face_mesh

# Paths
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.npy"
NAMES_FILE = "names.npy"
CATEGORIES_FILE = "categories.npy"

def get_face_embedding(image):
    """Return a 468-dim face embedding using Mediapipe Face Mesh."""
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        # Flatten x, y, z coordinates
        embedding = np.array([[p.x, p.y, p.z] for p in landmarks]).flatten()
        return embedding

def main():
    averaged_encodings = []
    names_list = []
    categories_list = []

    for category in os.listdir(KNOWN_FACES_DIR):
        cat_path = os.path.join(KNOWN_FACES_DIR, category)
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
                    print(f"❌ Cannot read {img_path}")
                    continue

                embedding = get_face_embedding(img)
                if embedding is None:
                    print(f"❌ No face detected in {img_path}")
                    continue

                person_embeddings.append(embedding)
                print(f"✅ Processed {person}/{img_file}")

            if person_embeddings:
                # Average all embeddings for this person
                avg_embedding = np.mean(person_embeddings, axis=0)
                averaged_encodings.append(avg_embedding)
                names_list.append(person)
                categories_list.append(category)

    # Save averaged embeddings
    np.save(ENCODINGS_FILE, averaged_encodings)
    np.save(NAMES_FILE, names_list)
    np.save(CATEGORIES_FILE, categories_list)
    print("✅ Training done! Averaged embeddings saved. Recognition should be more stable now.")

if __name__ == "__main__":
    main()
