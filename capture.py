import cv2
import os

# Where we keep everyone's faces
BASE_DIR = "known_faces"

# Make sure folders exist for parents and students
def ensure_dirs():
    os.makedirs(f"{BASE_DIR}/parents", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/students", exist_ok=True)

def capture_images():
    ensure_dirs()
    name = input("Type the person's name (no spaces): ").strip()
    category = input("Is this a parent or student? ").strip().lower()

    if category not in ("parents", "students"):
        print("Oops! You typed wrong. Must be 'parents' or 'students'")
        return

    save_dir = f"{BASE_DIR}/{category}/{name}"
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    print("Press 'c' to take a photo | Press 'q' to stop")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera broke 😭")
            break

        cv2.imshow("Capture Mode", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            path = f"{save_dir}/{name}_{count}.jpg"
            cv2.imwrite(path, frame)
            print("Saved photo:", path)
            count += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_images()
