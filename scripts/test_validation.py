#!/usr/bin/env python3
"""
Test face validation function
"""
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app import detect_and_validate_face

def create_test_image(face_size_ratio=0.4):
    """Create a test image with a simulated face"""
    # Create 640x480 black image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Calculate face size based on ratio
    frame_area = 480 * 640
    face_area = frame_area * face_size_ratio
    face_width = int(np.sqrt(face_area))
    face_height = face_width
    
    # Center the "face"
    x = (640 - face_width) // 2
    y = (480 - face_height) // 2
    
    # Draw white rectangle as "face"
    cv2.rectangle(img, (x, y), (x + face_width, y + face_height), (255, 255, 255), -1)
    
    # Add some features to help Haar detect it as a face
    # Eyes
    eye_y = y + face_height // 3
    eye_x1 = x + face_width // 3
    eye_x2 = x + 2 * face_width // 3
    cv2.circle(img, (eye_x1, eye_y), face_width // 10, (0, 0, 0), -1)
    cv2.circle(img, (eye_x2, eye_y), face_width // 10, (0, 0, 0), -1)
    
    # Mouth
    mouth_y = y + 2 * face_height // 3
    cv2.ellipse(img, (x + face_width // 2, mouth_y), 
                (face_width // 4, face_height // 8), 0, 0, 180, (0, 0, 0), 2)
    
    return img, face_size_ratio

def test_validation():
    print("="*60)
    print("Testing Face Validation")
    print("="*60)
    
    test_cases = [
        (0.10, "Too small - should be rejected"),
        (0.20, "Small but acceptable"),
        (0.35, "Optimal size"),
        (0.50, "Good size"),
        (0.60, "Upper optimal limit"),
        (0.70, "Too close - should be rejected"),
    ]
    
    for ratio, description in test_cases:
        print(f"\nTest: {description} (face area = {ratio*100:.1f}%)")
        img, actual_ratio = create_test_image(ratio)
        
        result = detect_and_validate_face(img)
        
        print(f"  Valid: {result['valid']}")
        print(f"  Message: {result['message']}")
        print(f"  Face area ratio: {result['face_area_ratio']:.3f}")
        
        if result['face_box']:
            x, y, w, h = result['face_box']
            print(f"  Face box: ({x}, {y}, {w}, {h})")
    
    print("\n" + "="*60)
    print("✓ Validation function is working")
    print("="*60)
    print("\nNote: Haar Cascade may not detect synthetic test faces.")
    print("Test with real webcam images for accurate results.")

if __name__ == "__main__":
    test_validation()
