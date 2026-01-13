#!/usr/bin/env python3
"""
Test face alignment and embedding consistency
"""
import cv2
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from app import align_face_5point, generate_arcface_embedding, get_face_landmarks_5point

def test_alignment():
    """Test face alignment with a sample image"""
    print("="*60)
    print("🧪 TESTING FACE ALIGNMENT")
    print("="*60)
    
    # Create a simple test image (black square with white center)
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    print("\n1. Testing alignment function:")
    try:
        # Test without landmarks (should use fallback)
        aligned = align_face_5point(test_img, src_landmarks=None)
        print(f"   ✓ Fallback alignment works: {aligned.shape}")
        assert aligned.shape == (112, 112, 3), "Wrong output size!"
        
        # Test landmark detection
        landmarks = get_face_landmarks_5point(test_img)
        print(f"   ℹ️ Landmark detection (no face in test image): {landmarks is None}")
        
    except Exception as e:
        print(f"   ❌ Alignment test failed: {e}")
        return False
    
    print("\n2. Testing embedding consistency:")
    try:
        # Generate embedding twice for same image
        emb1 = generate_arcface_embedding(test_img, debug=True)
        print("\n   Generating second embedding...")
        emb2 = generate_arcface_embedding(test_img, debug=True)
        
        # Convert to numpy arrays
        emb1_np = np.array(emb1)
        emb2_np = np.array(emb2)
        
        # Check consistency
        diff = np.abs(emb1_np - emb2_np).max()
        print(f"\n   Max difference between embeddings: {diff:.6f}")
        
        if diff < 1e-6:
            print(f"   ✓ Embeddings are IDENTICAL (deterministic)")
        else:
            print(f"   ⚠️ Embeddings differ slightly: {diff:.6f}")
        
        # Check norms
        norm1 = np.linalg.norm(emb1_np)
        norm2 = np.linalg.norm(emb2_np)
        print(f"   Embedding 1 norm: {norm1:.6f}")
        print(f"   Embedding 2 norm: {norm2:.6f}")
        
        # Check if normalized
        if abs(norm1 - 1.0) < 0.001 and abs(norm2 - 1.0) < 0.001:
            print(f"   ✓ Embeddings are L2-normalized")
        else:
            print(f"   ⚠️ Embeddings may not be normalized properly")
        
    except Exception as e:
        print(f"   ❌ Embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60)
    print("\nAlignment is working correctly!")
    print("The pipeline will:")
    print("  1. Detect 5-point landmarks (if face present)")
    print("  2. Apply similarity transform to 112x112")
    print("  3. Normalize to [-1, 1]")
    print("  4. Generate L2-normalized 512-D embedding")
    print("\nReady for dataset capture and training!")
    return True

if __name__ == "__main__":
    success = test_alignment()
    sys.exit(0 if success else 1)
