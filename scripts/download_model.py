"""
Download ArcFace ONNX model for face recognition
"""
import os
import urllib.request
import ssl

# Create SSL context that doesn't verify certificates (for corporate proxies)
ssl._create_default_https_context = ssl._create_unverified_context

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "arcface_w600k_r50.onnx")

# Alternative model URLs (smaller models)
MODEL_URLS = [
    # InsightFace lightweight model (~6MB)
    ("https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip", "buffalo_l.zip"),
    
    # ONNX Model Zoo ArcFace (~100MB)
    ("https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx", "arcface_w600k_r50.onnx"),
]

def download_with_progress(url, output_path):
    """Download file with progress indicator"""
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
        print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
    
    urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
    print("\n✓ Download complete!")

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH)
        print(f"✓ Model already exists: {MODEL_PATH} ({file_size:,} bytes)")
        
        # Check if file is corrupted (too small)
        if file_size < 1000000:  # Less than 1MB is suspicious
            print("⚠️ File seems corrupted (too small), deleting...")
            os.remove(MODEL_PATH)
        else:
            print("Model file looks valid. If you want to re-download, delete it first.")
            return
    
    print("\n" + "="*60)
    print("📥 ArcFace Model Download")
    print("="*60)
    print("\nNote: This will download a large file (~100MB)")
    print("For faster testing, you can use the hash-based fallback embeddings.")
    print("\nAttempting download...")
    
    # Try ONNX model
    url = MODEL_URLS[1][0]
    try:
        download_with_progress(url, MODEL_PATH)
        print(f"\n✅ Model downloaded successfully to: {MODEL_PATH}")
        print(f"File size: {os.path.getsize(MODEL_PATH):,} bytes")
        return
    except KeyboardInterrupt:
        print("\n\n⚠️ Download interrupted by user")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
    
    print("\n" + "="*60)
    print("⚠️ Could not download model automatically")
    print("="*60)
    print("\nOptions:")
    print("1. Run this script again with better internet connection")
    print("2. Download manually from:")
    print("   https://github.com/onnx/models/tree/main/validated/vision/body_analysis/arcface")
    print(f"3. Save as: {MODEL_PATH}")
    print("4. Use the hash-based fallback (works but less accurate)")

if __name__ == "__main__":
    main()
