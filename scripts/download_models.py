import os
import urllib.request
import zipfile

def download_file(url, destination):
    """Download a file with progress"""
    print(f"Downloading {os.path.basename(destination)}...")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"✓ Downloaded: {destination}")
        return True
    except Exception as e:
        print(f"✗ Error downloading: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    print(f"Extracting {os.path.basename(zip_path)}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ Extracted to: {extract_to}")
        return True
    except Exception as e:
        print(f"✗ Error extracting: {e}")
        return False

# Create models directory
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

print("="*60)
print("InsightFace Model Downloader")
print("="*60)

# Download buffalo_l model pack (includes SCRFD detector and ArcFace models)
buffalo_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
buffalo_zip = os.path.join(models_dir, "buffalo_l.zip")

if download_file(buffalo_url, buffalo_zip):
    # Extract the zip
    buffalo_dir = os.path.join(models_dir, "buffalo_l")
    if extract_zip(buffalo_zip, models_dir):
        # Rename det_10g.onnx to scrfd_10g_bnkps.onnx
        det_model = os.path.join(buffalo_dir, "det_10g.onnx")
        scrfd_model = os.path.join(models_dir, "scrfd_10g_bnkps.onnx")
        
        if os.path.exists(det_model):
            os.rename(det_model, scrfd_model)
            print(f"✓ Renamed to: scrfd_10g_bnkps.onnx")
        
        # Find and copy the landmark model (2d106det.onnx for 5-point landmarks)
        landmark_src = os.path.join(buffalo_dir, "2d106det.onnx")
        landmark_dst = os.path.join(models_dir, "coordinate_reg_mean.onnx")
        
        if os.path.exists(landmark_src):
            import shutil
            shutil.copy(landmark_src, landmark_dst)
            print(f"✓ Copied landmark model: coordinate_reg_mean.onnx")
        
        # Check if ArcFace model exists
        arcface_model = os.path.join(buffalo_dir, "w600k_r50.onnx")
        if os.path.exists(arcface_model):
            arcface_dst = os.path.join(models_dir, "w600k_r50.onnx")
            import shutil
            shutil.copy(arcface_model, arcface_dst)
            print(f"✓ Copied ArcFace model: w600k_r50.onnx")
        
        # Clean up zip file
        os.remove(buffalo_zip)
        print("✓ Cleaned up zip file")

print("="*60)
print("Model Download Complete!")
print("="*60)
print(f"Models available in: {os.path.abspath(models_dir)}")
print("- scrfd_10g_bnkps.onnx (SCRFD detector)")
print("- coordinate_reg_mean.onnx (5-point landmark)")
print("- w600k_r50.onnx (ArcFace recognition)")
