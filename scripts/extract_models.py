import zipfile
import os
import shutil

print("Extracting buffalo_l_new.zip...")

# Extract
with zipfile.ZipFile('models/buffalo_l_new.zip', 'r') as zip_ref:
    zip_ref.extractall('models')
    print("✓ Extracted")

# List extracted files
buffalo_dir = 'models/buffalo_l'
if os.path.exists(buffalo_dir):
    files = os.listdir(buffalo_dir)
    print(f"Found files: {files}")
    
    # Copy det_10g.onnx to scrfd_10g_bnkps.onnx
    if 'det_10g.onnx' in files:
        shutil.copy(f'{buffalo_dir}/det_10g.onnx', 'models/scrfd_10g_bnkps.onnx')
        print("✓ Copied SCRFD detector: scrfd_10g_bnkps.onnx")
    
    # Copy 2d106det.onnx to coordinate_reg_mean.onnx
    if '2d106det.onnx' in files:
        shutil.copy(f'{buffalo_dir}/2d106det.onnx', 'models/coordinate_reg_mean.onnx')
        print("✓ Copied 5-point landmark: coordinate_reg_mean.onnx")
    
    # Copy w600k_r50.onnx for ArcFace
    if 'w600k_r50.onnx' in files:
        shutil.copy(f'{buffalo_dir}/w600k_r50.onnx', 'models/w600k_r50.onnx')
        print("✓ Copied ArcFace model: w600k_r50.onnx")

print("\nFinal models directory:")
for f in os.listdir('models'):
    if f.endswith('.onnx'):
        size = os.path.getsize(f'models/{f}') / (1024 * 1024)
        print(f"  {f} ({size:.1f} MB)")
