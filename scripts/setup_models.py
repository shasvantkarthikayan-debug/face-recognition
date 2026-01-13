import zipfile
import shutil
import os

try:
    print("Starting extraction...")
    
    # Extract buffalo_l_new.zip
    with zipfile.ZipFile('models/buffalo_l_new.zip', 'r') as zip_ref:
        zip_ref.extractall('models')
    print("✓ Extracted buffalo_l_new.zip")
    
    # Check what was extracted
    buffalo_dir = 'models/buffalo_l'
    if os.path.exists(buffalo_dir):
        files = os.listdir(buffalo_dir)
        print(f"\nExtracted files in buffalo_l/:")
        for f in files:
            size = os.path.getsize(os.path.join(buffalo_dir, f)) / (1024 * 1024)
            print(f"  - {f} ({size:.1f} MB)")
        
        # Copy SCRFD detector
        if 'det_10g.onnx' in files:
            src = os.path.join(buffalo_dir, 'det_10g.onnx')
            dst = 'models/scrfd_10g_bnkps.onnx'
            shutil.copy(src, dst)
            print(f"\n✓ Copied SCRFD detector -> scrfd_10g_bnkps.onnx")
        
        # Copy 5-point landmark model
        if '2d106det.onnx' in files:
            src = os.path.join(buffalo_dir, '2d106det.onnx')
            dst = 'models/coordinate_reg_mean.onnx'
            shutil.copy(src, dst)
            print(f"✓ Copied landmark model -> coordinate_reg_mean.onnx")
        
        # Copy ArcFace model (if different from existing)
        if 'w600k_r50.onnx' in files:
            src = os.path.join(buffalo_dir, 'w600k_r50.onnx')
            dst = 'models/w600k_r50.onnx'
            if not os.path.exists('models/arcface_w600k_r50.onnx'):
                shutil.copy(src, dst)
                print(f"✓ Copied ArcFace model -> w600k_r50.onnx")
        
        print("\n" + "="*60)
        print("MODEL SETUP COMPLETE!")
        print("="*60)
        
        # List final models
        print("\nFinal models in models/ directory:")
        for f in sorted(os.listdir('models')):
            if f.endswith('.onnx'):
                size = os.path.getsize(f'models/{f}') / (1024 * 1024)
                print(f"  ✓ {f} ({size:.1f} MB)")
        
        print("\nYou can now start the Flask server!")
        
    else:
        print("ERROR: buffalo_l directory not found after extraction")
        print("Please manually extract buffalo_l_new.zip")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
