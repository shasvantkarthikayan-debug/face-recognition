"""
Download AdaFace Model - Better Face Recognition
AdaFace handles pose, lighting, and expression variations much better than ArcFace
"""

import os
from pathlib import Path

print("="*70)
print("🚀 AdaFace Model Downloader")
print("="*70)
print("\nAdaFace is superior to ArcFace for handling:")
print("  ✓ Pose variations (side profiles, tilted heads)")
print("  ✓ Lighting changes (shadows, bright light)")
print("  ✓ Expression variations (smiling, serious)")
print("  ✓ Partial occlusions (glasses, masks)")
print("  ✓ Age variations")
print("\n" + "="*70)

# Get models directory
script_dir = Path(__file__).parent
models_dir = script_dir.parent / 'models'
models_dir.mkdir(exist_ok=True)

print(f"\n📁 Models directory: {models_dir}")
print("\n" + "="*70)
print("📥 DOWNLOAD INSTRUCTIONS")
print("="*70)

print("\n🎯 OPTION 1: AdaFace IR50 WebFace4M (RECOMMENDED)")
print("-" * 70)
print("Best balance of accuracy and speed")
print("\n1. Download from:")
print("   https://github.com/mk-minchul/AdaFace/releases/download/v1.0/adaface_ir50_webface4m.onnx")
print("\n2. Or use this command:")
print("   curl -L -o models/adaface_ir50_webface4m.onnx \\")
print("     https://github.com/mk-minchul/AdaFace/releases/download/v1.0/adaface_ir50_webface4m.onnx")
print("\n3. File size: ~92 MB")

print("\n\n🎯 OPTION 2: Use InsightFace Buffalo-L")
print("-" * 70)
print("Already installed alternative")
print("\n1. Download buffalo_l.zip from:")
print("   https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip")
print("\n2. Extract and copy these files to models folder:")
print("   - det_10g.onnx")
print("   - w600k_r50.onnx (or glintr100.onnx for better quality)")

print("\n\n🎯 OPTION 3: Quick Fix - Adjust Current Settings")
print("-" * 70)
print("I've already lowered the thresholds for better recognition:")
print("  • Threshold: 0.65 → 0.55 (more lenient)")
print("  • Margin: 0.10 → 0.08 (less strict)")
print("  • Min samples: 5 → 3 (easier training)")
print("  • Quality checks: Relaxed for varied conditions")
print("\nTry recognition again - it should work better now!")

print("\n" + "="*70)
print("💡 QUICK TIP")
print("="*70)
print("\nIf you still have misrecognition issues:")
print("1. Capture 8-10 photos per person with varied:")
print("   - Angles (front, left, right, up, down)")
print("   - Lighting (normal, bright, dim)")
print("   - Expressions (neutral, smiling)")
print("2. Retrain the model")
print("3. Check system info to see active model")

print("\n" + "="*70)
print("📊 Current Model Status")
print("="*70)

# Check what models are currently available
current_models = {
    'det_10g.onnx': 'Detection',
    'w600k_r50.onnx': 'Recognition (ArcFace R50)',
    'glintr100.onnx': 'Recognition (ArcFace R100 - Better)',
    'adaface_ir50_webface4m.onnx': 'Recognition (AdaFace - Best)',
}

for model_file, description in current_models.items():
    model_path = models_dir / model_file
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"   ✓ {model_file:<30} ({size_mb:.1f} MB) - {description}")
    else:
        print(f"   ✗ {model_file:<30} - {description} [NOT FOUND]")

print("\n" + "="*70)
print("✅ NEXT STEPS")
print("="*70)
print("\n1. If you downloaded AdaFace, restart the server")
print("2. Visit http://127.0.0.1:5000/system_info_page to verify")
print("3. Recapture training photos (more angles = better results)")
print("4. Retrain the model")
print("5. Test recognition - should be much better!")

print("\n" + "="*70)
