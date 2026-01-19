"""
Download AdaFace Model using Python
More reliable than PowerShell for large files
"""

import urllib.request
import os
from pathlib import Path

def download_with_progress(url, destination):
    """Download file with progress bar"""
    print(f"📥 Downloading AdaFace model...")
    print(f"   URL: {url}")
    print(f"   Destination: {destination}")
    print(f"   Size: ~92 MB (this may take 2-5 minutes)\n")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded * 100 / total_size, 100)
            bar_length = 50
            filled = int(bar_length * percent / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f'\r   [{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)', end='', flush=True)
    
    try:
        urllib.request.urlretrieve(url, destination, progress_hook)
        print("\n\n✅ Download completed successfully!")
        
        # Verify file size
        file_size = os.path.getsize(destination) / (1024 * 1024)
        print(f"✓ File size: {file_size:.1f} MB")
        
        if file_size > 80:  # Should be around 92 MB
            print("✓ File appears valid")
            print("\n🚀 Next steps:")
            print("1. Restart your FacePass server (Ctrl+C then run again)")
            print("2. The system will automatically use AdaFace")
            print("3. Recapture training photos with multiple angles")
            print("4. Retrain the model")
            print("5. Test recognition - should be MUCH better!")
            return True
        else:
            print("⚠️ File size seems small - download may be incomplete")
            return False
            
    except Exception as e:
        print(f"\n\n❌ Download failed: {e}")
        print("\n💡 Alternative: Download manually from:")
        print("   https://github.com/mk-minchul/AdaFace/releases/download/v1.0/adaface_ir50_webface4m.onnx")
        print(f"\n   Then save it to: {destination}")
        return False

if __name__ == "__main__":
    print("="*70)
    print("🚀 AdaFace Model Downloader (Python)")
    print("="*70)
    
    # Get paths
    script_dir = Path(__file__).parent
    models_dir = script_dir.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    destination = models_dir / 'adaface_ir50_webface4m.onnx'
    
    # Check if already exists
    if destination.exists():
        file_size = destination.stat().st_size / (1024 * 1024)
        print(f"\n✓ AdaFace model already exists!")
        print(f"  Location: {destination}")
        print(f"  Size: {file_size:.1f} MB")
        
        overwrite = input("\n❓ Download again? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("\n✅ Using existing model. Restart server to activate!")
            exit(0)
    
    # Download
    url = "https://github.com/mk-minchul/AdaFace/releases/download/v1.0/adaface_ir50_webface4m.onnx"
    
    print("\n⏳ Starting download (this may take several minutes)...")
    print("   Please wait and don't interrupt...\n")
    
    success = download_with_progress(url, str(destination))
    
    if not success:
        print("\n" + "="*70)
        print("❌ DOWNLOAD FAILED")
        print("="*70)
        print("\nDon't worry! The improved settings are already active:")
        print("  • Threshold: 0.55 (more lenient)")
        print("  • Better quality checks")
        print("  • Smarter training")
        print("\nTry recognition again - it should work better now!")
        print("\nYou can download AdaFace manually later if needed.")
    
    print("\n" + "="*70)
