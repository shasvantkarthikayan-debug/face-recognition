"""
Face Recognition System - Cross-Platform Setup Script
Automates environment setup, package installation, and model downloads
"""

import os
import sys
import subprocess
import urllib.request
import shutil
from pathlib import Path

# Color codes for terminal output
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{text}{Colors.RESET}")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_info(text):
    print(f"{Colors.CYAN}ℹ {text}{Colors.RESET}")

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    print_header("[1/6] Checking Python version...")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python {version_str} detected")
        return True
    else:
        print_error(f"Python {version_str} detected. Python 3.8+ required!")
        return False

def create_directories():
    """Create necessary project directories"""
    print_header("[2/6] Creating project directories...")
    
    directories = [
        "models",
        "data",
        "static/css",
        "static/js",
        "templates",
        "known_faces"
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print_success(f"Created: {directory}")
        else:
            print_info(f"Exists: {directory}")

def install_packages():
    """Install required Python packages"""
    print_header("[3/6] Installing Python packages...")
    print_info("This may take a few minutes...")
    
    try:
        # Upgrade pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print_success("All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install packages: {e}")
        return False

def download_model(url, filepath):
    """Download a file with progress indication"""
    try:
        print(f"  Downloading {Path(filepath).name}...")
        
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r  Progress: {percent}%")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, filepath, reporthook)
        print()  # New line after progress
        
        # Check file size
        size_mb = Path(filepath).stat().st_size / (1024 * 1024)
        print_success(f"Downloaded: {Path(filepath).name} ({size_mb:.2f} MB)")
        return True
    except Exception as e:
        print_error(f"Download failed: {e}")
        return False

def download_models():
    """Download required ONNX models"""
    print_header("[4/6] Downloading ONNX models...")
    
    # Check if models already exist
    models_path = Path("models")
    det_model = models_path / "det_10g.onnx"
    rec_model = models_path / "w600k_r50.onnx"
    
    if det_model.exists() and rec_model.exists():
        print_info("Models already exist")
        response = input("Re-download models? (y/n): ").strip().lower()
        if response != 'y':
            print_info("Skipping model download")
            return True
    
    print_info("Downloading InsightFace models from HuggingFace...")
    print_warning("This may take several minutes depending on your connection")
    
    # Model URLs (these are example URLs - you'll need to update with actual URLs)
    models = {
        "det_10g.onnx": "https://huggingface.co/MonsterMMORPG/tools/resolve/main/scrfd_10g_bnkps.onnx",
        "w600k_r50.onnx": "https://huggingface.co/ezioruan/yunet_final/resolve/main/arcface_w600k_r50.onnx"
    }
    
    success = True
    for model_name, url in models.items():
        filepath = models_path / model_name
        if not download_model(url, str(filepath)):
            success = False
            print_warning(f"Failed to download {model_name}")
            print_info("You may need to download manually from:")
            print_info("https://github.com/deepinsight/insightface/tree/master/model_zoo")
    
    return success

def validate_setup():
    """Validate that all required files are present"""
    print_header("[5/6] Validating setup...")
    
    all_valid = True
    
    # Check models
    required_models = [
        "models/det_10g.onnx",
        "models/w600k_r50.onnx"
    ]
    
    for model_path in required_models:
        path = Path(model_path)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print_success(f"Found: {model_path} ({size_mb:.2f} MB)")
        else:
            print_error(f"Missing: {model_path}")
            all_valid = False
    
    # Check key files
    required_files = [
        "app.py",
        "requirements.txt",
        "templates/index.html"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print_success(f"Found: {file_path}")
        else:
            print_error(f"Missing: {file_path}")
            all_valid = False
    
    return all_valid

def create_run_script():
    """Create convenient run scripts"""
    print_header("[6/6] Creating run scripts...")
    
    # Windows batch file
    if sys.platform == "win32":
        run_script = Path("run.bat")
        with open(run_script, "w") as f:
            f.write("@echo off\n")
            f.write("echo Starting Face Recognition System...\n")
            f.write("python app.py\n")
            f.write("pause\n")
        print_success("Created: run.bat")
    
    # Unix shell script
    else:
        run_script = Path("run.sh")
        with open(run_script, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("echo Starting Face Recognition System...\n")
            f.write("python app.py\n")
        run_script.chmod(0o755)
        print_success("Created: run.sh")

def main():
    """Main setup routine"""
    print(f"\n{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}Face Recognition System - Setup{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}")
    
    # Change to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Create directories
    create_directories()
    
    # Step 3: Install packages
    if not install_packages():
        print_warning("Package installation failed. Please install manually.")
    
    # Step 4: Download models
    response = input("\nDownload ONNX models? (y/n) [recommended for first setup]: ").strip().lower()
    if response == 'y':
        download_models()
    else:
        print_info("Skipping model download")
        print_info("Models required: det_10g.onnx, w600k_r50.onnx")
        print_info("See: docs/MODEL_DOWNLOAD_INSTRUCTIONS.md")
    
    # Step 5: Validate setup
    all_valid = validate_setup()
    
    # Step 6: Create run scripts
    create_run_script()
    
    # Final message
    print(f"\n{Colors.CYAN}{'='*60}{Colors.RESET}")
    
    if all_valid:
        print_success("Setup completed successfully!")
        print(f"\n{Colors.CYAN}To start the application:{Colors.RESET}")
        print(f"  {Colors.BOLD}python app.py{Colors.RESET}")
        print(f"\n{Colors.CYAN}Then open:{Colors.RESET}")
        print(f"  {Colors.BOLD}http://127.0.0.1:5000{Colors.RESET}")
    else:
        print_warning("Setup completed with warnings")
        print_warning("Please resolve missing files/models before running")
    
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
