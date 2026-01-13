# Face Recognition System - Windows Setup Script
# This script automates the complete setup process

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Face Recognition System - Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] Found: $pythonVersion" -ForegroundColor Green
    
    # Check if Python version is 3.8+
    $versionNumber = [regex]::Match($pythonVersion, '\d+\.\d+').Value
    if ([version]$versionNumber -lt [version]"3.8") {
        Write-Host "[ERROR] Python 3.8+ required. Please upgrade Python." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "[ERROR] Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Create virtual environment (optional)
Write-Host "[2/6] Setting up virtual environment..." -ForegroundColor Yellow
$createVenv = Read-Host "Create virtual environment? (y/n) [recommended: y]"

if ($createVenv -eq "y" -or $createVenv -eq "Y") {
    if (Test-Path ".venv") {
        Write-Host "[INFO] Virtual environment already exists" -ForegroundColor Cyan
        $recreate = Read-Host "Recreate it? (y/n)"
        if ($recreate -eq "y" -or $recreate -eq "Y") {
            Remove-Item -Recurse -Force .venv
            python -m venv .venv
            Write-Host "[OK] Virtual environment recreated" -ForegroundColor Green
        }
    } else {
        python -m venv .venv
        Write-Host "[OK] Virtual environment created" -ForegroundColor Green
    }
    
    Write-Host "Activating virtual environment..." -ForegroundColor Cyan
    & .\.venv\Scripts\Activate.ps1
    Write-Host "[OK] Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "[SKIP] Skipping virtual environment" -ForegroundColor Yellow
}

Write-Host ""

# Install Python packages
Write-Host "[3/6] Installing Python packages..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Cyan

try {
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    Write-Host "[OK] All packages installed successfully" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Failed to install packages. Check requirements.txt" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Create necessary directories
Write-Host "[4/6] Creating project directories..." -ForegroundColor Yellow

$directories = @("models", "data", "static/css", "static/js", "templates", "known_faces")

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "[OK] Created: $dir" -ForegroundColor Green
    } else {
        Write-Host "[OK] Exists: $dir" -ForegroundColor Cyan
    }
}

Write-Host ""

# Download ONNX models
Write-Host "[5/6] Downloading ONNX models..." -ForegroundColor Yellow

$downloadModels = Read-Host "Download InsightFace ONNX models? (y/n) [required for first setup: y]"

if ($downloadModels -eq "y" -or $downloadModels -eq "Y") {
    Write-Host "Starting model download (this may take several minutes)..." -ForegroundColor Cyan
    
    try {
        python scripts/download_models.py
        Write-Host "[OK] Models downloaded successfully" -ForegroundColor Green
    } catch {
        Write-Host "[WARN] Model download failed. You may need to download manually." -ForegroundColor Yellow
        Write-Host "See: docs/MODEL_DOWNLOAD_INSTRUCTIONS.md" -ForegroundColor Cyan
    }
} else {
    Write-Host "[SKIP] Skipping model download" -ForegroundColor Yellow
    Write-Host "[INFO] Models required: det_10g.onnx, w600k_r50.onnx" -ForegroundColor Cyan
}

Write-Host ""

# Validation
Write-Host "[6/6] Validating setup..." -ForegroundColor Yellow

$allValid = $true

# Check models
$requiredModels = @("models/det_10g.onnx", "models/w600k_r50.onnx")
foreach ($model in $requiredModels) {
    if (Test-Path $model) {
        $size = (Get-Item $model).Length / 1MB
        $sizeRounded = [math]::Round($size, 2)
        Write-Host "[OK] Found: $model ($sizeRounded MB)" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Missing: $model" -ForegroundColor Red
        $allValid = $false
    }
}

# Check key files
$requiredFiles = @("app.py", "requirements.txt", "templates/index.html")
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "[OK] Found: $file" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Missing: $file" -ForegroundColor Red
        $allValid = $false
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan

if ($allValid) {
    Write-Host "[SUCCESS] Setup completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "To start the application:" -ForegroundColor Cyan
    Write-Host "  python app.py" -ForegroundColor White
    Write-Host ""
    Write-Host "Then open: http://127.0.0.1:5000" -ForegroundColor Cyan
} else {
    Write-Host "[WARN] Setup completed with warnings" -ForegroundColor Yellow
    Write-Host "Please resolve missing files/models before running" -ForegroundColor Yellow
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Keep window open
Read-Host "Press Enter to exit"
