# System preflight inspection script for Windows PowerShell
Write-Host "=== Preflight System Inspection: Semantic NOMA-OFDMA ===" -ForegroundColor Cyan

# 1. Disk Space Check
$drive = Get-PSDrive C
$freeGB = [math]::Round($drive.Free / 1GB, 2)
Write-Host "Disk Free Space on C:: $freeGB GB"
if ($freeGB -lt 5.0) {
    Write-Error "Insufficient disk space ($freeGB GB < 5 GB minimum requirement)."
    exit 1
} else {
    Write-Host "  -> Disk space requirement met (>= 5 GB)." -ForegroundColor Green
}

# 2. Python Environment Check
$venvPython = ".\.venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $pyVer = & $venvPython --version
    Write-Host "Virtual Environment Python: $pyVer" -ForegroundColor Green
    
    # Check PyTorch CPU and device
    & $venvPython -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
} else {
    Write-Host "Virtual environment .venv not yet created." -ForegroundColor Yellow
}

Write-Host "Preflight check completed." -ForegroundColor Cyan
