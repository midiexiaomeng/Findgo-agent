# findgo environment setup (Windows PowerShell)
#
#   .\scripts\setup.ps1              # install into current Python
#   .\scripts\setup.ps1 -Venv        # create .venv and install there
#   .\scripts\setup.ps1 -Cpu         # force CPU-only torch
#   .\scripts\setup.ps1 -Mirror https://pypi.org/simple
#
# This script is idempotent: re-running it only fills in what is missing.

param(
    [switch]$Venv,
    [switch]$Cpu,
    [string]$Mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "== findgo setup ==" -ForegroundColor Cyan
Write-Host "project: $Root"

# 1. Python
$Python = "python"
if ($Venv) {
    if (-not (Test-Path ".venv\Scripts\python.exe")) {
        Write-Host "creating .venv ..."
        & python -m venv .venv
    }
    $Python = Join-Path $Root ".venv\Scripts\python.exe"
}

$version = & $Python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
Write-Host "python: $version"
if ([version]$version -lt [version]"3.9") {
    Write-Host "ERROR: Python 3.9+ required" -ForegroundColor Red
    exit 1
}

# 2. torch first (it has its own index and is the biggest download)
$torchOk = $false
try {
    & $Python -c "import torch" 2>$null
    if ($LASTEXITCODE -eq 0) { $torchOk = $true }
} catch { $torchOk = $false }

if ($torchOk) {
    Write-Host "torch: already installed, skip"
} elseif ($Cpu) {
    Write-Host "installing CPU torch ..."
    & $Python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
} else {
    Write-Host "installing CUDA 12.4 torch (use -Cpu for CPU-only) ..."
    & $Python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
}

# 3. rest of requirements
Write-Host "installing requirements ..."
& $Python -m pip install --upgrade pip -i $Mirror
& $Python -m pip install -r requirements.txt -i $Mirror
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip install failed" -ForegroundColor Red
    exit 1
}

# 4. models
Write-Host ""
Write-Host "Next step: download models" -ForegroundColor Yellow
Write-Host "  $Python scripts\download_models.py"
Write-Host "  $Python scripts\download_models.py --with-llm   # also pull Qwen2.5-7B via Ollama"
Write-Host ""
Write-Host "Then start:"
if ($Venv) {
    Write-Host "  .\.venv\Scripts\python.exe app.py"
} else {
    Write-Host "  python app.py"
}
Write-Host "Done." -ForegroundColor Green
