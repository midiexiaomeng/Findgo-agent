# imgsearch one-click deployment (Windows PowerShell)
#
#   .\scripts\deploy.ps1
#   .\scripts\deploy.ps1 -Cpu                 # CPU only
#   .\scripts\deploy.ps1 -Venv                # isolated virtualenv
#   .\scripts\deploy.ps1 -NoLlm               # skip Qwen download
#   .\scripts\deploy.ps1 -Index D:\Photos     # build index after setup
#
# Models are never bundled. This script pulls them from mirrors and is
# safe to re-run: existing files are skipped, partial downloads resume.

param(
    [switch]$Venv,
    [switch]$Cpu,
    [switch]$NoLlm,
    [switch]$SkipDeps,
    [switch]$SkipModels,
    [string]$Mirror = "https://pypi.tuna.tsinghua.edu.cn/simple",
    [string]$HfMirror = "https://hf-mirror.com",
    [string]$Index = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " imgsearch deploy"
Write-Host " root: $Root"
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

# 1. Python
$Python = "python"
$pyVersion = & $Python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
if ([version]$pyVersion -lt [version]"3.9") {
    Write-Host "ERROR: Python 3.9+ required, found $pyVersion" -ForegroundColor Red
    exit 1
}
Write-Host "[1/4] Python $pyVersion"

# 2. Dependencies
if ($SkipDeps) {
    Write-Host "[2/4] dependencies: skipped"
} else {
    Write-Host "[2/4] installing dependencies ..."
    & "$PSScriptRoot\setup.ps1" -Venv:$Venv -Cpu:$Cpu -Mirror $Mirror
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: dependency installation failed" -ForegroundColor Red
        exit 1
    }
}

if ($Venv -and (Test-Path ".venv\Scripts\python.exe")) {
    $Python = Join-Path $Root ".venv\Scripts\python.exe"
}

# 3. Models
if ($SkipModels) {
    Write-Host "[3/4] models: skipped"
} else {
    Write-Host "[3/4] downloading CLIP + BLIP ..."
    $modelArgs = @("scripts\download_models.py", "--mirror", $HfMirror)
    if (-not $NoLlm) {
        $modelArgs += "--with-llm"
    }
    & $Python @modelArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Some models failed. Re-run this script later to resume." -ForegroundColor Yellow
    }
}

# 4. Optional index
if ($Index) {
    Write-Host "[4/4] indexing: $Index"
    & $Python "cli.py" index $Index
} else {
    Write-Host "[4/4] index: skipped (use the UI or cli.py index later)"
}

Write-Host ""
Write-Host "==============================================" -ForegroundColor Green
Write-Host " deploy finished"
Write-Host "==============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Start with any of:"
Write-Host "  double-click  the launcher .bat in the project root"
Write-Host "  $Python desktop.py --mode edge"
Write-Host "  $Python app.py"
Write-Host ""
Write-Host "Health check: $Python cli.py doctor"
Write-Host ""
