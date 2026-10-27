# Build a release zip that can be unzipped and used directly.
# Models are NOT included; they are fetched by scripts/download_models.py.
#
#   .\scripts\build_package.ps1
#   .\scripts\build_package.ps1 -Version 2.0.1

param(
    [string]$Version = "2.0.0"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Dist = Join-Path $Root "dist"
$Name = "imgsearch-$Version"
$Stage = Join-Path $Dist $Name
$Zip = Join-Path $Dist "$Name.zip"

Write-Host "== build package ==" -ForegroundColor Cyan
Write-Host "source: $Root"

New-Item -ItemType Directory -Force -Path $Dist | Out-Null
if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }

# Directories that must never ship: models, indexes, venv, caches, old code
$ExcludeDirs = @(
    "models", "data", "datasets", "ollama", ".venv", ".venv-local", ".venv-api", "venv",
    "__pycache__", ".git", "legacy", "dist", ".pytest_cache", ".idea", ".vscode", "logs"
)
$ExcludeFiles = @("*.pyc", "*.pyo", "*.log", "*.sqlite", "*.zip", "config.local.yaml", "config.api.yaml", "agent.yaml")

Write-Host "copying ..."
$null = robocopy $Root $Stage /E /XD $ExcludeDirs /XF $ExcludeFiles /NFL /NDL /NJH /NJS /NP
if ($LASTEXITCODE -ge 8) {
    Write-Host "ERROR: robocopy failed ($LASTEXITCODE)" -ForegroundColor Red
    exit 1
}

# Keep the expected runtime folders with a short note
New-Item -ItemType Directory -Force -Path (Join-Path $Stage "models") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Stage "data") | Out-Null
@"
Models are not bundled.

Run:
  python scripts\download_models.py
  python scripts\download_models.py --with-llm

Or let the first search download CLIP automatically.
"@ | Set-Content -Path (Join-Path $Stage "models\README.txt") -Encoding UTF8

# Version stamp
$Version | Set-Content -Path (Join-Path $Stage "VERSION") -Encoding UTF8

if (Test-Path $Zip) { Remove-Item -Force $Zip }
Write-Host "compressing ..."
Compress-Archive -Path $Stage -DestinationPath $Zip -CompressionLevel Optimal

$sizeMb = [math]::Round((Get-Item $Zip).Length / 1MB, 2)
Write-Host ""
Write-Host "Done: $Zip  ($sizeMb MB)" -ForegroundColor Green
Write-Host ""
Write-Host "On the target machine:"
Write-Host "  1. unzip"
Write-Host "  2. double-click deploy-local.bat   (local Qwen via Ollama)"
Write-Host "     or          deploy-api.bat     (cloud LLM API)"
Write-Host "  3. double-click start-local.bat or start-api.bat"
