# findgo - portable Ollama installer
#
# Installs Ollama into <project>\ollama and keeps models in
# <project>\ollama\models. Nothing is written to C: except temp files.
#
#   .\scripts\install-ollama-portable.ps1
#   .\scripts\install-ollama-portable.ps1 -Model qwen2.5:3b-instruct
#   .\scripts\install-ollama-portable.ps1 -Zip D:\downloads\ollama-windows-amd64.zip
#
# The official release zip is about 1.1GB. If GitHub is slow, pass -Url
# with a mirror, or download the zip yourself and use -Zip.

param(
    [string]$Zip = "",
    [string]$Url = "https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.zip",
    [string]$Model = "qwen2.5:7b-instruct",
    [switch]$SkipPull,
    [switch]$NoStart
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$OllamaDir = Join-Path $Root "ollama"
$ModelsDir = Join-Path $OllamaDir "models"
$Exe = Join-Path $OllamaDir "ollama.exe"

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " portable Ollama"
Write-Host " program: $OllamaDir"
Write-Host " models : $ModelsDir"
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

New-Item -ItemType Directory -Force -Path $OllamaDir | Out-Null
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

# 1. obtain the archive
if (-not (Test-Path $Exe)) {
    $userZip = $Zip
    if (-not $Zip) {
        $Zip = Join-Path $OllamaDir "ollama-windows-amd64.zip"
    }

    $zipReady = (Test-Path $Zip) -and ((Get-Item $Zip).Length -ge 100MB)
    if (-not $zipReady) {
        if ($userZip -and -not (Test-Path $Zip)) {
            Write-Host "ERROR: archive not found: $userZip" -ForegroundColor Red
            exit 1
        }
        if ($userZip) {
            Write-Host "[1/4] local archive looks incomplete: $Zip" -ForegroundColor Yellow
        } else {
            Write-Host "[1/4] downloading $Url"
            Write-Host "      about 1.1GB. Safe to interrupt and re-run: it resumes."
        }

        # --ssl-no-revoke 绕过 schannel 吊销检查
        # -C - 断点续传，中断后重跑不会从头开始
        & curl.exe -L --fail --ssl-no-revoke -C - --retry 3 --retry-delay 2 -o $Zip $Url
        if ($LASTEXITCODE -ne 0) {
            Write-Host "      curl failed, trying Invoke-WebRequest (no resume) ..."
            try {
                $oldProgress = $ProgressPreference
                $ProgressPreference = "SilentlyContinue"
                Invoke-WebRequest -Uri $Url -OutFile $Zip -UseBasicParsing
                $ProgressPreference = $oldProgress
            } catch {
                Write-Host "      Invoke-WebRequest failed: $_" -ForegroundColor Yellow
            }
        }

        if (-not (Test-Path $Zip) -or (Get-Item $Zip).Length -lt 100MB) {
            Write-Host "ERROR: download failed or incomplete" -ForegroundColor Red
            Write-Host "Download manually in a browser or a download manager, then:"
            Write-Host "  .\scripts\install-ollama-portable.ps1 -Zip D:\downloads\ollama-windows-amd64.zip"
            Write-Host "  source: $Url"
            exit 1
        }
    } else {
        Write-Host "[1/4] using local archive: $Zip"
    }

    Write-Host "[2/4] extracting ..."
    # Windows 10+ 自带 bsdtar，解 1GB 的 zip 比 Expand-Archive 快很多
    & tar.exe -xf $Zip -C $OllamaDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "      tar failed, falling back to Expand-Archive ..."
        Expand-Archive -Path $Zip -DestinationPath $OllamaDir -Force
    }
} else {
    Write-Host "[1/4] ollama.exe already present"
    Write-Host "[2/4] extract: skipped"
}

if (-not (Test-Path $Exe)) {
    Write-Host "ERROR: ollama.exe not found after extraction." -ForegroundColor Red
    Write-Host "Check that the zip contains ollama.exe at its root."
    exit 1
}

# 2. version
$env:OLLAMA_MODELS = $ModelsDir
$ver = & $Exe --version 2>&1 | Select-Object -First 1
Write-Host "[3/4] $ver"

# 3. launcher
$launcher = @"
@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "OLLAMA_MODELS=%~dp0models"
"%~dp0ollama.exe" serve
"@
$launcherPath = Join-Path $OllamaDir "start-ollama.bat"
Set-Content -Path $launcherPath -Value $launcher -Encoding ASCII
Write-Host "      launcher: $launcherPath"

# 4. start service and pull model
if (-not $NoStart) {
    $listening = $false
    try {
        $listening = [bool](Get-NetTCPConnection -LocalPort 11434 -State Listen -ErrorAction SilentlyContinue)
    } catch {
        $listening = $false
    }

    if (-not $listening) {
        Write-Host "      starting ollama serve ..."
        Start-Process -FilePath $Exe -ArgumentList "serve" -WindowStyle Hidden -WorkingDirectory $OllamaDir
        Start-Sleep -Seconds 4
    } else {
        Write-Host "      11434 already listening"
    }

    if (-not $SkipPull) {
        Write-Host "[4/4] pulling $Model ..."
        & $Exe pull $Model
        if ($LASTEXITCODE -ne 0) {
            Write-Host "      pull failed. Retry with:" -ForegroundColor Yellow
            Write-Host "      $Exe pull $Model"
        }
    } else {
        Write-Host "[4/4] model pull: skipped"
    }
} else {
    Write-Host "[4/4] service start: skipped"
}

Write-Host ""
Write-Host "==============================================" -ForegroundColor Green
Write-Host " DONE"
Write-Host "==============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Start service manually: $launcherPath"
Write-Host "Verify: python scripts\test_llm.py"
Write-Host ""
