# findgo deploy - local model mode
#
# Double-click deploy-local.bat. Or run:
#   .\scripts\deploy-local.ps1
#   .\scripts\deploy-local.ps1 -Cpu
#   .\scripts\deploy-local.ps1 -Model qwen2.5:3b-instruct
#
# Creates an isolated .venv-local, installs only what is needed,
# downloads CLIP + BLIP, sets up Ollama, and writes config.local.yaml.

param(
    [switch]$Cpu,
    [switch]$Dev,
    [switch]$SkipDeps,
    [switch]$SkipModels,
    [switch]$SkipOllama,
    [string]$Model = "qwen2.5:7b-instruct",
    [string]$HfMirror = "https://hf-mirror.com",
    [string]$PypiMirror = "https://pypi.tuna.tsinghua.edu.cn/simple",
    [string]$Index = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$VenvDir = Join-Path $Root ".venv-local"
$Py = Join-Path $VenvDir "Scripts\python.exe"

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " findgo deploy - LOCAL MODEL"
Write-Host " root: $Root"
Write-Host " venv: $VenvDir"
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

# 1. Python
$pyVersion = & python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
if ([version]$pyVersion -lt [version]"3.9") {
    Write-Host "ERROR: Python 3.9+ required, found $pyVersion" -ForegroundColor Red
    exit 1
}
Write-Host "[1/6] system Python $pyVersion"

# 2. isolated venv + dependencies
if ($SkipDeps) {
    Write-Host "[2/6] dependencies: skipped"
} else {
    if (-not (Test-Path $Py)) {
        Write-Host "[2/6] creating isolated env .venv-local ..."
        & python -m venv $VenvDir
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: failed to create virtualenv" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "[2/6] reusing existing .venv-local"
    }

    & $Py -m pip install --upgrade pip -i $PypiMirror | Out-Null

    & $Py -c "import torch" 2>$null
    if ($LASTEXITCODE -ne 0) {
        if ($Cpu) {
            Write-Host "      installing CPU torch ..."
            & $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        } else {
            Write-Host "      installing CUDA 12.4 torch ..."
            & $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
        }
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: torch install failed" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "      torch already present"
    }

    Write-Host "      installing project requirements ..."
    & $Py -m pip install -r requirements-core.txt -i $PypiMirror
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: dependency install failed" -ForegroundColor Red
        exit 1
    }

    if ($Dev) {
        & $Py -m pip install -r requirements-dev.txt -i $PypiMirror
    }
}

if (-not (Test-Path $Py)) { $Py = "python" }

# 3. CLIP + BLIP
if ($SkipModels) {
    Write-Host "[3/6] models: skipped"
} else {
    Write-Host "[3/6] downloading CLIP + BLIP ..."
    & $Py scripts\download_models.py --mirror $HfMirror
    if ($LASTEXITCODE -ne 0) {
        Write-Host "      some models failed, re-run later to resume" -ForegroundColor Yellow
    }
}

# 4. Ollama
function Find-Ollama {
    # portable install inside the project takes priority
    $portable = Join-Path $Root "ollama\ollama.exe"
    if (Test-Path $portable) { return $portable }

    $cmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"),
        (Join-Path $env:ProgramFiles "Ollama\ollama.exe")
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

if ($SkipOllama) {
    Write-Host "[4/6] Ollama: skipped"
} else {
    Write-Host "[4/6] setting up Ollama ..."
    $ollamaExe = Find-Ollama
    if (-not $ollamaExe) {
        Write-Host "      Ollama not found, trying winget ..."
        try {
            winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
        } catch {
            Write-Host "      winget failed: $_" -ForegroundColor Yellow
        }
        $ollamaExe = Find-Ollama
    }

    if ($ollamaExe) {
        Write-Host "      pulling $Model ..."
        $oldModels = $env:OLLAMA_MODELS
        $portableModels = Join-Path $Root "ollama\models"
        if ((Test-Path (Join-Path $Root "ollama\ollama.exe")) -and (Test-Path $portableModels)) {
            $env:OLLAMA_MODELS = $portableModels
        }
        & $ollamaExe pull $Model
        if ($oldModels) { $env:OLLAMA_MODELS = $oldModels } else { Remove-Item Env:\OLLAMA_MODELS -ErrorAction SilentlyContinue }
        if ($LASTEXITCODE -ne 0) {
            Write-Host "      pull failed. Run manually: ollama pull $Model" -ForegroundColor Yellow
        }
    } else {
        Write-Host "      Ollama is not installed." -ForegroundColor Yellow
        Write-Host "      Install from https://ollama.com/download then run:"
        Write-Host "        ollama pull $Model"
        Write-Host "      Retrieval still works without it (vector-only ranking)."
    }
}

# 5. config
Write-Host "[5/6] writing config.local.yaml ..."
& $Py scripts\make_config.py --mode local --out config.local.yaml --llm-model $Model

if (-not (Test-Path "configs\agent.yaml")) {
    Write-Host "      initializing configs/agent.yaml from template ..."
    & $Py scripts\init_agent_config.py
}


# 6. optional index
if ($Index) {
    Write-Host "[6/6] indexing: $Index"
    & $Py cli.py index $Index
} else {
    Write-Host "[6/6] index: skipped"
}

Write-Host ""
Write-Host "==============================================" -ForegroundColor Green
Write-Host " DONE"
Write-Host "==============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Start: double-click start-local.bat"
Write-Host "Check: .venv-local\Scripts\python.exe cli.py doctor"
Write-Host ""
