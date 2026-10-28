# findgo deploy - cloud API mode
#
# Double-click deploy-api.bat. Or run:
#   .\scripts\deploy-api.ps1 -Provider deepseek -ApiKey sk-xxx
#   .\scripts\deploy-api.ps1 -Provider custom -BaseUrl http://localhost:8000/v1 -ApiKey x -Model qwen
#
# Creates an isolated .venv-api, installs only what is needed,
# downloads CLIP + BLIP locally, and points the LLM reranker at an API.
# Embedding and captioning still run on this machine.

param(
    [ValidateSet("dashscope", "deepseek", "moonshot", "openai", "custom")]
    [string]$Provider = "dashscope",
    [string]$BaseUrl = "",
    [string]$ApiKey = "",
    [string]$Model = "",
    [switch]$Cpu,
    [switch]$Dev,
    [switch]$SkipDeps,
    [switch]$SkipModels,
    [string]$HfMirror = "https://hf-mirror.com",
    [string]$PypiMirror = "https://pypi.tuna.tsinghua.edu.cn/simple",
    [string]$Index = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$VenvDir = Join-Path $Root ".venv-api"
$Py = Join-Path $VenvDir "Scripts\python.exe"

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " findgo deploy - API MODE"
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
        Write-Host "[2/6] creating isolated env .venv-api ..."
        & python -m venv $VenvDir
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: failed to create virtualenv" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "[2/6] reusing existing .venv-api"
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

# 4. API endpoint
Write-Host "[4/6] configuring LLM API ..."
if (-not $BaseUrl -or -not $Model) {
    switch ($Provider) {
        "dashscope" {
            if (-not $BaseUrl) { $BaseUrl = "https://dashscope.aliyuncs.com/compatible-mode/v1" }
            if (-not $Model) { $Model = "qwen-plus" }
        }
        "deepseek" {
            if (-not $BaseUrl) { $BaseUrl = "https://api.deepseek.com/v1" }
            if (-not $Model) { $Model = "deepseek-chat" }
        }
        "moonshot" {
            if (-not $BaseUrl) { $BaseUrl = "https://api.moonshot.cn/v1" }
            if (-not $Model) { $Model = "moonshot-v1-8k" }
        }
        "openai" {
            if (-not $BaseUrl) { $BaseUrl = "https://api.openai.com/v1" }
            if (-not $Model) { $Model = "gpt-4o-mini" }
        }
        "custom" {
            if (-not $BaseUrl) { $BaseUrl = Read-Host "Base URL (e.g. http://localhost:8000/v1)" }
            if (-not $Model) { $Model = Read-Host "Model name" }
        }
    }
}

if (-not $ApiKey) {
    Write-Host "      provider : $Provider"
    Write-Host "      endpoint : $BaseUrl"
    Write-Host "      model    : $Model"
    $secure = Read-Host "      API Key" -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        $ApiKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    } finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
}

if (-not $ApiKey) {
    Write-Host "ERROR: API key is required" -ForegroundColor Red
    exit 1
}

# 5. config + connectivity test
Write-Host "[5/6] writing config.api.yaml ..."
& $Py scripts\make_config.py --mode api --out config.api.yaml --base-url $BaseUrl --api-key $ApiKey --model $Model

if (-not (Test-Path "configs\agent.yaml")) {
    Write-Host "      initializing configs/agent.yaml from template ..."
    & $Py scripts\init_agent_config.py
}


Write-Host "      testing endpoint ..."
& $Py scripts\test_llm.py --config config.api.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "      API test failed. Check key, endpoint and model name." -ForegroundColor Yellow
    Write-Host "      Retrieval still works, but LLM rerank will be skipped."
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
Write-Host "Start: double-click start-api.bat"
Write-Host "Check: .venv-api\Scripts\python.exe cli.py doctor"
Write-Host ""
Write-Host "Note: config.api.yaml contains your API key. Do not share it."
Write-Host ""
