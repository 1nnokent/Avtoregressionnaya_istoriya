param([ValidateSet('cpu', 'cu128')][string]$Device = 'cpu')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:UV_CACHE_DIR = Join-Path $projectRoot 'data/.cache/uv'
uv venv --python 3.12 .venv
if ($LASTEXITCODE -ne 0) { throw 'Failed to create Python environment' }
uv pip install --python .venv/Scripts/python.exe torch==2.9.1 torchvision==0.24.1 --index-url "https://download.pytorch.org/whl/$Device"
if ($LASTEXITCODE -ne 0) { throw 'Failed to install PyTorch' }
uv pip install --python .venv/Scripts/python.exe -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { throw 'Failed to install requirements' }
Write-Output '.venv/Scripts/python.exe run_experiment.py train --config configs/mnist_baseline.yaml'
