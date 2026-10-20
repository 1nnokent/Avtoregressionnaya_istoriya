param([string]$Python = "python", [switch]$Dev)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path $PSScriptRoot -Parent
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    & $Python -m venv (Join-Path $projectRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Could not create Python environment" }
}
& $venvPython -c "import sys; assert sys.version_info[:2] == (3, 12), 'Use Python 3.12'"
if ($LASTEXITCODE -ne 0) { throw "Python 3.12 is required" }
& $venvPython -m ensurepip
if ($LASTEXITCODE -ne 0) { throw "Could not set up pip" }
$requirementsName = if ($Dev) { "requirements-dev.txt" } else { "requirements.txt" }
& $venvPython -m pip install -r (Join-Path $projectRoot $requirementsName)
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
& $venvPython (Join-Path $projectRoot "run_experiment.py") --dry-run
if ($LASTEXITCODE -ne 0) { throw "Configuration validation failed" }
