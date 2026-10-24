param(
    [string]$Python = 'python',
    [switch]$SkipModels
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& $Python -c "import sys; assert sys.version_info[:2] == (3, 11), 'Use Python 3.11 for this pinned Windows setup'"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 is required. Pass -Python with its executable path.' }
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
}
$envPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $envPython -c "import sys; assert sys.version_info[:2] == (3, 11)"
if ($LASTEXITCODE -ne 0) { throw 'Existing virtual environment must use Python 3.11' }
& $envPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }
& $envPython scripts/download_artifacts.py --group cuda
if ($LASTEXITCODE -ne 0) { throw 'CUDA download failed' }
& $envPython -m pip install '.downloads/torch-2.6.0+cu124-cp311-cp311-win_amd64.whl' '.downloads/torchaudio-2.6.0+cu124-cp311-cp311-win_amd64.whl'
if ($LASTEXITCODE -ne 0) { throw 'CUDA installation failed' }
& $envPython -m pip install -r requirements-windows-cu124.lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $envPython -m pip install --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'Chatterbox installation failed' }
& $envPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
& $envPython -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed' }
if (-not $SkipModels) {
    & $envPython scripts/download_artifacts.py --group models
    if ($LASTEXITCODE -ne 0) { throw 'Model download failed' }
}
Write-Output 'Setup complete. Run .\Start.ps1 to start the server.'
