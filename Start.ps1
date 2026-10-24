$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$envPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $envPython)) { throw 'Run Setup.ps1 first.' }
if (Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Port 7860 is already in use. Stop the existing Chatterbox server before starting this copy.'
}
& $envPython local_app.py
if ($LASTEXITCODE -ne 0) { throw 'Chatterbox stopped with an error.' }
