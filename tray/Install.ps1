param(
    [string]$Python,
    [switch]$StartWithWindows,
    [switch]$NoLaunch,
    [string]$Cargo = 'cargo'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = Join-Path $repo '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python)) { throw 'Pass -Python with the server environment executable, or run Setup.ps1 first.' }
$Python = (Resolve-Path -LiteralPath $Python).Path
& $Python -c "import chatterbox, gradio, torch; assert torch.cuda.is_available(), 'CUDA unavailable'"
if ($LASTEXITCODE -ne 0) { throw 'The Python environment cannot run the server.' }
Push-Location $PSScriptRoot
try {
    & $Cargo build --release --locked
    if ($LASTEXITCODE -ne 0) { throw 'Rust build failed. Install Rust and its Windows build tools first.' }
} finally { Pop-Location }
$install = Join-Path $env:LOCALAPPDATA 'Programs\Chatterbox Server'
New-Item -ItemType Directory -Path $install -Force | Out-Null
$exe = Join-Path $install 'chatterbox-tray.exe'
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'target\release\chatterbox-tray.exe') -Destination $exe -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'app.ico') -Destination (Join-Path $install 'app.ico') -Force
$configuration = @{ repository = $repo; python = $Python } | ConvertTo-Json
[IO.File]::WriteAllText((Join-Path $install 'config.json'), $configuration, (New-Object Text.UTF8Encoding($false)))
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Programs')) 'Chatterbox Server.lnk'))
$shortcut.TargetPath = $exe
$shortcut.WorkingDirectory = $install
$shortcut.IconLocation = (Join-Path $install 'app.ico')
$shortcut.Description = 'Chatterbox local speech server and tray controls'
$shortcut.Save()
$runKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run'
if ($StartWithWindows) {
    New-Item -Path $runKey -Force | Out-Null
    New-ItemProperty -Path $runKey -Name ChatterboxServer -Value ('"' + $exe + '"') -PropertyType String -Force | Out-Null
}
$uninstallKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\ChatterboxServerTray'
New-Item -Path $uninstallKey -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'Uninstall.ps1') -Destination (Join-Path $install 'Uninstall.ps1') -Force
$uninstall = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' + (Join-Path $install 'Uninstall.ps1') + '"'
(@{ DisplayName = 'Chatterbox Server'; DisplayVersion = '0.1.0'; Publisher = 'nemoxion'; InstallLocation = $install; DisplayIcon = (Join-Path $install 'app.ico'); UninstallString = $uninstall }).GetEnumerator() | ForEach-Object {
    New-ItemProperty -Path $uninstallKey -Name $_.Key -Value $_.Value -PropertyType String -Force | Out-Null
}
New-ItemProperty -Path $uninstallKey -Name NoModify -Value 1 -PropertyType DWord -Force | Out-Null
New-ItemProperty -Path $uninstallKey -Name NoRepair -Value 1 -PropertyType DWord -Force | Out-Null
if (-not $NoLaunch) { Start-Process -FilePath $exe -WorkingDirectory $install -WindowStyle Hidden }
Write-Output "Installed: $exe"
Write-Output 'Launch Chatterbox Server from the Start menu. Right-click its system-tray icon for controls.'
