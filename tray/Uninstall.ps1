$ErrorActionPreference = 'Stop'
$install = [IO.Path]::GetFullPath($PSScriptRoot)
$expected = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'Programs\Chatterbox Server'))
if ($install -ne $expected) { throw 'Run the installed uninstaller from the application directory.' }
$running = Get-Process -Name chatterbox-tray -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq (Join-Path $install 'chatterbox-tray.exe') }
if ($running) { throw 'Quit Chatterbox Server using its tray menu, then run uninstall again.' }
Remove-ItemProperty -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run' -Name ChatterboxServer -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path ([Environment]::GetFolderPath('Programs')) 'Chatterbox Server.lnk') -ErrorAction SilentlyContinue
Remove-Item -LiteralPath 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\ChatterboxServerTray' -Recurse -ErrorAction SilentlyContinue
# Only application files are removed. Repository, models, voices and Python are preserved.
foreach ($name in @('chatterbox-tray.exe', 'app.ico', 'config.json', 'Uninstall.ps1')) {
    Remove-Item -LiteralPath (Join-Path $install $name) -ErrorAction SilentlyContinue
}
Write-Output 'Chatterbox tray application uninstalled. Server source and data are preserved.'
