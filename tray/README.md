# Chatterbox Server tray application

A small native Windows application written in Rust. Launch **Chatterbox Server** from the Start menu; it starts the Python server without opening a console and places a waveform icon in the system tray (possibly under the hidden-icons arrow).

Double-click the icon to open Chatterbox. Right-click for server status, the web interface, API documentation, Start/Stop server, logs, and Quit. Quit stops a server launched by this application. If a server was already running elsewhere, the app leaves that process alone and disables Stop for it.

The wrapper manages the server process tree with a Windows Job Object. Closing its process also releases the managed Python processes and CUDA context. A named mutex prevents duplicate tray instances. Explorer restarts are handled by re-registering the tray icon. Tailscale Serve remains independently configured.

## Install

First configure the Python server using the repository's `Setup.ps1`. Install Rust with a Windows toolchain and its linker/build tools. Either the MSVC toolchain with Visual Studio C++ Build Tools or the GNU toolchain with MinGW is suitable. Then, from PowerShell:

```powershell
.\tray\Install.ps1
```

To reuse an existing Python environment:

```powershell
.\tray\Install.ps1 -Python 'C:\path\to\environment\Scripts\python.exe'
```

Installation builds the release binary using Cargo.lock, copies the application and icon to `%LOCALAPPDATA%\Programs\Chatterbox Server`, adds a Start-menu shortcut, and registers **Chatterbox Server** under Windows installed apps. It does not require administrator permissions or bundle Python/model weights. The installed `config.json` records the repository and Python paths; keep those locations available.

Add `-StartWithWindows` to enable starting the tray app at Windows sign-in. This is disabled by default. Add `-NoLaunch` to install without launching. `-Cargo` accepts a path to cargo.exe if Cargo is not on PATH. Quit the existing tray application before reinstalling/upgrading it.

The server continues to read ignored data from `data/models` and `data/voices` and write logs to `runtime`. An existing `.venv` in the repository is the default Python environment. The installer checks its imports and CUDA availability before installing.

Uninstall using Windows installed apps, or run the installed `Uninstall.ps1` after quitting the tray app. Uninstallation preserves the repository, Python environment, downloaded models, and saved voices.

## Build and control

```powershell
Set-Location tray
cargo build --release --locked
cargo fmt --check
cargo clippy --locked -- -D warnings
```

The executable also supports `--start-server`, `--stop-server`, and `--quit` to send a command to an already running tray instance. Start/Stop only affects the managed server. A raw build requires `config.json` and `app.ico` alongside the executable; normally the installer provides these.

This wrapper is currently Windows-only. Speech inference remains in Python/PyTorch.
