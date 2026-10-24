# Chatterbox local REST server

Windows/CUDA setup for Chatterbox Multilingual V3, with a browser interface, REST speech generation, explicit GPU model management, persistent voice samples, and private Tailscale HTTPS access.

This repository retains Resemble AI's upstream Git history and source. Local additions are on branch `local-server`; remote `upstream` points to https://github.com/resemble-ai/chatterbox. The upstream base commit is `5de7a54aa4e5e2baadb0182dde554908b48b85c2`. See [README.upstream.md](README.upstream.md) for model information and [LICENSE](LICENSE) for the upstream MIT license.

## Setup

Requirements: Windows, Python 3.11 x64, Git, a CUDA-capable NVIDIA GPU with a suitable driver, and enough disk space for the environment and model files. The configuration was tested on an RTX 4090 with 24 GB VRAM. Tailscale is optional for local use. FFmpeg is useful for browser audio input.

```powershell
Set-Location D:\Git\Projects\chatterbox-server
.\Setup.ps1 -Python C:\ProgramData\anaconda3\python.exe
.\Start.ps1
```

Pass your own Python 3.11 executable if different. Setup creates `.venv`, installs the pinned Windows/CUDA dependencies, installs this source in editable mode, and downloads the multilingual model. Large artifacts use chunked downloads with SHA256 verification against `scripts/artifacts.json`. Already verified files are reused. `-SkipModels` installs just the environment; download models into `data/models` later with:

```powershell
.\.venv\Scripts\python.exe scripts/download_artifacts.py --group models
```

The Python environment, downloaded models, voice recordings, logs, and other runtime files are excluded from Git. No private voice sample is included. This repository is a reproducible source copy; the server source and runtime data now live in this repository. Models are stored in `data/models` and saved recordings plus metadata in `data/voices`. The entire `data` folder is ignored by Git. The existing Python environment remains in the original Codex workspace; the original launcher reuses it to run this repository's source. Run `Setup.ps1` to create an independent environment here.

## Browser and REST API

For a Windows system-tray application with Start-menu installation and server controls, see [tray/README.md](tray/README.md). It manages the existing Python server; inference stays in PyTorch.

Browser: http://127.0.0.1:7860/
Interactive API docs: http://127.0.0.1:7860/docs
OpenAPI: http://127.0.0.1:7860/openapi.json

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/model/status` | State and PyTorch GPU allocation metrics |
| POST | `/model/load` | Load Multilingual V3 onto the GPU |
| POST | `/model/unload` | Release model weights and audio caches |
| GET / PUT | `/model/settings` | Read/save `idle_timeout_seconds` (default 7200; 0 disables; maximum 30 days) |
| POST | `/v1/audio/speech` | Generate WAV audio from JSON |
| POST | `/voices` | Upload multipart fields `name` and `sample` |
| GET | `/voices` | List persistent voice IDs and metadata |
| GET | `/voices/{id}` | Retrieve a voice's metadata |
| GET | `/voices/{id}/sample` | Download its WAV reference |
| DELETE | `/voices/{id}` | Delete the stored WAV and metadata (204; unknown IDs return 404) |

Speech automatically loads the model when needed. Load, unload, and generation operations share a lock, so unloading waits for active generation. The model automatically unloads after two hours of inactivity by default. Configure `/model/settings`, the tray timeout presets, or Model memory in the web interface; zero disables automatic unloading. Settings persist in ignored `data/settings.json`. The timer resets after generation finishes or an explicit load request, and status polling does not reset it. Active generation is never interrupted. A small CUDA runtime allocation/context can remain after unloading; approximately 8 MB of PyTorch allocations remained in testing. Metrics report PyTorch allocations rather than all process VRAM.

Speech JSON accepts `input` (alias `text`, 1–3000 characters), `language` (default `en`), `voice` (optional stored ID), `exaggeration` (0–2, default 0.5), `cfg_weight` (0–1, default 0.5), and `response_format` (`wav` only). This is a local speech endpoint, not a complete implementation of the OpenAI API.

Voice samples must contain 1–120 seconds of usable audio, at most 25 MB. WAV and FLAC are recommended. Samples are stored as mono PCM WAV with JSON metadata. Names are labels; use the returned ID for generation. Recordings persist across restarts. In the browser, a selected stored voice takes priority over an uploaded reference. Click Refresh stored voices after adding samples through REST.

```python
import requests

base = 'http://127.0.0.1:7860'
with open('my-voice.wav', 'rb') as audio:
    response = requests.post(base + '/voices', data={'name': 'My voice'},
        files={'sample': ('my-voice.wav', audio, 'audio/wav')}, timeout=60)
response.raise_for_status()
voice_id = response.json()['id']

print(requests.get(base + '/voices', timeout=30).json())
response = requests.post(base + '/v1/audio/speech', json={
    'input': 'Hello from my saved voice.', 'language': 'en', 'voice': voice_id,
}, timeout=180)
response.raise_for_status()
with open('speech.wav', 'wb') as output:
    output.write(response.content)
requests.post(base + '/model/unload', timeout=180).raise_for_status()
```

To remove a saved voice in the browser, select it and click **Delete selected stored voice**. Deletion removes its WAV and metadata, resets the selection, and waits for active speech generation to finish. Built-in voices cannot be deleted. API clients can use `DELETE /voices/{id}`; subsequent retrieval or generation with that ID returns 404.

## Tailscale

Connect Tailscale on the server and client devices, then run:

```powershell
.\Configure-Tailscale.ps1
```

This publishes localhost:7860 privately over HTTPS on port 8443. It preserves other Serve endpoints and refuses to replace an unrelated service on that port. HTTPS may require enabling it through the Tailscale account's setup flow on a new tailnet. The script prints the device URL; use it as the REST base URL remotely. Access follows existing tailnet rules. No public Funnel or application API key is configured.

Remove just this proxy with `tailscale serve --https=8443 off`. The proxy persists independently of the application. The app must be started again after reboot; no automatic-start task is installed.

## Verification

Run a REST integration check against a running instance:

```powershell
.\.venv\Scripts\python.exe scripts/verify_api.py
```

It checks model loading, speech generation, storing/retrieving a synthetic demo voice, generation with that voice, and unloading. It leaves one synthetic test sample in the private voice store and finishes with the model unloaded. Pass `--base-url` to test the Tailscale URL.

Idle timeout verification: `.\.venv\Scripts\python.exe scripts/verify_idle_timeout.py`. This briefly changes the timeout to test real GPU generation and idle expiration, then restores the previous setting and unloads the model.
