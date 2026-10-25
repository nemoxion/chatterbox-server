# Speech API and word timestamps

The local server supports the request/response shape of OpenAI's speech API where the local model can implement it. It runs Chatterbox, not OpenAI's hosted models or voices. Reference: https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create

## Standard speech endpoint

`POST /v1/audio/speech` returns binary audio. Supported fields:

| Field | Behavior |
| --- | --- |
| `model` | `chatterbox-multilingual-v3`; omitted defaults to this model for existing clients |
| `input` | 1–4096 characters; legacy `text` alias remains accepted |
| `voice` | `default`, a saved voice ID, or `{ "id": "saved-voice-id" }`; omitted/null uses default |
| `response_format` | `mp3` (default), `wav`, `flac`, `opus` (Ogg), `aac` (ADTS), or `pcm` |
| `speed` | 0.25–4, default 1; FFmpeg tempo adjustment preserves pitch |
| `stream_format` | `audio` only; complete synthesis is buffered before returning bytes |
| `language` | Local extension: model language code, default `en` |
| `exaggeration`, `cfg_weight` | Existing Chatterbox controls |

Raw PCM is mono 24 kHz signed 16-bit little-endian without headers. FFmpeg is required for MP3/Opus/AAC and speed changes. WAV and FLAC use libsndfile. Opus decodes at 48 kHz.

`GET /v1/models` and `GET /v1/models/chatterbox-multilingual-v3` support model discovery. Speech errors use an OpenAI-style `error` object with `message`, `type`, `param`, and `code`; invalid speech requests return 400. Unknown model or voice IDs return 404. Model/voice administration endpoints remain local extensions.

Nonempty `instructions`, SSE output, unknown fields, OpenAI model names, and unavailable OpenAI voice names are rejected explicitly. There is no hosted OpenAI connection or billing. Bearer headers from SDK clients are accepted, but the local service does not enforce an API key; use private Tailscale access. This is speech-client compatibility, not complete OpenAI API compatibility or incremental low-latency streaming.

```python
from openai import OpenAI

client = OpenAI(base_url='http://127.0.0.1:7860/v1', api_key='local')
response = client.audio.speech.create(
    model='chatterbox-multilingual-v3',
    voice='default',  # or a saved voice ID / {'id': '...'}
    input='Hello from Chatterbox.',
    response_format='wav',
    speed=1.1,
)
response.write_to_file('speech.wav')
```

Use `extra_body={'language': 'de'}` for German through the SDK. For remote calls, use the Tailscale HTTPS URL with `/v1` appended.

**Compatibility change:** older versions defaulted to WAV; the default is now MP3 to match OpenAI. Existing clients that need WAV must explicitly send `response_format: "wav"`.

## Optional audio plus timestamps

`POST /v1/audio/speech/with-timestamps` accepts the same JSON request and returns JSON with base64-encoded audio plus measured word alignment. This endpoint is a local extension; OpenAI's speech endpoint does not define this response.

```python
import base64
import requests

result = requests.post('http://127.0.0.1:7860/v1/audio/speech/with-timestamps', json={
    'model': 'chatterbox-multilingual-v3',
    'input': 'Hello world. This is a test.',
    'voice': 'default',
    'language': 'en',
    'response_format': 'wav',
}, timeout=240)
result.raise_for_status()
result = result.json()
with open('speech.wav', 'wb') as output:
    output.write(base64.b64decode(result['audio_base64']))
for word in result['words']:
    print(word['word'], word['start'], word['end'], word['confidence'])
```

Response fields include `audio_base64`, `content_type`, `response_format`, `sample_rate`, `duration`, `text`, `words`, and alignment method/language/warnings. Each word has its original text, start/end in seconds, confidence, `aligned`, and character offsets `text_start`/`text_end` into the input. Words are split on whitespace. Punctuation-only tokens may have null timings. Numeric tokens are expanded for the aligner, but their original form is retained in the output.

The server runs Wav2Vec2 inference followed by CTC forced alignment of the known transcript against the generated speech. It aligns the audio after speed adjustment and encoding/decoding, so codec padding is included. Alignment is a model estimate: confidence is a character-emission score, not a calibrated probability of correctness. TTS pronunciation differences, hallucinations, unusual symbols, and text normalization can reduce alignment accuracy. Unalignable tokens are marked with null timings; no evenly spaced timestamps are fabricated. A failure to align returns an error.

Alignment currently supports **English (`en`) and German (`de`)**. Other Chatterbox languages remain available through the normal speech endpoint. Alignment runs on CPU, serialized to bound memory use, with model weights scoped to the request; it adds latency and temporary RAM use without retaining another GPU model.

## Install alignment weights

```powershell
.\Setup.ps1 -Alignment
# Or in an existing environment:
.\.venv\Scripts\python.exe scripts/download_artifacts.py --group alignment
```

Pinned, checksum-verified artifacts are described in `scripts/alignment-artifacts.json` and stored in ignored `data/alignment`. They are optional for regular TTS. The English model is `facebook/wav2vec2-base-960h`; German uses `jonatasgrosman/wav2vec2-large-xlsr-53-german`. Both model cards specify Apache 2.0: https://huggingface.co/facebook/wav2vec2-base-960h and https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-german.

To verify all formats, actual SDK compatibility, and English/German timestamps, install the optional `openai` Python client and run `scripts/verify_speech_compatibility.py`. The server itself does not require that client.
