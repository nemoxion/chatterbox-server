"""Local Chatterbox browser UI and REST API."""
import gc
import io
import json
import re
import threading
import uuid
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import AliasChoices, BaseModel, Field
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
from chatterbox.models.s3gen.s3gen import get_resampler
from chatterbox.models.s3gen.utils import mel as mel_utils
from s3tokenizer.utils import _mel_filters

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
VOICES = DATA / 'voices'
VOICES.mkdir(parents=True, exist_ok=True)
model = None
default_conds = None
state = 'unloaded'
lock = threading.Lock()
voice_lock = threading.Lock()
SETTINGS_FILE = DATA / 'settings.json'
idle_timeout_seconds = 7200
if SETTINGS_FILE.exists():
    idle_timeout_seconds = json.loads(SETTINGS_FILE.read_text(encoding='utf-8'))['idle_timeout_seconds']
    if type(idle_timeout_seconds) is not int or not 0 <= idle_timeout_seconds <= 2592000:
        raise ValueError('Invalid idle timeout in data/settings.json')
last_used = None

def unload_locked():
    global model, default_conds, state, last_used
    state = 'unloading'
    model = None
    default_conds = None
    get_resampler.cache_clear()
    mel_utils.mel_basis.clear()
    mel_utils.hann_window.clear()
    _mel_filters.cache_clear()
    torch.backends.cuda.cufft_plan_cache.clear()
    gc.collect()
    torch.cuda.empty_cache()
    last_used = None
    state = 'unloaded'

def idle_watch(stop):
    while not stop.wait(1):
        if not lock.acquire(blocking=False):
            continue
        try:
            if model is not None and idle_timeout_seconds and last_used is not None and time.monotonic() - last_used >= idle_timeout_seconds:
                unload_locked()
        finally:
            lock.release()

@asynccontextmanager
async def lifespan(app):
    stop = threading.Event()
    worker = threading.Thread(target=idle_watch, args=(stop,), daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=2)

def status():
    return {
        'model': 'chatterbox-multilingual-v3', 'state': state,
        'loaded': model is not None, 'device': 'cuda',
        'idle_timeout_seconds': idle_timeout_seconds,
        'idle_seconds': round(time.monotonic() - last_used, 1) if last_used is not None else None,
        'gpu': torch.cuda.get_device_name(0),
        'gpu_allocated_mb': round(torch.cuda.memory_allocated() / 1024**2, 1),
        'gpu_reserved_mb': round(torch.cuda.memory_reserved() / 1024**2, 1),
    }

def load_locked():
    global model, default_conds, state, last_used
    if model is None:
        state = 'loading'
        try:
            model = ChatterboxMultilingualTTS.from_local(DATA / 'models', device='cuda', t3_model='v3')
            default_conds = model.conds
        finally:
            state = 'loaded' if model is not None else 'unloaded'
    last_used = time.monotonic()

def voice_metadata(voice_id):
    if not re.fullmatch(r'[0-9a-f]{32}', voice_id):
        raise HTTPException(404, 'Voice not found')
    path = VOICES / f'{voice_id}.json'
    if not path.is_file():
        raise HTTPException(404, 'Voice not found')
    return json.loads(path.read_text(encoding='utf-8'))

def list_voices():
    with voice_lock:
        return sorted([json.loads(p.read_text(encoding='utf-8')) for p in VOICES.glob('*.json')], key=lambda v: v['created_at'])

def delete_voice(voice_id):
    # Wait until generation has finished reading the selected sample.
    with lock, voice_lock:
        voice_metadata(voice_id)
        (VOICES / f'{voice_id}.wav').unlink(missing_ok=True)
        (VOICES / f'{voice_id}.json').unlink()

def save_voice(name, data):
    name = name.strip()
    if not name or len(name) > 100:
        raise HTTPException(422, 'Name must contain 1 to 100 characters')
    if len(data) > 25 * 1024**2:
        raise HTTPException(413, 'Sample must be smaller than 25 MB')
    try:
        source = io.BytesIO(data)
        info = sf.info(source)
        if not 1 <= info.duration <= 120 or not 8000 <= info.samplerate <= 192000 or not 1 <= info.channels <= 8:
            raise ValueError('Sample must be 1-120 seconds, 8-192 kHz, and have 1-8 channels')
        source.seek(0)
        audio, sr = sf.read(source, dtype='float32', always_2d=True)
        audio = audio.mean(axis=1)
        if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 0.0001:
            raise ValueError('Sample contains no usable audio')
    except Exception as exc:
        raise HTTPException(422, f'Invalid audio sample: {exc}') from exc
    voice_id = uuid.uuid4().hex
    metadata = {'id': voice_id, 'name': name, 'created_at': datetime.now(timezone.utc).isoformat(),
                'duration_seconds': round(len(audio) / sr, 3), 'sample_rate': sr,
                'sample_url': f'/voices/{voice_id}/sample'}
    with voice_lock:
        sf.write(VOICES / f'{voice_id}.wav', audio, sr, subtype='PCM_16')
        (VOICES / f'{voice_id}.json').write_text(json.dumps(metadata, ensure_ascii=False), encoding='utf-8')
    return metadata

def synthesize(text, language='en', reference=None, voice=None, exaggeration=0.5, cfg_weight=0.5):
    global last_used
    if not text.strip() or len(text) > 3000:
        raise HTTPException(422, 'Text must contain 1 to 3000 characters')
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(422, 'Unsupported language')
    with lock:
        if voice:
            voice_metadata(voice)
            reference = str(VOICES / f'{voice}.wav')
        load_locked()
        # References must not leak into subsequent default-voice calls.
        model.conds = default_conds
        try:
            with torch.inference_mode():
                wav = model.generate(text, language_id=language, audio_prompt_path=reference or None,
                                     exaggeration=exaggeration, cfg_weight=cfg_weight)
            return model.sr, wav.squeeze(0).cpu().numpy()
        finally:
            model.conds = default_conds
            last_used = time.monotonic()

api = FastAPI(title='Chatterbox REST API', version='1.0.0', lifespan=lifespan)

class ModelSettings(BaseModel):
    idle_timeout_seconds: int = Field(default=7200, ge=0, le=2592000, strict=True)

@api.get('/model/settings')
def model_settings():
    return {'idle_timeout_seconds': idle_timeout_seconds}

@api.put('/model/settings')
def update_model_settings(settings: ModelSettings):
    global idle_timeout_seconds
    with lock:
        temporary = SETTINGS_FILE.with_suffix('.tmp')
        temporary.write_text(json.dumps(settings.model_dump()), encoding='utf-8')
        temporary.replace(SETTINGS_FILE)
        idle_timeout_seconds = settings.idle_timeout_seconds
        return model_settings()

def ui_timeout(minutes):
    seconds = minutes * 60
    if not 0 <= seconds <= 2592000 or seconds != int(seconds):
        raise gr.Error('Enter minutes between 0 and 43200 (0 disables automatic unloading)')
    return update_model_settings(ModelSettings(idle_timeout_seconds=int(seconds)))

class SpeechRequest(BaseModel):
    input: str = Field(validation_alias=AliasChoices('input', 'text'), min_length=1, max_length=3000)
    language: str = 'en'
    voice: str | None = None
    exaggeration: float = Field(default=0.5, ge=0, le=2)
    cfg_weight: float = Field(default=0.5, ge=0, le=1)
    response_format: str = 'wav'

@api.get('/model/status')
def model_status():
    return status()

@api.post('/model/load')
def model_load():
    with lock:
        load_locked()
        return status()

@api.post('/model/unload')
def model_unload():
    with lock:
        unload_locked()
        return status()

@api.get('/voices')
def voices_list():
    return {'voices': list_voices()}

@api.post('/voices', status_code=201)
def voices_add(name: str = Form(...), sample: UploadFile = File(...)):
    try:
        return save_voice(name, sample.file.read(25 * 1024**2 + 1))
    finally:
        sample.file.close()

@api.get('/voices/{voice_id}')
def voices_get(voice_id: str):
    with voice_lock:
        return voice_metadata(voice_id)

@api.delete('/voices/{voice_id}', status_code=204)
def voices_delete(voice_id: str):
    delete_voice(voice_id)
    return Response(status_code=204)

@api.get('/voices/{voice_id}/sample')
def voices_sample(voice_id: str):
    with voice_lock:
        voice_metadata(voice_id)
        data = (VOICES / f'{voice_id}.wav').read_bytes()
    return Response(data, media_type='audio/wav', headers={'Content-Disposition': f'attachment; filename="{voice_id}.wav"'})

@api.post('/v1/audio/speech', responses={200: {'content': {'audio/wav': {}}}})
def speech(request: SpeechRequest):
    if request.response_format != 'wav':
        raise HTTPException(422, 'Only WAV output is supported')
    sr, audio = synthesize(request.input, request.language, voice=request.voice,
                           exaggeration=request.exaggeration, cfg_weight=request.cfg_weight)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format='WAV', subtype='PCM_16')
    return Response(buffer.getvalue(), media_type='audio/wav', headers={'Content-Disposition': 'attachment; filename="speech.wav"'})

def ui_generate(text, language, reference, voice=''):
    try:
        return synthesize(text, language, reference, voice)
    except HTTPException as exc:
        raise gr.Error(exc.detail) from exc

def voice_choices():
    return [('Built-in / uploaded reference', '')] + [(v['name'], v['id']) for v in list_voices()]

def ui_save_voice(name, reference):
    if not reference:
        raise gr.Error('Upload a sample first')
    try:
        with open(reference, 'rb') as f:
            saved = save_voice(name, f.read(25 * 1024**2 + 1))
    except HTTPException as exc:
        raise gr.Error(exc.detail) from exc
    return gr.Dropdown(choices=voice_choices(), value=saved['id'])

def ui_delete_voice(voice_id):
    if not voice_id:
        raise gr.Error('Select a stored voice to delete')
    try:
        delete_voice(voice_id)
    except HTTPException as exc:
        raise gr.Error(exc.detail) from exc
    gr.Info('Stored voice deleted')
    return gr.Dropdown(choices=voice_choices(), value='')

def ui_request_delete(voice_id):
    if not voice_id:
        raise gr.Error('Select a stored voice to delete')
    try:
        metadata = voice_metadata(voice_id)
    except HTTPException as exc:
        raise gr.Error(exc.detail) from exc
    return gr.Group(visible=True), f"{metadata['name']} ({voice_id})", voice_id

def ui_confirm_delete(pending_id):
    selection = ui_delete_voice(pending_id)
    return selection, gr.Group(visible=False), None

with gr.Blocks(title='Chatterbox TTS') as ui:
    gr.Markdown('# Chatterbox TTS\nGenerate speech locally. [REST API documentation](/docs)')
    text = gr.Textbox(label='Text', lines=5, value='Hello! Chatterbox is running locally on this computer.')
    language = gr.Dropdown(choices=[(name, code) for code, name in SUPPORTED_LANGUAGES.items()], value='en', label='Language')
    voice = gr.Dropdown(choices=voice_choices(), value='', label='Stored voice (takes priority over uploaded reference)')
    voice_id = gr.Textbox(value='', label='Voice ID', interactive=False, buttons=['copy'])
    refresh = gr.Button('Refresh stored voices')
    delete = gr.Button('Delete selected stored voice', variant='stop')
    pending_delete = gr.State(None)
    with gr.Group(visible=False) as confirmation:
        gr.Markdown('Permanently delete this stored voice and its audio sample?')
        deletion_target = gr.Textbox(label='Voice to delete', interactive=False)
        confirm_delete = gr.Button('Confirm deletion', variant='stop')
        cancel_delete = gr.Button('Cancel')
    delete.click(ui_request_delete, voice, [confirmation, deletion_target, pending_delete], api_name=False)
    confirm_delete.click(ui_confirm_delete, pending_delete, [voice, confirmation, pending_delete], api_name=False)
    cancel_delete.click(lambda: (gr.Group(visible=False), None), outputs=[confirmation, pending_delete], api_name=False, queue=False)
    voice.change(lambda selected: (selected or '', gr.Group(visible=False), None), voice, [voice_id, confirmation, pending_delete], api_name=False, queue=False)
    reference = gr.Audio(sources=['upload', 'microphone'], type='filepath', label='Voice reference (optional)')
    name = gr.Textbox(label='Name for saved voice')
    save = gr.Button('Save reference as stored voice')
    save.click(ui_save_voice, [name, reference], voice, api_name=False)
    refresh.click(lambda: gr.Dropdown(choices=voice_choices(), value=''), outputs=voice, api_name=False)
    generate_button = gr.Button('Generate speech', variant='primary')
    audio = gr.Audio(label='Generated speech')
    generate_button.click(ui_generate, [text, language, reference, voice], audio, api_name='synthesize', concurrency_limit=1)
    with gr.Accordion('Model memory', open=False):
        model_info = gr.JSON(label='Model status')
        timeout = gr.Number(value=idle_timeout_seconds / 60, label='Auto-unload after idle minutes (0 disables)', minimum=0, maximum=43200)
        gr.Button('Save idle timeout').click(ui_timeout, timeout, model_info, api_name=False)
        ui.load(lambda: idle_timeout_seconds / 60, outputs=timeout, api_name=False)
        gr.Button('Status').click(model_status, outputs=model_info, api_name=False)
        gr.Button('Load model').click(model_load, outputs=model_info, api_name=False)
        gr.Button('Unload model').click(model_unload, outputs=model_info, api_name=False)

app = gr.mount_gradio_app(api, ui.queue(), path='/', server_name='127.0.0.1', server_port=7860)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=7860)
