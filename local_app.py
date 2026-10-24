"""Local Chatterbox browser UI and REST API."""
import gc
import io
import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import AliasChoices, BaseModel, Field
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
from chatterbox.models.s3gen.s3gen import get_resampler
from chatterbox.models.s3gen.utils import mel as mel_utils
from s3tokenizer.utils import _mel_filters

ROOT = Path(__file__).resolve().parent
VOICES = ROOT / 'voices'
VOICES.mkdir(exist_ok=True)
model = None
default_conds = None
state = 'unloaded'
lock = threading.Lock()
voice_lock = threading.Lock()

def status():
    return {
        'model': 'chatterbox-multilingual-v3', 'state': state,
        'loaded': model is not None, 'device': 'cuda',
        'gpu': torch.cuda.get_device_name(0),
        'gpu_allocated_mb': round(torch.cuda.memory_allocated() / 1024**2, 1),
        'gpu_reserved_mb': round(torch.cuda.memory_reserved() / 1024**2, 1),
    }

def load_locked():
    global model, default_conds, state
    if model is None:
        state = 'loading'
        try:
            model = ChatterboxMultilingualTTS.from_local(ROOT / 'models', device='cuda', t3_model='v3')
            default_conds = model.conds
        finally:
            state = 'loaded' if model is not None else 'unloaded'

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
    if not text.strip() or len(text) > 3000:
        raise HTTPException(422, 'Text must contain 1 to 3000 characters')
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(422, 'Unsupported language')
    if voice:
        voice_metadata(voice)
        reference = str(VOICES / f'{voice}.wav')
    with lock:
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

api = FastAPI(title='Chatterbox REST API', version='1.0.0')

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
    global model, default_conds, state
    with lock:
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
        state = 'unloaded'
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
    return voice_metadata(voice_id)

@api.get('/voices/{voice_id}/sample')
def voices_sample(voice_id: str):
    voice_metadata(voice_id)
    return FileResponse(VOICES / f'{voice_id}.wav', media_type='audio/wav', filename=f'{voice_id}.wav')

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

with gr.Blocks(title='Chatterbox TTS') as ui:
    gr.Markdown('# Chatterbox TTS\nGenerate speech locally. [REST API documentation](/docs)')
    text = gr.Textbox(label='Text', lines=5, value='Hello! Chatterbox is running locally on this computer.')
    language = gr.Dropdown(choices=[(name, code) for code, name in SUPPORTED_LANGUAGES.items()], value='en', label='Language')
    voice = gr.Dropdown(choices=voice_choices(), value='', label='Stored voice (takes priority over uploaded reference)')
    refresh = gr.Button('Refresh stored voices')
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
        gr.Button('Status').click(model_status, outputs=model_info, api_name=False)
        gr.Button('Load model').click(model_load, outputs=model_info, api_name=False)
        gr.Button('Unload model').click(model_unload, outputs=model_info, api_name=False)

app = gr.mount_gradio_app(api, ui.queue(), path='/', server_name='127.0.0.1', server_port=7860)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=7860)
