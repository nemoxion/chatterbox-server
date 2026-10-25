"""Audio formats, speed adjustment and measured CTC word alignment."""
import io
import re
import shutil
import subprocess
import threading
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import HTTPException
from num2words import num2words
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

ALIGNMENT_LANGUAGES = {'en', 'de'}
alignment_lock = threading.Lock()
FORMATS = {'wav': ('audio/wav', []), 'mp3': ('audio/mpeg', ['-c:a', 'libmp3lame', '-b:a', '128k']),
           'flac': ('audio/flac', ['-c:a', 'flac']), 'opus': ('audio/ogg', ['-c:a', 'libopus', '-b:a', '96k']),
           'aac': ('audio/aac', ['-c:a', 'aac', '-b:a', '128k']), 'pcm': ('audio/pcm', [])}

def wav_bytes(audio, sr):
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format='WAV', subtype='PCM_16')
    return buffer.getvalue()

def ffmpeg(args, data):
    executable = shutil.which('ffmpeg')
    if not executable:
        raise HTTPException(503, 'FFmpeg is required for speed adjustment and compressed formats')
    try:
        result = subprocess.run([executable, '-hide_banner', '-loglevel', 'error', *args], input=data,
                                capture_output=True, timeout=120, creationflags=subprocess.CREATE_NO_WINDOW)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(503, 'Audio conversion timed out') from exc
    if result.returncode:
        raise HTTPException(500, 'Audio conversion failed')
    return result.stdout

def adjust_speed(audio, sr, speed):
    if speed == 1:
        return audio
    # Chain atempo stages to preserve quality outside the 0.5-2 range.
    factors = []
    remaining = speed
    while remaining < 0.5:
        factors.append(0.5); remaining /= 0.5
    while remaining > 2:
        factors.append(2); remaining /= 2
    factors.append(remaining)
    result = ffmpeg(['-i', 'pipe:0', '-af', ','.join(f'atempo={v}' for v in factors),
                     '-f', 'f32le', '-ar', str(sr), '-ac', '1', 'pipe:1'], wav_bytes(audio, sr))
    return np.frombuffer(result, dtype='<f4').copy()

def encode_audio(audio, sr, output_format):
    mime, codec = FORMATS[output_format]
    if output_format == 'wav':
        return wav_bytes(audio, sr), mime
    if output_format == 'flac':
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format='FLAC', subtype='PCM_16')
        return buffer.getvalue(), mime
    if output_format == 'pcm':
        return (np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes(), mime
    container = {'opus': 'ogg', 'aac': 'adts'}.get(output_format, output_format)
    return ffmpeg(['-i', 'pipe:0', *codec, '-f', container, 'pipe:1'], wav_bytes(audio, sr)), mime

def decoded_audio(encoded, output_format, sr):
    if output_format in {'wav', 'flac'}:
        return sf.read(io.BytesIO(encoded), dtype='float32')[0]
    if output_format == 'pcm':
        return np.frombuffer(encoded, dtype='<i2').astype('float32') / 32768
    raw = ffmpeg(['-i', 'pipe:0', '-f', 'f32le', '-ar', str(sr), '-ac', '1', 'pipe:1'], encoded)
    return np.frombuffer(raw, dtype='<f4').copy()

def check_alignment(language, root):
    if language not in ALIGNMENT_LANGUAGES:
        raise HTTPException(422, 'Word alignment currently supports en and de; ordinary TTS supports all model languages')
    path = root / 'data' / 'alignment' / language
    if not (path / 'config.json').is_file() or not any(path.glob('*.safetensors')) and not (path / 'pytorch_model.bin').is_file():
        raise HTTPException(503, 'Alignment model missing. Run scripts/download_artifacts.py --group alignment')
    return path

def align_words(audio, sr, text, language, root):
    path = check_alignment(language, root)
    # CPU alignment avoids changing TTS VRAM usage; serialize to bound memory consumption.
    # Models are scoped to the request and released afterwards.
    with alignment_lock, torch.inference_mode():
        processor = Wav2Vec2Processor.from_pretrained(path, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(path, local_files_only=True).eval()
        vocab = processor.tokenizer.get_vocab()
        blank = processor.tokenizer.pad_token_id
        records, tokens, token_ranges = [], [], []
        for match in re.finditer(r'\S+', text):
            word = match.group()
            normalized = word.strip('.,!?;:()[]{}"“”„')
            if normalized.isdigit():
                normalized = num2words(int(normalized), lang=language)
            normalized = unicodedata.normalize('NFC', normalized)
            normalized = normalized.upper() if 'A' in vocab else normalized.lower()
            ids = [vocab[c] for c in normalized if c in vocab and c not in ['|', '<pad>', '<unk>']]
            if any(c.isalnum() and c not in vocab for c in normalized):
                ids = []
            records.append({'word': word, 'start': None, 'end': None, 'confidence': None,
                            'text_start': match.start(), 'text_end': match.end(), 'aligned': False})
            if ids and tokens and '|' in vocab:
                tokens.append(vocab['|'])
            token_ranges.append((len(tokens), len(tokens) + len(ids)))
            tokens.extend(ids)
        if not tokens:
            raise HTTPException(422, 'No alignable speech characters in input')
        waveform = torchaudio.functional.resample(torch.from_numpy(np.asarray(audio).copy()), sr, 16000)
        inputs = processor(waveform.numpy(), sampling_rate=16000, return_tensors='pt')
        emissions = model(**inputs).logits.log_softmax(-1)
        try:
            aligned, scores = torchaudio.functional.forced_align(emissions, torch.tensor([tokens]), blank=blank)
        except RuntimeError as exc:
            raise HTTPException(422, 'Audio is too short or does not support alignment to the requested text') from exc
        spans = torchaudio.functional.merge_tokens(aligned[0], scores[0].exp(), blank=blank)
        if len(spans) != len(tokens):
            raise HTTPException(422, 'Alignment could not match all transcript tokens')
        ratio = len(audio) / sr / emissions.shape[1]
        for record, (start, end) in zip(records, token_ranges):
            if end > start:
                selected = spans[start:end]
                record.update(start=round(selected[0].start * ratio, 4), end=round(selected[-1].end * ratio, 4),
                              confidence=round(sum(s.score for s in selected) / len(selected), 4), aligned=True)
        return records
