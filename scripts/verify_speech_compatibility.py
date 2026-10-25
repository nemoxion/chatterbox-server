"""Real local speech, codecs, forced alignment and OpenAI SDK compatibility checks."""
import base64
import io
import sys
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from openai import OpenAI, BadRequestError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from speech_tools import decoded_audio

BASE = 'http://127.0.0.1:7860'
MODEL = 'chatterbox-multilingual-v3'

if __name__ == '__main__':
    client = OpenAI(base_url=BASE + '/v1', api_key='local', timeout=180)
    assert client.models.list().data[0].id == MODEL
    assert client.models.retrieve(MODEL).id == MODEL
    try:
        client.audio.speech.create(model=MODEL, voice='default', input='Hello', instructions='Whisper softly')
    except BadRequestError as error:
        assert error.status_code == 400 and error.body['type'] == 'invalid_request_error'
    else:
        raise AssertionError('Unsupported instructions were silently accepted')
    for payload in [{'input': ''}, {'input': 'Hi', 'speed': 0.1}, {'input': 'Hi', 'response_format': 'bogus'},
                    {'input': 'Hi', 'stream_format': 'sse'}, {'input': 'Hi', 'extra_unknown': True}]:
        response = requests.post(BASE + '/v1/audio/speech', json=payload, timeout=10)
        assert response.status_code == 400 and response.json()['error']['type'] == 'invalid_request_error'
    for output in ['wav', 'mp3', 'opus', 'aac', 'flac', 'pcm']:
        response = client.audio.speech.create(model=MODEL, voice='default', input='Hello world. This is an audio test.', response_format=output, speed=1.25)
        audio = decoded_audio(response.content, output, 24000)
        assert len(audio) > 24000 and np.isfinite(audio).all() and np.max(np.abs(audio)) > 0.01
        print('SDK audio format:', output, 'seconds:', round(len(audio) / 24000, 3), flush=True)
    default = client.audio.speech.create(model=MODEL, voice='default', input='The default format is MP3.')
    assert default.response.headers['content-type'].startswith('audio/mpeg')
    voices = requests.get(BASE + '/voices', timeout=10).json()['voices']
    if voices:
        client.audio.speech.create(model=MODEL, voice={'id': voices[0]['id']}, input='A saved voice object works.', response_format='wav')
    for language, text, output in [('en', 'Hello world. This is an alignment test.', 'mp3'),
                                   ('de', 'Hallo Welt. Das ist ein Test.', 'wav')]:
        response = requests.post(BASE + '/v1/audio/speech/with-timestamps', json={
            'model': MODEL, 'voice': 'default', 'input': text, 'language': language,
            'speed': 1.3, 'response_format': output}, timeout=240)
        response.raise_for_status()
        result = response.json()
        assert [word['word'] for word in result['words']] == text.split()
        audio = decoded_audio(base64.b64decode(result['audio_base64']), output, 24000)
        assert abs(len(audio) / 24000 - result['duration']) < 0.001
        previous = 0
        for word in result['words']:
            assert word['aligned'] and previous <= word['start'] < word['end'] <= result['duration']
            assert 0 <= word['confidence'] <= 1
            previous = word['end']
        print('Forced word alignment:', language, result['words'], flush=True)
    requests.post(BASE + '/model/unload', timeout=60).raise_for_status()
    print('Speech compatibility and measured alignment checks passed.', flush=True)
