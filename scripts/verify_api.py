"""Integration check against a running server; retains one synthetic test voice."""
import argparse
import io
import numpy as np
import requests
import soundfile as sf

def verify(base):
    def call(method, path, expected=200, **kwargs):
        response = requests.request(method, base.rstrip('/') + path, timeout=180, **kwargs)
        if response.status_code != expected:
            raise AssertionError((path, response.status_code, response.text[:1000]))
        return response

    for path in ['/', '/docs', '/openapi.json', '/model/status']:
        call('GET', path)
    call('POST', '/v1/audio/speech', expected=422, json={'input': '   '})
    call('POST', '/v1/audio/speech', expected=404, json={'input': 'Hello', 'voice': 'missing'})
    call('POST', '/voices', expected=422, data={'name': 'Invalid'}, files={'sample': ('bad.wav', b'bad')})
    loaded = call('POST', '/model/load').json()
    assert loaded['loaded']
    assert call('POST', '/model/load').json()['loaded']
    speech = call('POST', '/v1/audio/speech', json={'input': 'Hello. This is a local REST interface test.'})
    audio, sr = sf.read(io.BytesIO(speech.content))
    assert len(audio) > sr and np.isfinite(audio).all() and np.max(np.abs(audio)) > 0.001
    temporary = call('POST', '/voices', expected=201, data={'name': 'Temporary deletion check'},
                     files={'sample': ('test.wav', speech.content, 'audio/wav')}).json()
    temporary_id = temporary['id']
    call('GET', f'/voices/{temporary_id}/sample')
    assert not call('DELETE', f'/voices/{temporary_id}', expected=204).content
    for path in [f'/voices/{temporary_id}', f'/voices/{temporary_id}/sample']:
        call('GET', path, expected=404)
    call('DELETE', f'/voices/{temporary_id}', expected=404)
    call('DELETE', '/voices/invalid-id', expected=404)
    assert all(v['id'] != temporary_id for v in call('GET', '/voices').json()['voices'])
    voices = call('GET', '/voices').json()['voices']
    voice = next((v for v in voices if v['name'] == 'Demo - synthetic test voice'), None)
    if voice is None:
        voice = call('POST', '/voices', expected=201, data={'name': 'Demo - synthetic test voice'},
                     files={'sample': ('test.wav', speech.content, 'audio/wav')}).json()
    voice_id = voice['id']
    assert call('GET', f'/voices/{voice_id}').json()['id'] == voice_id
    assert any(v['id'] == voice_id for v in call('GET', '/voices').json()['voices'])
    sf.read(io.BytesIO(call('GET', f'/voices/{voice_id}/sample').content))
    cloned = call('POST', '/v1/audio/speech', json={'text': 'Stored voice selection works.', 'voice': voice_id})
    audio, sr = sf.read(io.BytesIO(cloned.content))
    assert len(audio) > sr and np.isfinite(audio).all()
    unloaded = call('POST', '/model/unload').json()
    assert not unloaded['loaded']
    assert unloaded['gpu_allocated_mb'] < loaded['gpu_allocated_mb'] * 0.05
    assert not call('POST', '/model/unload').json()['loaded']
    print('REST integration checks passed.')
    print('Unloaded:', unloaded)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:7860')
    verify(parser.parse_args().base_url)
