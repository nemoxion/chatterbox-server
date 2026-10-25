"""Check idle expiration, active requests, disabling, and restore the previous setting."""
import io
import time
import requests
import soundfile as sf

BASE = 'http://127.0.0.1:7860'

def request(method, path, **kwargs):
    response = requests.request(method, BASE + path, timeout=180, **kwargs)
    response.raise_for_status()
    return response

def configure(seconds):
    return request('PUT', '/model/settings', json={'idle_timeout_seconds': seconds}).json()

if __name__ == '__main__':
    original = request('GET', '/model/settings').json()['idle_timeout_seconds']
    try:
        for invalid in [-1, 1.5, 2592001]:
            assert requests.put(BASE + '/model/settings', json={'idle_timeout_seconds': invalid}, timeout=10).status_code == 422
        configure(2)
        request('POST', '/model/load')
        # Generation lasts longer than the timeout but must still finish normally.
        audio = request('POST', '/v1/audio/speech', json={'input': 'This request must finish before the idle timer starts counting down.', 'response_format': 'wav'})
        wav, sr = sf.read(io.BytesIO(audio.content))
        assert len(wav) > sr
        assert request('GET', '/model/status').json()['loaded']
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if not request('GET', '/model/status').json()['loaded']:
                break
            time.sleep(0.5)
        else:
            raise AssertionError('Model did not unload after idle timeout')
        # Status polling must not keep the model loaded; server remains responsive.
        configure(0)
        request('POST', '/model/load')
        time.sleep(3)
        assert request('GET', '/model/status').json()['loaded']
        print('Idle unload, active generation protection, status polling and disabled timeout passed.')
    finally:
        configure(original)
        request('POST', '/model/unload')
        print('Original setting restored:', original)
