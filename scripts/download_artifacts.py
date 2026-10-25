"""Download pinned CUDA wheels or model files with SHA256 verification."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def download(item, workers=8):
    path = ROOT / item['path']
    if path.exists() and path.stat().st_size == item['size'] and checksum(path) == item['sha256']:
        print('Verified existing', path.name, flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.part')
    size = item['size']
    chunk = 8 * 1024 * 1024
    with partial.open('wb') as stream:
        stream.truncate(size)

    def fetch(start):
        end = min(start + chunk, size) - 1
        for attempt in range(5):
            try:
                request = urllib.request.Request(item['url'], headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    expected_range = f'bytes {start}-{end}/{size}'
                    if response.headers.get('Content-Range') != expected_range:
                        raise ValueError('Server returned an unexpected byte range')
                    data = response.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise ValueError('Incomplete download chunk')
                with partial.open('r+b') as stream:
                    stream.seek(start)
                    stream.write(data)
                return len(data)
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        received = 0
        for count in pool.map(fetch, range(0, size, chunk)):
            received += count
            if received % (chunk * 20) == 0 or received == size:
                print(path.name, f'{received / size:.0%}', flush=True)
    if checksum(partial) != item['sha256']:
        raise ValueError(f'SHA256 mismatch: {path.name}')
    partial.replace(path)
    print('Verified', path.name, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=['cuda', 'models', 'alignment', 'all'], default='all')
    parser.add_argument('--workers', type=int, choices=range(1, 33), default=8)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'scripts/artifacts.json').read_text())
    manifest += json.loads((ROOT / 'scripts/alignment-artifacts.json').read_text())
    for artifact in manifest:
        if args.group in ('all', artifact['group']):
            download(artifact, args.workers)
