"""Verified parallel ranges for large public weights; no credentials or remote code."""
import concurrent.futures
import hashlib
import shutil
import time
import urllib.request


def download_ranges(url, target, size, sha, workers=8):
    if target.exists():
        with target.open('rb') as f:
            if target.stat().st_size == size and hashlib.file_digest(f, 'sha256').hexdigest() == sha:
                return
        raise ValueError('Existing model checksum mismatch')
    target.parent.mkdir(parents=True, exist_ok=True)
    block = 64 * 1024 * 1024
    parts = [target.with_name(target.name + f'.range-{i}') for i in range((size + block - 1) // block)]
    def fetch(i):
        start, end = i * block, min(size, (i + 1) * block) - 1
        path = parts[i]
        if path.exists() and path.stat().st_size == end - start + 1:
            return
        for attempt in range(5):
            try:
                request = urllib.request.Request(url + f'?part={i}', headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(request, timeout=90) as r:
                    if r.status != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                        raise ValueError('Invalid range response')
                    with path.open('wb') as f:
                        shutil.copyfileobj(r, f, 1024 * 1024)
                if path.stat().st_size != end - start + 1:
                    raise ValueError('Incomplete range')
                return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for n, task in enumerate(concurrent.futures.as_completed([pool.submit(fetch, i) for i in range(len(parts))]), 1):
            task.result()
            print(target.name, f'{n}/{len(parts)} ranges', flush=True)
    tmp = target.with_name(target.name + '.assembled')
    digest = hashlib.sha256()
    with tmp.open('wb') as f:
        for part in parts:
            with part.open('rb') as p:
                while data := p.read(4 * 1024 * 1024):
                    f.write(data)
                    digest.update(data)
    if tmp.stat().st_size != size or digest.hexdigest() != sha:
        raise ValueError('Assembled checksum mismatch')
    tmp.replace(target)
    for part in parts:
        part.unlink()
    old = target.with_name(target.name + '.part')
    if old.exists():
        old.unlink()
