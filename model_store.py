"""Pinned, resumable model downloads; verification receipts never replace hashes."""
from pathlib import Path
import hashlib
import json
import subprocess
import time
import os
import threading
import argparse

APP = Path(__file__).resolve().parent
MODEL_DIR = APP / 'models'
MANIFEST = json.loads((APP / 'models.json').read_text(encoding='utf-8'))
VARIANTS = MANIFEST['variants']
RECEIPT_LOCK = threading.RLock()
SETTINGS = APP / 'model-settings.json'


def selected_model():
    try:
        value = json.loads(SETTINGS.read_text(encoding='utf-8')).get('model', 'int8')
    except (OSError, ValueError):
        value = 'int8'
    return value if value in VARIANTS else 'int8'


def select_model(variant):
    if variant not in VARIANTS:
        raise ValueError('Choose the smaller INT8 or larger BF16 model.')
    with RECEIPT_LOCK:
        stage = SETTINGS.with_name('model-settings-' + str(time.time_ns()) + '.tmp')
        stage.write_text(json.dumps({'model': variant}, indent=2), encoding='utf-8')
        os.replace(stage, SETTINGS)


def model_files(variant=None):
    variant = variant or selected_model()
    if variant not in VARIANTS:
        raise ValueError('Unknown model selection.')
    image_path = VARIANTS[variant]['path']
    entries = MANIFEST['files'] + MANIFEST.get('optional_files', [])
    return [item for item in entries if not item['path'].startswith('diffusion_models/')
            or item['path'] == image_path]


def model_names(variant=None):
    entries = model_files(variant)
    return {item['path'].split('/')[0]: Path(item['path']).name for item in entries}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(full=False, variant=None):
    with RECEIPT_LOCK:
        return _verify(full, variant)


def _verify(full=False, variant=None):
    receipt_path = MODEL_DIR / 'verified.json'
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, ValueError):
        receipt = {}
    stamps = dict(receipt)
    for entry in model_files(variant):
        path = MODEL_DIR / entry['path']
        if not path.is_file() or path.stat().st_size != entry['size']:
            return False, 'Missing or incomplete: ' + entry['path']
        stamp = [path.stat().st_size, path.stat().st_mtime_ns, entry['sha256']]
        if full or receipt.get(entry['path']) != stamp:
            if digest(path) != entry['sha256']:
                return False, 'SHA-256 mismatch: ' + entry['path']
        stamps[entry['path']] = stamp
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if stamps != receipt:
        stage = receipt_path.with_name('verified-' + str(time.time_ns()) + '.tmp')
        stage.write_text(json.dumps(stamps, indent=2))
        os.replace(stage, receipt_path)
    return True, 'Installed — selected image model, text encoder and VAE verified.'


def download(variant=None, progress=None, cancel=None):
    variant = variant or selected_model()
    notify = progress or (lambda message: print(message, flush=True))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    entries = model_files(variant)
    for entry in entries:
        if cancel is not None and cancel.is_set():
            raise InterruptedError('Download stopped. Partial files are kept; click Install to resume.')
        target = MODEL_DIR / entry['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            notify('Verifying ' + target.name)
            if target.stat().st_size == entry['size'] and digest(target) == entry['sha256']:
                continue
            # Preserve an unexpected file; never delete customer data.
            target.rename(target.with_name(target.name + f'.unverified-{time.time_ns()}'))
        partial = target.with_name(target.name + '.partial')
        if partial.is_file() and partial.stat().st_size >= entry['size']:
            if partial.stat().st_size == entry['size'] and digest(partial) == entry['sha256']:
                partial.rename(target)
                continue
            partial.rename(partial.with_name(partial.name + f'.unverified-{time.time_ns()}'))
        revision = entry.get('revision', MANIFEST['revision'])
        url = f"https://huggingface.co/{MANIFEST['repo']}/resolve/{revision}/{entry['path']}"
        notify('Downloading / resuming ' + target.name)
        # Curl's errors are retained on disk, never buffered into an unread PIPE.
        with partial.with_suffix('.download.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen(['curl.exe', '--fail', '--location', '--silent', '--show-error',
                '--retry', '5', '--retry-delay', '3', '--connect-timeout', '30',
                '--speed-limit', '1024', '--speed-time', '60', '--continue-at', '-',
                '--output', str(partial), url], stdout=log, stderr=log,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            try:
                while process.poll() is None:
                    if cancel is not None and cancel.is_set():
                        raise InterruptedError('Download stopped. Partial files are kept; click Install to resume.')
                    size = partial.stat().st_size if partial.exists() else 0
                    notify(f'Downloading {target.name}: {size / 1e9:.2f} / {entry["size"] / 1e9:.2f} GB')
                    time.sleep(0.5)
                if process.returncode:
                    raise RuntimeError(f'Download failed for {target.name}. Check your connection and retry to resume.')
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
        notify('Checking SHA-256: ' + target.name)
        if partial.stat().st_size != entry['size'] or digest(partial) != entry['sha256']:
            raise RuntimeError('Download did not pass size/SHA-256 verification: ' + target.name)
        partial.rename(target)
    ready, note = verify(full=True, variant=variant)
    notify(note)
    if not ready:
        raise RuntimeError(note)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=list(VARIANTS), default=None)
    args = parser.parse_args()
    download(args.variant)
