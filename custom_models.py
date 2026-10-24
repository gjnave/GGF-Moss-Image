"""PC-local, non-copying model overrides. Only safetensors are accepted."""
from pathlib import Path
import json
import os
import subprocess
import sys
import threading
import time
from model_store import APP, MODEL_DIR, RECEIPT_LOCK, model_files, model_names, verify_entries, digest

SETTINGS = APP / 'custom-models.json'
CATEGORIES = ('diffusion_models', 'text_encoders', 'vae', 'lora')
CACHE = {}
CACHE_LOCK = threading.RLock()


def load():
    try:
        data = json.loads(SETTINGS.read_text(encoding='utf-8'))
        return {k: str(data.get(k, '')) for k in CATEGORIES}
    except (OSError, ValueError, TypeError):
        return dict.fromkeys(CATEGORIES, '')


def check(value, category):
    """Check container integrity and the expected architecture without GPU loading."""
    if not value or not str(value).strip():
        return '', 'Default'
    try:
        path = Path(str(value).strip().strip('"')).resolve(strict=True)
        if not path.is_file() or path.suffix.lower() != '.safetensors':
            raise ValueError('choose a .safetensors file')
        stat = path.stat()
        stamp = (str(path), stat.st_size, stat.st_mtime_ns, category)
        with CACHE_LOCK:
            if stamp in CACHE:
                return str(path), 'Custom model ready'
        from safetensors import safe_open
        with safe_open(str(path), framework='pt', device='cpu') as tensors:
            keys = set(tensors.keys())
            if category == 'diffusion_models':
                valid = any(k.endswith('video_embedder.weight') for k in keys) and any('layers.41.proj.' in k for k in keys)
            elif category == 'text_encoders':
                valid = any(k.endswith('model.language_model.embed_tokens.weight') for k in keys) and any('language_model.layers.0.' in k for k in keys)
            elif category == 'vae':
                valid = {'bn.running_mean','bn.running_var','decoder.conv_in.weight','encoder.conv_in.weight'} <= keys
            elif category == 'lora':
                valid = any('layers.' in k and '.proj.' in k and ('lora_down' in k or 'hada_w1_a' in k) for k in keys)
            else:
                valid = False
            if not valid:
                raise ValueError('not a supported MOSS / Qwen3.5 / FLUX.2 file for this slot')
        with CACHE_LOCK:
            CACHE[stamp] = True
        return str(path), 'Custom model ready'
    except Exception as error:
        return '', 'Default used: ' + str(error)


def save(*values):
    data = dict(zip(CATEGORIES, values))
    notes = []
    for category in CATEGORIES:
        value = str(data.get(category) or '').strip().strip('"')
        data[category] = value
        _, note = check(value, category)
        notes.append(category + ': ' + note)
    stage = SETTINGS.with_name('custom-models-' + str(time.time_ns()) + '.tmp')
    stage.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(stage, SETTINGS)
    return '\n\n'.join(notes) + '\n\nSaved. Applies to the next generation; no restart needed.'


def resolve(variant=None, mode='standard', settings=None):
    from model_store import selected_model
    variant = 'bf16' if mode == 'fast4' else variant or selected_model()
    data = load() if settings is None else settings
    custom, notes = {}, []
    for category in CATEGORIES:
        path, note = check(data.get(category, ''), category)
        if path and mode == 'fast4' and category == 'diffusion_models':
            # Fast's adapter is tied to this exact SFT base, not arbitrary MOSS LoRAs/checkpoints.
            entry = next(x for x in model_files('bf16') if x['path'].startswith('diffusion_models/'))
            stat = Path(path).stat()
            key = ('fast-base',path,stat.st_size,stat.st_mtime_ns)
            with CACHE_LOCK:
                approved = CACHE.get(key)
            if approved is None:
                approved = stat.st_size == entry['size'] and digest(Path(path)) == entry['sha256']
                with CACHE_LOCK:
                    CACHE[key] = approved
            if not approved:
                path, note = '', 'Fast uses the verified BF16 SFT base; custom image model applies only in Standard.'
        if path and mode == 'fast4' and category == 'lora':
            path, note = '', 'Fast uses its own acceleration adapter; custom LoRA applies only in Standard.'
        if path:
            custom[category] = path
        if data.get(category):
            notes.append(category + ': ' + note)
    return custom, notes


def ready(variant=None, mode='standard', overrides=None):
    custom, _ = resolve(variant, mode) if overrides is None else (overrides, [])
    entries = [x for x in model_files('bf16' if mode == 'fast4' else variant)
               if x['path'].split('/')[0] not in custom]
    if mode == 'fast4':
        from fast_mode import ADAPTER
        entries.append(ADAPTER)
    with RECEIPT_LOCK:
        return verify_entries(entries)


def register(folder_paths, variant, custom):
    """Each category has one custom parent; prepend avoids basename collisions."""
    names = model_names(variant)
    for category in ('diffusion_models', 'text_encoders', 'vae'):
        if category in custom:
            path = Path(custom[category])
            folder_paths.add_model_folder_path(category, str(path.parent), is_default=True)
            names[category] = path.name
    return names


def browse():
    """Open Windows' native PC file picker in an isolated subprocess, not a web upload."""
    completed = subprocess.run([sys.executable, str(Path(__file__)), '--browse'],
        capture_output=True, text=True, encoding='utf-8', timeout=300,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    return completed.stdout.strip()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    import ctypes
    from ctypes import wintypes
    class OPENFILENAMEW(ctypes.Structure):
        _fields_ = [('lStructSize',wintypes.DWORD),('hwndOwner',wintypes.HWND),('hInstance',wintypes.HINSTANCE),
            ('lpstrFilter',wintypes.LPCWSTR),('lpstrCustomFilter',wintypes.LPWSTR),('nMaxCustFilter',wintypes.DWORD),
            ('nFilterIndex',wintypes.DWORD),('lpstrFile',wintypes.LPWSTR),('nMaxFile',wintypes.DWORD),
            ('lpstrFileTitle',wintypes.LPWSTR),('nMaxFileTitle',wintypes.DWORD),('lpstrInitialDir',wintypes.LPCWSTR),
            ('lpstrTitle',wintypes.LPCWSTR),('Flags',wintypes.DWORD),('nFileOffset',wintypes.WORD),
            ('nFileExtension',wintypes.WORD),('lpstrDefExt',wintypes.LPCWSTR),('lCustData',ctypes.c_ssize_t),
            ('lpfnHook',ctypes.c_void_p),('lpTemplateName',wintypes.LPCWSTR),('pvReserved',ctypes.c_void_p),
            ('dwReserved',wintypes.DWORD),('FlagsEx',wintypes.DWORD)]
    buffer = ctypes.create_unicode_buffer(32768)
    dialog = OPENFILENAMEW()
    dialog.lStructSize = ctypes.sizeof(dialog)
    dialog.lpstrFile = ctypes.cast(buffer, wintypes.LPWSTR)
    dialog.nMaxFile = len(buffer)
    dialog.lpstrFilter = 'Safe model files (*.safetensors)\0*.safetensors\0\0'
    dialog.lpstrTitle = 'Choose a model on this PC'
    dialog.Flags = 0x1000 | 0x800 | 0x8  # FILEMUSTEXIST, PATHMUSTEXIST, NOCHANGEDIR
    if ctypes.windll.comdlg32.GetOpenFileNameW(ctypes.byref(dialog)):
        print(buffer.value)
