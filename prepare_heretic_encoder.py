"""Default encoder: verified HF shards -> single Comfy-readable safetensors.

No remote Python code is executed, and no original model is replaced or deleted.
Source weights: pottokao/MOSS-Image-Text-Encoder-Heretic, Apache-2.0.
"""
from pathlib import Path
import json
import struct
import time
from model_store import MODEL_DIR, download_files, digest

REPO = 'pottokao/MOSS-Image-Text-Encoder-Heretic'
REVISION = 'b97159dc1ff68ccdde364ce5e1b18cdf86f792fe'
SHARDS = [
    (5276436216, '62be16bc8855331111b983a3bd1bb289e3bd4e3c9aeb052c3b5f53ec770b2da6'),
    (5335161512, '0b29d4615d05cf71d605c959a7bbf8860cd80d82f0560b9b309bab6522d23b83'),
    (5368717440, '5da40e83f0530d383593acae09682080f197c46976a34b00be5bcf0460ba81ff'),
    (3325995712, '7a778406726d024ec073696511bfa48c8f2a6ae2bba1505b2f3d1090fa207328'),
]
OUTPUT = MODEL_DIR / 'text_encoders/qwen3.5_9b_heretic_bf16.safetensors'
EXPECTED_SIZE = 19306312456
EXPECTED_SHA256 = '71a1bf315a5a1441aaf5e54ad7e2a694900eda7e56daa9f1aa5bafafb54505ff'


def canonical_header(raw):
    """Rust metadata map order varies; normalize it for reproducible SHA-256."""
    header=json.loads(raw)
    header['__metadata__']={'source_revision':REVISION,'license':'Apache-2.0','source_repo':REPO}
    encoded=json.dumps(header,separators=(',',':'),ensure_ascii=False).encode('utf-8')
    if len(encoded)>len(raw):
        raise RuntimeError('Unexpected safetensors header length.')
    return encoded.ljust(len(raw),b' ')


def main(progress=None, cancel=None):
    notify=progress or (lambda message: print(message,flush=True))
    def check_cancel():
        if cancel is not None and cancel.is_set():
            raise InterruptedError('Download / preparation stopped. Partial files are preserved; click Install to resume.')
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    if OUTPUT.is_file() and OUTPUT.stat().st_size==EXPECTED_SIZE and digest(OUTPUT)==EXPECTED_SHA256:
        notify('Heretic encoder already verified.')
        return
    if OUTPUT.exists():
        raise RuntimeError('Existing encoder did not pass verification; preserved for inspection.')
    entries=[]
    for index,(size,sha256) in enumerate(SHARDS,1):
        name=f'model.safetensors-{index:05d}-of-00004.safetensors'
        entries.append(dict(repo=REPO,revision=REVISION,remote_path=name,
            path='text_encoders/heretic-source/'+name,size=size,sha256=sha256))
    download_files(entries,progress=notify,cancel=cancel)
    check_cancel()
    stage=OUTPUT.with_suffix('.safetensors.partial')
    if stage.exists():
        if stage.stat().st_size==EXPECTED_SIZE and digest(stage)==EXPECTED_SHA256:
            stage.rename(OUTPUT)
            notify('Heretic encoder preparation resumed and verified.')
            return
        stage.rename(stage.with_name(stage.name+'.unverified-'+str(time.time_ns())))
    tensors={}
    for entry in entries:
        check_cancel()
        notify('Preparing Heretic encoder: reading '+entry['remote_path'])
        shard=load_file(str(MODEL_DIR/entry['path']),device='cpu')
        if set(tensors).intersection(shard):
            raise RuntimeError('Duplicate tensors across shards.')
        tensors.update(shard)
    check_cancel()
    notify(f'Preparing Heretic encoder: writing {len(tensors)} unchanged tensors. This may take a few minutes…')
    save_file(tensors,str(stage),metadata={'source_repo':REPO,'source_revision':REVISION,'license':'Apache-2.0'})
    with stage.open('r+b') as stream:
        size=struct.unpack('<Q',stream.read(8))[0]
        normalized=canonical_header(stream.read(size))
        stream.seek(8)
        stream.write(normalized)
    check_cancel()
    notify('Verifying prepared encoder against the original tensors…')
    with safe_open(str(stage),framework='pt',device='cpu') as converted:
        if set(converted.keys()) != set(tensors):
            raise RuntimeError('Converted tensor list differs from source.')
        for key,value in tensors.items():
            check_cancel()
            if not converted.get_tensor(key).equal(value):
                raise RuntimeError('Conversion changed tensor '+key)
    if stage.stat().st_size!=EXPECTED_SIZE or digest(stage)!=EXPECTED_SHA256:
        raise RuntimeError('Prepared encoder did not match the pinned SHA-256; partial file preserved.')
    stage.rename(OUTPUT)
    notify('Heretic encoder installed and verified.')


if __name__=='__main__':
    main()
