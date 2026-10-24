import math
import secrets
from dimensions import DEFAULT_SIZE, validate_dimensions


def validate(data):
    data = dict(data)
    data['prompt'] = str(data.get('prompt','')).strip()
    if not data['prompt']:
        raise ValueError('Describe the image you want to create.')
    for name, default in [('width',DEFAULT_SIZE),('height',DEFAULT_SIZE),('steps',25),('seed',-1),('enhance_seed',0)]:
        value = float(data.get(name,default))
        if not math.isfinite(value) or int(value) != value:
            raise ValueError(name + ' must be a whole number.')
        data[name] = int(value)
    data['width'], data['height'] = validate_dimensions(data['width'], data['height'])
    if not 1 <= data['steps'] <= 100:
        raise ValueError('Steps must be between 1 and 100.')
    data['cfg'] = float(data.get('cfg',2.5))
    if not math.isfinite(data['cfg']) or not 0 <= data['cfg'] <= 15:
        raise ValueError('Guidance must be between 0 and 15.')
    for name in ['seed','enhance_seed']:
        if data[name] == -1:
            data[name] = secrets.randbits(32)
        if not 0 <= data[name] <= 2**53-1:
            raise ValueError('Seed must be -1 or between 0 and 9007199254740991.')
    data.setdefault('negative','worst quality, bad anatomy, watermark, jpeg artifact')
    data.setdefault('enhance',True)
    data.setdefault('slg',False)
    data.setdefault('lora','None')
    data['lora_strength'] = float(data.get('lora_strength',1))
    if not math.isfinite(data['lora_strength']) or not -2 <= data['lora_strength'] <= 2:
        raise ValueError('LoRA strength must be between -2 and 2.')
    return data
