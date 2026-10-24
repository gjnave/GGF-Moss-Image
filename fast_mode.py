"""Optional Euler-native 4-step adapter; the standard workflow is unchanged.

Adapter: akhaliq/MOSS-Image-SFT-4-steps (Apache-2.0).
https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps
Keep FP32 low-rank deltas separate from BF16 base weights, including fused QKV.
"""
from model_store import MANIFEST, MODEL_DIR, RECEIPT_LOCK, model_files, verify_entries, download_files

MODES = {'Standard · original workflow': 'standard', 'Fast · 4-step acceleration (experimental)': 'fast4'}
ADAPTER = {
    'repo': 'akhaliq/MOSS-Image-SFT-4-steps',
    'revision': '2ded95c37a029360245926fe72e475e30039bbfb',
    'remote_path': 'moss_image_sft_4step_v2.0.safetensors',
    'path': 'loras/moss_image_sft_4step_v2.0.safetensors',
    'size': 1189176752,
    'sha256': '154c7f1455c3bf9925945c2f1ea2e4f09d0d0c67256081384a094b2b585f913f',
}


def files():
    return model_files('bf16') + [ADAPTER]


def ready():
    with RECEIPT_LOCK:
        return verify_entries(files())


def install(progress=None, cancel=None):
    download_files(files(), progress, cancel)


def configure(request):
    """Apply mode-specific constraints; never silently mix incompatible recipes."""
    mode = request.get('generation_mode', 'standard')
    if mode not in ('standard', 'fast4'):
        raise ValueError('Choose Standard or Fast generation mode.')
    if mode == 'fast4':
        if request.get('lora', 'None') != 'None':
            raise ValueError('Fast mode uses its own adapter. Set Optional LoRA to None.')
        if request['width'] * request['height'] > 1048576:
            raise ValueError('Fast 4-step testing supports up to 1 megapixel. Choose a smaller size or Standard.')
        request.update(model='bf16', steps=4, cfg=1.0, slg=False, negative='', sampler='euler')
    else:
        request['sampler'] = 'er_sde'
    return request


def attach_adapter(model):
    """Forward hooks preserve FP32 deltas without replacing Comfy offload modules.

    Adapter tensors stay on CPU. Each projection transfers only its own factors,
    avoiding a permanent extra 1.2 GB GPU allocation outside Comfy's VRAM budget.
    Hooks also apply all three Q/K/V deltas to Comfy's fused QKV projection.
    """
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    weights = load_file(str(MODEL_DIR / ADAPTER['path']), device='cpu')
    transformer = model.model.diffusion_model
    consumed = set()

    def factors(index, name):
        prefix = f'diffusion_model.layers.{index}.proj.{name}'
        keys = [prefix + suffix for suffix in ('.lora_down.weight', '.lora_up.weight', '.alpha')]
        down, up, alpha = (weights[key] for key in keys)
        consumed.update(keys)
        return down.float(), up.float(), float(alpha) / down.shape[0]

    def hook(parts, fused=False):
        def apply(module, inputs, output):
            x = inputs[0]
            with torch.autocast(device_type=x.device.type, enabled=False):
                deltas = []
                for down, up, scale in parts:
                    delta = F.linear(F.linear(x.float(), down.to(x.device)), up.to(x.device)) * scale
                    deltas.append(delta)
                delta = torch.cat(deltas, dim=-1) if fused else deltas[0]
            return output + delta.to(output.dtype)
        return apply

    for index, layer in enumerate(transformer.layers):
        if 'qkv' in layer.proj:
            layer.proj['qkv'].register_forward_hook(hook([factors(index, n) for n in ('q', 'k', 'v')], True))
        else:
            for name in ('q', 'k', 'v'):
                layer.proj[name].register_forward_hook(hook([factors(index, name)]))
        for name in ('o', 'ffn_up_gate', 'ffn_down'):
            layer.proj[name].register_forward_hook(hook([factors(index, name)]))
        # Fused activation+linear paths otherwise skip module forward hooks.
        layer.proj['ffn_down'].comfy_force_forward = True
    if consumed != set(weights):
        raise RuntimeError('Acceleration adapter has unexpected tensors; refusing partial application.')
    return model
