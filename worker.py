"""One-shot worker: translates workflow operations directly into local core calls."""
from pathlib import Path
import contextlib
import json
import os
import sys
import time
import traceback
from caption_output import CaptionOutputError, image_caption

APP = Path(__file__).resolve().parent
PROTOCOL = sys.stdout


def emit(**fields):
    print(json.dumps(fields, ensure_ascii=True), file=PROTOCOL, flush=True)


def main(request):
    from model_store import MODEL_DIR, verify, model_names
    variant = request.get('model', 'int8')
    ready, note = verify(variant=variant)
    if not ready:
        raise RuntimeError(note + ' Open Settings and install the selected model.')
    from validation import validate
    request = validate(request)
    from prompt_example import layout_prompt
    prepared_prompt, structured = layout_prompt(request['prompt'], request['width'], request['height'],request.get('layout_source'))
    names = model_names(variant)
    core = APP / 'vendor/comfy_core'
    sys.path.insert(0, str(core))
    sys.argv = [sys.argv[0], '--models-directory', str(MODEL_DIR), '--disable-mmap']
    os.chdir(core)
    # Upstream app.governance is a package; this worker never imports our app.py.
    import torch
    import nodes
    import folder_paths
    import comfy.model_management as mm
    from comfy_extras.nodes_flux import EmptyFlux2LatentImage
    from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple
    from comfy_extras.nodes_textgen import TextGenerate
    if not torch.cuda.is_available():
        raise RuntimeError('An NVIDIA CUDA GPU is required.')
    for category in ['diffusion_models', 'text_encoders', 'vae', 'loras']:
        (MODEL_DIR / category).mkdir(exist_ok=True)
        folder_paths.add_model_folder_path(category, str(MODEL_DIR / category), is_default=True)
    flow = json.loads((APP / 'workflow.json').read_text())
    began = time.monotonic()
    emit(status='Loading the image text encoder…')
    clip = nodes.CLIPLoader().load_clip(names['text_encoders'], 'moss_image', 'default')[0]
    prompt = prepared_prompt
    with torch.no_grad():
        if request['enhance'] and not structured:
            emit(status='Expanding your idea into a detailed image caption…')
            prompt = TextGenerate.execute(clip=clip, prompt=prompt, max_length=4096,
                sampling_mode={'sampling_mode':'on', 'temperature':0.7, 'top_k':64,
                    'top_p':0.95, 'min_p':0.05, 'repetition_penalty':1.0,
                    'seed':request['enhance_seed'], 'presence_penalty':0.0},
                thinking=False, use_default_template=True, mtp='auto',
                system_prompt=flow['14']['inputs']['value'])[0]
            print('Raw caption response: '+json.dumps(prompt,ensure_ascii=True),file=sys.stderr,flush=True)
            prompt = image_caption(prompt)
        emit(prompt=prompt, status=('Encoding your JSON layout…' if structured else
            'Encoding the expanded caption…' if request['enhance'] else
            'Encoding your original prompt, unchanged…'))
        positive = nodes.CLIPTextEncode().encode(clip, prompt)[0]
        negative = nodes.CLIPTextEncode().encode(clip, request['negative'])[0]
        # The encoder and image transformer do not need to occupy VRAM together.
        del clip
        mm.unload_all_models()
        mm.soft_empty_cache()
        emit(status='Loading MOSS image model…')
        model = nodes.UNETLoader().load_unet(names['diffusion_models'], 'default')[0]
        if request['lora'] != 'None':
            lora_path = (MODEL_DIR / 'loras' / request['lora']).resolve()
            if not lora_path.is_relative_to((MODEL_DIR / 'loras').resolve()) or not lora_path.is_file():
                raise ValueError('Selected LoRA is missing from this app.')
            import comfy.sd, comfy.utils
            weights = comfy.utils.load_torch_file(str(lora_path), safe_load=True)
            model, _ = comfy.sd.load_lora_for_models(model, None, weights, request['lora_strength'], 0)
        if request['slg']:
            model = SkipLayerGuidanceDiTSimple.execute(model, start_percent=0.0,
                end_percent=0.5, double_layers='', single_layers='22')[0]
        latent = EmptyFlux2LatentImage.execute(request['width'], request['height'], 1)[0]
        import latent_preview
        def callback_factory(model, steps, x0_output_dict=None):
            def callback(step, x0, x, total):
                emit(step=step+1, total=total, status=f'Generating · step {step+1} of {total}')
            return callback
        latent_preview.prepare_callback = callback_factory
        emit(status='Generating the image…')
        samples = nodes.KSampler().sample(model=model, seed=request['seed'],
            steps=request['steps'], cfg=request['cfg'], sampler_name='er_sde', scheduler='simple',
            positive=positive, negative=negative, latent_image=latent, denoise=1.0)[0]
        del model
        mm.unload_all_models()
        mm.soft_empty_cache()
        emit(status='Decoding the image…')
        vae = nodes.VAELoader().load_vae(names['vae'])[0]
        pixels = nodes.VAEDecode().decode(vae, samples)[0][0]
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    import numpy as np
    result = Image.fromarray((pixels.detach().cpu().numpy().clip(0,1)*255).astype(np.uint8))
    output = Path(request['output'])
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = PngInfo()
    metadata.add_text('GGF_MOSS', json.dumps({**request,'expanded_prompt':prompt}))
    result.save(output, pnginfo=metadata)
    emit(complete=str(output), seed=request['seed'], prompt=prompt,
         seconds=round(time.monotonic()-began,2))


if __name__ == '__main__':
    try:
        payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
        with contextlib.redirect_stdout(sys.stderr):
            main(payload)
    except CaptionOutputError as error:
        emit(error=str(error), code=error.code)
        sys.exit(2)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        emit(error=f'{type(error).__name__}: {error}')
        sys.exit(1)
