# Third-party notices

- MOSS-Image by OpenMOSS: https://github.com/OpenMOSS/MOSS-Image and
  https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT. Apache-2.0 according to
  the publisher. The selected model repack is downloaded separately from
  https://huggingface.co/Comfy-Org/MOSS-Image; weights are never included here.
- ComfyUI inference source: https://github.com/Comfy-Org/ComfyUI, pinned to
  d80736d2aa63d318bbe60417e75dd1d49d532eaf. GPL-3.0; original license and source
  notice are included in vendor/comfy_core. App integration source is supplied
  under GPL-3.0 consistently with these embedded modules. This package is not
  endorsed by Comfy Org or OpenMOSS.
- Gradio: https://github.com/gradio-app/gradio, Apache-2.0.
- Default Qwen3.5 text encoder: https://huggingface.co/pottokao/MOSS-Image-Text-Encoder-Heretic,
  Apache-2.0. Pinned upstream shards are downloaded separately and repacked
  without changing weights; no model weights are distributed in the installer.
- Optional 4-step adapter: https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps,
  Apache-2.0. Downloaded separately; its FP32 deltas remain unmerged on the BF16 base.
- Transformers: https://github.com/huggingface/transformers, Apache-2.0.
- PyTorch: https://github.com/pytorch/pytorch, BSD-style license.
- uv: https://github.com/astral-sh/uv, MIT or Apache-2.0.

Installed dependency distributions retain their own license files. The GGF brain
logo and supplied about/disclaimer materials are owner-provided brand assets;
their inclusion does not grant independent trademark rights.
