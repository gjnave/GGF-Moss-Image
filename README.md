# Moss Image

[![Moss Image](assets/moss-image-1.png)](https://getgoingfast.pro/tools/moss-image/)

[Get Going Fast — app page and member quick installer](https://getgoingfast.pro/tools/moss-image/)

A local text-to-image application built around the supplied MOSS-Image workflow.
It uses selected Comfy inference modules inside its own disposable Python worker.
It does not install or start the ComfyUI application, browser frontend, node editor,
custom-node manager, or server. It does not connect to another app's environment.

## Manual installation (Windows Command Prompt)

This source repository does not include the member quick installer. Install
[Git](https://git-scm.com/downloads/win) and
[uv](https://docs.astral.sh/uv/getting-started/installation/) first. uv provisions
the required Python privately; your global Python version is not used.

1. Open Command Prompt in the folder where you want the app, then clone it:

   ```bat
   git clone https://codeberg.org/Cognibuild/GGF-Moss-Image.git
   cd GGF-Moss-Image
   ```

   GitHub alternative: `git clone https://github.com/gjnave/GGF-Moss-Image.git`.

2. Create the private Python environment:

   ```bat
   uv venv --managed-python --python 3.12 .venv
   ```

3. Install the CUDA-enabled PyTorch packages, then the app dependencies:

   ```bat
   uv pip install --python .venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
   uv pip install --python .venv\Scripts\python.exe -r requirements-lock.txt
   uv pip check --python .venv\Scripts\python.exe
   .venv\Scripts\python.exe check.py
   ```

4. Double-click RUN.bat. In Settings, choose a model and click Install / resume.
   Wait for the green Installed message, then open Create. Your PC runs the app;
   phone access is optional in Settings. The member quick installer automates
   dependency setup; both routes download models from inside Settings.

## Member package

Extract the complete package to a writable folder. Double-click **INSTALL.bat**
beside `GGF-Moss-Image/`. Setup provisions uv 0.12.13, managed Python 3.12,
a private environment and CUDA 12.8 PyTorch.
No system Python or Git is required. Setup executes a real CUDA matrix operation
before reporting success. Model selection and downloading take place in Settings.
The first launch opens Settings and L/iZ's guide when the selected model is missing.
Choose Smaller (INT8) or Larger (BF16), click Install / resume, and wait for the
green Installed message before opening Create. Later launches open Create when
the selected model is installed. Downloads resume partial files and verify size
and SHA-256 against pinned publisher metadata. No model weights belong in a ZIP.

Then double-click **RUN.bat**. The local page opens automatically, without login.
An NVIDIA GPU and a compatible driver are required. The real validation hardware
and measured results are recorded in the accompanying validation report.
Allow at least 70 GB free for model preparation, dependencies and installation
cache. Model downloads are about 25.0 GB for Smaller or 29.7 GB for Larger.
Both use **Heretic Qwen3.5-9B BF16** as their default text encoder. Settings downloads
four pinned, SHA-256-verified parts from
[pottokao/MOSS-Image-Text-Encoder-Heretic](https://huggingface.co/pottokao/MOSS-Image-Text-Encoder-Heretic)
and automatically combines them into one verified file without changing weights
or running remote model code. Keep the app open until the green Installed message.
The source parts are retained (about 19.3 GB additional disk space); nothing is
deleted automatically. Existing custom encoder selections remain in effect;
clear the custom encoder field and save to use the new default. This encoder
change is not a promised speed or image-quality improvement; caption validation
remains enabled. Both image choices are the same 5B SFT image model at different precision,
not different parameter counts. They use the same app-local text encoder and VAE.
Both image files can remain installed; switching does not remove either one.

## Create

### Custom model files and LoRAs

Settings includes full-path fields with native PC **Browse** buttons for a MOSS
image model, Qwen3.5 text encoder, FLUX.2 VAE, and optional MOSS LoRA. Choose a
compatible `.safetensors` file, then click **Save custom model paths**. Blank,
missing, corrupt, or wrong-category selections fall back to the normal defaults.
Files remain where they are; the app does not copy or move them. Browse and Save
are restricted to the local PC page, while saved choices work for phone generation.
The custom LoRA takes precedence over the Create dropdown and uses its strength
slider. Clear and save the path to stop using it. Fast's image model must match
the verified SFT BF16 base, and Fast always uses its own acceleration adapter;
other custom image models and custom LoRAs apply only in Standard. Custom text
encoders and VAEs can be used in either mode. Header checks establish file type,
not the quality or trustworthiness of third-party weights. Actual selected model
paths are recorded in each output image's metadata.

### Optional Fast mode (local experimental test)

In Settings, click **Install / resume 4-step acceleration**. This adds the BF16
SFT image model and a pinned 1.2 GB Apache-2.0 adapter from
[akhaliq/MOSS-Image-SFT-4-steps](https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps).
It reuses this app's encoder and VAE; nothing is removed. Wait for **Fast mode
installed**, then choose **Fast** on Create. It uses 4 steps, Euler/simple, CFG 1,
no layer guidance or negative prompt, and at most 1 megapixel. Additional LoRAs
are disabled in this mode. The BF16 base keeps separate FP32 adapter calculations,
including fused Q/K/V projections, to avoid rounding the small changes into the
base weights. Model loading and optional caption expansion still take time.
Standard mode retains the original workflow and selected model. Fast is not the
same as the existing half-size Turbo preview. This test has not been published.

The initial prompt is a Get Going Fast branded JSON poster example. Its layout
boxes automatically scale to the selected canvas, including Turbo previews.
Use Load Get Going Fast JSON example to restore it without erasing saved images.
Or describe your image and any exact visible text in quotation marks. Local caption
expansion is enabled by default for ordinary descriptions. Structured JSON layouts
go directly to MOSS so their coordinates and exact text are retained. Invalid JSON
or boxes outside the chosen canvas are rejected before model loading.
The prompt sent to the model is displayed and saved
with the image. Disable expansion to send your prompt directly to the model.
If the caption model declines or returns commentary instead of an image
description, the job stops before image conditioning and sampling. The app
does not extract an alternative from a refusal or retry the original prompt.
The full reply stays in the private job log; the page shows a short status.

The lower-resolution default is 768×768. The sampling settings are retained: 25 steps, CFG 2.5,
ER-SDE sampler, simple schedule, full denoise, random image seed (-1), and its exact negative
prompt. Layer guidance is off; enabling it skips layer 22 on the negative pass
over the first half of sampling. No LoRA is applied by default because the
workflow's Power LoRA Loader has no active LoRA entries. Optional compatible
MOSS diffusion LoRAs can be added under `models/loras/` and selected in the UI.

Choose from 18 size presets or custom 16-pixel-step width/height sliders.
MOSS requires positive multiples of 16; its documented operating range is
262,144–4,194,304 total pixels (512×512 to 2048×2048 worth of pixels), with arbitrary
aspect ratios. This app additionally bounds individual edges to 256–4096 pixels.
Invalid dimensions and nonfinite values are rejected server-side before generation.
For Full HD-like output the preset uses 1920×1088, not unaligned 1920×1080.
Source: https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md

Turbo preview halves dimensions, stopping at the supported minimum pixel count,
and retains sampling settings. At 512×512 it cannot reduce the dimensions further. Sequential
batches generate 1–16 images without multiplying model VRAM. The seed increments
for each batch image. Seed -1 randomizes the starting seed; caption seed is
separate, defaulting to 0 as in the workflow. Ctrl+Enter and Generate use the
same handler. Stop terminates only this app's active worker tree. The worker
exits after each image, releasing its GPU allocations.

Unique PNG filenames prevent overwriting saved results. PNG text metadata records
the original prompt, expanded caption, seed and generation settings. Files remain
under `outputs/` even when a new run clears the displayed result. Job requests
and diagnostic logs are retained under `jobs/`. Browser workspaces restore saved
controls and completed images after refresh; generation continues if the page
closes. Different browser origins have separate browser workspace tokens.
Use **Show saved images on this PC** to reopen up to 50 recent PNGs. All older
files also remain in the outputs folder.

## Phone access and controls

Settings offers PC-only (default), Wi-Fi, and a temporary public phone link.
The PC does the work and must remain on. A separate remote page shares the same
GPU lock. A blank remote password disables remote login; a nonblank password is
stored as a salted PBKDF2 hash. The local PC page always remains login-free.
Save settings and restart the app to apply changes. Preferred local port is 7891,
remote port 8891, with fallback to the next 19 ports.

Settings includes Release models / free GPU memory and confirmed Stop server.
They refuse while generation is active. Stop server closes local and remote
servers; restart RUN.bat on the PC to use the application again.

## Sources and scope

Workflow: `workflow.json`, copied unchanged from the supplied `txt2img.json`.
UI size, prompt and model selections override the corresponding workflow inputs.
Models: Comfy-Org/MOSS-Image revision ee64003102c8470ee32501cf883419235db81884;
optional SFT BF16 image file pinned to 24147db865fcdc207270f844777587223fed8d6e.
Inference core: Comfy-Org/ComfyUI d80736d2aa63d318bbe60417e75dd1d49d532eaf.
See `models.json`, `THIRD-PARTY-NOTICES.md`, and `vendor/comfy_core/SOURCE.md`.

Source repositories: [Codeberg](https://codeberg.org/Cognibuild/GGF-Moss-Image) and
[GitHub](https://github.com/gjnave/GGF-Moss-Image). Close the app before updating.
The member package includes UPDATE.bat. For a manual installation, run
`.venv\Scripts\python.exe update.py`, then repeat the dependency commands above.
The updater checks repository commits and retrieves the host-generated archive;
there is no manually maintained source ZIP. It backs up replaced source files
and preserves local models, environments, jobs, outputs and private settings.

The final installation uses ordinary app-relative folders on D:. The initial
E:-linked setup was preserved as a separate backup while the final packaged
installer was tested with a fresh private environment. The customer package
has no dependency on either drive letter.
