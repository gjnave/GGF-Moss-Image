"""Installer diagnostics execute a real CUDA kernel and import the local core."""
from pathlib import Path
import sys
import torch

root = Path(__file__).resolve().parent
assert sys.version_info[:2] == (3,12), 'Python 3.12 required'
assert torch.cuda.is_available(), 'NVIDIA CUDA GPU required'
print('Torch:',torch.__version__,'CUDA:',torch.version.cuda)
print('GPU:',torch.cuda.get_device_name(0),'capability:',torch.cuda.get_device_capability(0))
print('Architectures:',torch.cuda.get_arch_list())
assert 'sm_120' in torch.cuda.get_arch_list(), 'RTX 50 support missing'
x=torch.randn(128,128,device='cuda')
assert torch.isfinite((x @ x.T).sum()).item()
torch.cuda.synchronize()
print('CUDA matrix operation: PASS')
sys.path.insert(0,str(root/'vendor/comfy_core'))
sys.argv=[sys.argv[0]]
import nodes
from comfy_extras.nodes_textgen import TextGenerate
from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple
from comfy_extras.nodes_flux import EmptyFlux2LatentImage
assert 'er_sde' in nodes.comfy.samplers.KSampler.SAMPLERS
assert 'MOSS_IMAGE' in nodes.comfy.sd.CLIPType.__members__
assert 'server' not in sys.modules, 'Inference core unexpectedly imported a Comfy server'
print('MOSS core, ER-SDE, text generation, latent and layer guidance: PASS')
if '--models' in sys.orig_argv:
    from model_store import verify
    ready,note=verify(full=True)
    print(note)
    assert ready,note
