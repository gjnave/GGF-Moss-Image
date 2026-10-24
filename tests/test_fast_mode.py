import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
import fast_mode


class FastTests(unittest.TestCase):
    def test_standard_untouched_and_fast_constraints(self):
        original={'width':768,'height':768,'steps':25,'cfg':2.5,'slg':True,'negative':'blur','lora':'None'}
        standard=fast_mode.configure(copy.deepcopy(original))
        self.assertEqual(standard['steps'],25)
        self.assertTrue(standard['slg'])
        fast=fast_mode.configure({**original,'generation_mode':'fast4'})
        self.assertEqual((fast['steps'],fast['cfg'],fast['sampler'],fast['model']),(4,1.0,'euler','bf16'))
        self.assertFalse(fast['slg'])
        self.assertEqual(fast['negative'],'')
        for extra in [{'width':2048,'height':2048},{'lora':'other.safetensors'},{'generation_mode':'bad'}]:
            with self.assertRaises(ValueError):
                fast_mode.configure({**original,'generation_mode':'fast4',**extra})

    def test_fp32_adapter_and_fused_qkv_all_apply(self):
        for fused in (False,True):
            with self.subTest(fused=fused):
                weights={}
                projections=nn.ModuleDict()
                names=('qkv','o','ffn_up_gate','ffn_down') if fused else ('q','k','v','o','ffn_up_gate','ffn_down')
                for name in names:
                    module=nn.Linear(2,6 if name=='qkv' else 2,bias=False)
                    nn.init.zeros_(module.weight)
                    projections[name]=module
                for index,name in enumerate(('q','k','v','o','ffn_up_gate','ffn_down')):
                    prefix=f'diffusion_model.layers.0.proj.{name}'
                    weights[prefix+'.lora_down.weight']=torch.ones(1,2)
                    weights[prefix+'.lora_up.weight']=torch.full((2,1),float(index+1))
                    weights[prefix+'.alpha']=torch.tensor(1.)
                model=SimpleNamespace(model=SimpleNamespace(diffusion_model=SimpleNamespace(layers=[SimpleNamespace(proj=projections)])))
                with patch('safetensors.torch.load_file',return_value=weights):
                    fast_mode.attach_adapter(model)
                x=torch.ones(1,2)
                expected=torch.tensor([[2.,2.,4.,4.,6.,6.]])
                actual=projections['qkv'](x) if fused else torch.cat([projections[n](x) for n in ('q','k','v')],dim=-1)
                torch.testing.assert_close(actual,expected)
                torch.testing.assert_close(projections['ffn_down'](x),torch.tensor([[12.,12.]]))
                self.assertTrue(projections['ffn_down'].comfy_force_forward)


if __name__=='__main__':
    unittest.main()
