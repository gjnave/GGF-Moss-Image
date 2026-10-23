from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

APP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(APP))
import worker
from caption_output import CaptionOutputError


class WorkerCaptionGuardTests(unittest.TestCase):
    def test_unchecked_expansion_sends_original_prompt_to_encoder(self):
        previous_argv=list(sys.argv)
        previous_path=list(sys.path)
        previous_directory=Path.cwd()
        prompt='A mountain bookshop at dusk.'
        try:
            sys.argv=[str(APP/'worker.py')]
            sys.path.insert(0,str(APP/'vendor/comfy_core'))
            import nodes
            from comfy_extras.nodes_textgen import TextGenerate

            with patch('model_store.verify',return_value=(True,'verified')), \
                 patch.object(nodes.CLIPLoader,'load_clip',return_value=(object(),)), \
                 patch.object(TextGenerate,'execute') as expand, \
                 patch.object(nodes.CLIPTextEncode,'encode',side_effect=RuntimeError('encoded')) as encode:
                with self.assertRaisesRegex(RuntimeError,'encoded'):
                    worker.main({'prompt':prompt,'enhance':False})
                expand.assert_not_called()
                self.assertEqual(encode.call_args.args[1],prompt)
        finally:
            sys.argv=previous_argv
            sys.path[:]=previous_path
            os.chdir(previous_directory)

    def test_refusal_stops_before_image_encoding_or_sampling(self):
        previous_argv=list(sys.argv)
        previous_path=list(sys.path)
        previous_directory=Path.cwd()
        try:
            sys.argv=[str(APP/'worker.py')]
            sys.path.insert(0,str(APP/'vendor/comfy_core'))
            import nodes
            from comfy_extras.nodes_textgen import TextGenerate
            with patch('model_store.verify',return_value=(True,'verified')), \
                 patch.object(nodes.CLIPLoader,'load_clip',return_value=(object(),)), \
                 patch.object(TextGenerate,'execute',return_value=("I cannot create that. A different scene could be used instead.",)), \
                 patch.object(nodes.CLIPTextEncode,'encode') as encode, \
                 patch.object(nodes.UNETLoader,'load_unet') as load_image_model, \
                 patch.object(nodes.KSampler,'sample') as sample:
                with self.assertRaises(CaptionOutputError):
                    worker.main({'prompt':'A mountain bookshop at dusk.','enhance':True})
                encode.assert_not_called()
                load_image_model.assert_not_called()
                sample.assert_not_called()
        finally:
            sys.argv=previous_argv
            sys.path[:]=previous_path
            os.chdir(previous_directory)


if __name__=='__main__':
    unittest.main()
