from pathlib import Path
import copy
import json
import sys
import unittest
import importlib.util
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dimensions import SHAPES, validate_dimensions, preview_dimensions
from prompt_example import EXAMPLE, DEFAULT_PROMPT, layout_prompt
from validation import validate
import model_store
spec=importlib.util.spec_from_file_location('moss_ui_tests',Path(__file__).resolve().parents[1]/'app.py')
ui=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)


class ProductionTests(unittest.TestCase):
    def test_required_backend_model_code_is_packaged(self):
        backend=Path(__file__).resolve().parents[1]/'vendor/comfy_core/comfy/ldm/models/autoencoder.py'
        self.assertTrue(backend.is_file(),'Backend Python modules must not be excluded with model weights.')

    def test_all_presets_and_turbo_use_supported_grid_and_area(self):
        for name,size in SHAPES.items():
            if size:
                with self.subTest(name=name):
                    self.assertEqual(validate_dimensions(*size),size)
                    preview=preview_dimensions(*size)
                    self.assertEqual(validate_dimensions(*preview),preview)
                    self.assertLessEqual(preview[0],size[0])
                    self.assertLessEqual(preview[1],size[1])

    def test_invalid_dimensions_rejected(self):
        for size in [(1920,1080),(256,256),(4096,2048),(0,768),
                     (512.5,512),(float('nan'),512),(8192,512)]:
            with self.subTest(size=size),self.assertRaises(ValueError):
                validate_dimensions(*size)
        self.assertEqual(validate_dimensions(4096,1024),(4096,1024))
        self.assertEqual(validate_dimensions(256,1024),(256,1024))

    def test_branded_layout_scales_to_every_preset(self):
        for size in filter(None,SHAPES.values()):
            with self.subTest(size=size):
                prompt,structured=layout_prompt(DEFAULT_PROMPT,*size)
                self.assertTrue(structured)
                result=json.loads(prompt)
                self.assertEqual(result['items'][0]['text'],'GET GOING FAST')
                # Validate the transformed custom layout again, including Turbo.
                layout_prompt(prompt,*size)
                preview_dimensions(*size)

    def test_plain_prompt_unchanged_and_bad_layouts_rejected(self):
        self.assertEqual(layout_prompt('A gold robot',768,768),('A gold robot',False))
        for prompt in ['{bad json}', '{}', '{"summary":[],"items":[]}']:
            with self.assertRaises(ValueError):
                layout_prompt(prompt,768,768)
        value=copy.deepcopy(EXAMPLE)
        value['items'][0]['bbox'][0]=-10000
        with self.assertRaisesRegex(ValueError,'outside'):
            layout_prompt(json.dumps(value),768,768)

    def test_custom_layout_scales_for_turbo(self):
        value=copy.deepcopy(EXAMPLE)
        value['items'][0]['text']='A CUSTOM POSTER'
        value['items'][0]['bbox']=[-480,-480,-200,480]
        prompt,structured=layout_prompt(json.dumps(value),512,512,source_size=(1024,1024))
        self.assertTrue(structured)
        self.assertEqual(json.loads(prompt)['items'][0]['bbox'],[-240,-240,-100,240])

    def test_variants_select_distinct_image_weights_and_shared_support(self):
        small=model_store.model_files('int8')
        large=model_store.model_files('bf16')
        self.assertEqual(len(small),3)
        self.assertEqual(len(large),3)
        self.assertNotEqual(small[0]['path'],large[-1]['path'])
        self.assertIn('bf16',model_store.model_names('bf16')['diffusion_models'])
        self.assertEqual(model_store.model_names('bf16')['text_encoders'],
                         model_store.model_names('int8')['text_encoders'])
        self.assertEqual(sum(x['size'] for x in small),24969532754)
        self.assertEqual(sum(x['size'] for x in large),29720489604)
        self.assertEqual(model_store.model_names('int8')['text_encoders'],'qwen3.5_9b_heretic_bf16.safetensors')
        with self.assertRaises(ValueError):
            model_store.model_files('../../bad')

    def test_first_run_opens_settings_then_create_when_ready(self):
        app=ui
        with patch.object(app,'verify',return_value=(False,'missing')):
            self.assertEqual(app.initial_tab(),'settings')
        with patch.object(app,'verify',return_value=(True,'installed')):
            self.assertEqual(app.initial_tab(),'create')

    def test_busy_selection_preserves_model(self):
        app=ui
        with patch.object(app,'ACTIVE',{'token':'test'}),patch.object(app,'select_model') as select:
            _,note=app.change_model('bf16')
            select.assert_not_called()
            self.assertIn('Stop',note)

    def test_missing_models_disable_generation(self):
        app=ui
        with patch.object(app,'verify',return_value=(False,'missing')):
            values=app.refresh('a'*32)
            self.assertFalse(values[3]['interactive'])
            self.assertFalse(values[4]['interactive'])


if __name__=='__main__':
    unittest.main()
