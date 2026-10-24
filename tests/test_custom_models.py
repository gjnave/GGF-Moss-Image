import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import custom_models as cm
from model_store import MODEL_DIR, model_names


class CustomModelTests(unittest.TestCase):
    def test_blank_and_missing_use_default(self):
        self.assertEqual(cm.check('', 'vae')[0], '')
        self.assertEqual(cm.check(str(MODEL_DIR/'not-installed.safetensors'),'vae')[0], '')

    def test_installed_files_recognized_and_wrong_slot_rejected(self):
        for category,name in model_names('bf16').items():
            path=MODEL_DIR/category/name
            if not path.is_file():
                continue
            with self.subTest(category=category):
                self.assertEqual(cm.check(str(path),category)[0],str(path.resolve()))
                other='vae' if category!='vae' else 'diffusion_models'
                self.assertEqual(cm.check(str(path),other)[0],'')

    def test_only_replaced_default_is_skipped(self):
        with patch.object(cm,'verify_entries',return_value=(True,'ok')) as verify:
            cm.ready('int8',overrides={'text_encoders':'D:/chosen/encoder.safetensors'})
        paths=[x['path'] for x in verify.call_args.args[0]]
        self.assertEqual(len(paths),2)
        self.assertFalse(any(p.startswith('text_encoders/') for p in paths))

    def test_fast_ignores_custom_lora(self):
        with patch.object(cm,'check',side_effect=lambda value,category:(value,'ready') if value else ('','Default')):
            custom,notes=cm.resolve(mode='fast4',settings={'lora':'D:/example.safetensors'})
        self.assertNotIn('lora',custom)
        self.assertTrue(any('Standard' in n for n in notes))

    def test_custom_folder_precedes_default_even_for_same_basename(self):
        folders=MagicMock()
        name=model_names()['text_encoders']
        names=cm.register(folders,'int8',{'text_encoders':str(MODEL_DIR/'external'/name)})
        self.assertEqual(names['text_encoders'],name)
        self.assertEqual(folders.add_model_folder_path.call_args.kwargs,{'is_default':True})


if __name__=='__main__':
    unittest.main()
