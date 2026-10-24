import json
import unittest
from unittest.mock import MagicMock, patch
import model_store
import prepare_heretic_encoder as encoder


class EncoderInstallTests(unittest.TestCase):
    def test_metadata_order_is_reproducible(self):
        source={'__metadata__':{'source_repo':encoder.REPO,'license':'Apache-2.0','source_revision':encoder.REVISION},
                'tensor':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}
        raw=json.dumps(source,separators=(',',':')).encode().ljust(512,b' ')
        normal=encoder.canonical_header(raw)
        self.assertEqual(list(json.loads(normal)['__metadata__']),['source_revision','license','source_repo'])
        self.assertEqual(encoder.canonical_header(normal),normal)
        self.assertEqual(json.loads(normal)['tensor'],source['tensor'])

    def test_fresh_download_prepares_encoder_without_direct_url(self):
        entry=next(x for x in model_store.model_files('int8') if x.get('prepare'))
        directory=MagicMock()
        target=directory.__truediv__.return_value
        target.is_file.side_effect=[False,True]
        target.stat.return_value.st_size=entry['size']
        with patch.object(model_store,'MODEL_DIR',directory), \
             patch.object(encoder,'main') as prepare, \
             patch.object(model_store,'digest',return_value=entry['sha256']), \
             patch.object(model_store,'verify_entries',return_value=(True,'installed')):
            model_store.download_files([entry],progress=lambda _:None)
        prepare.assert_called_once()
        self.assertIn('progress',prepare.call_args.kwargs)

    def test_default_matches_verified_recipe(self):
        entry=next(x for x in model_store.model_files('int8') if x.get('prepare'))
        self.assertEqual(entry['size'],encoder.EXPECTED_SIZE)
        self.assertEqual(entry['sha256'],encoder.EXPECTED_SHA256)
        self.assertEqual(entry['prepare'],'heretic_encoder')


if __name__=='__main__':
    unittest.main()
