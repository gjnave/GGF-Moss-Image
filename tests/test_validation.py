import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from validation import validate


class ValidationTests(unittest.TestCase):
    def test_workflow_defaults(self):
        value=validate({'prompt':'A red book'})
        self.assertEqual((value['width'],value['height'],value['steps'],value['cfg'],value['seed']),
                         (768,768,25,2.5,0))
        self.assertTrue(value['enhance'])
        self.assertFalse(value['slg'])
        self.assertEqual(value['lora'],'None')

    def test_random_seed_and_explicit_zero(self):
        self.assertEqual(validate({'prompt':'x','seed':0})['seed'],0)
        value=validate({'prompt':'x','seed':-1})['seed']
        self.assertGreaterEqual(value,0)
        self.assertLess(value,2**32)

    def test_invalid_dimensions_and_values(self):
        for bad in [{'prompt':''},{'width':1025},{'width':-1},{'height':float('nan')},
                    {'steps':0},{'steps':1.5},{'cfg':float('inf')},{'seed':-2},
                    {'lora_strength':float('nan')}]:
            with self.subTest(bad=bad),self.assertRaises((ValueError,OverflowError)):
                validate({'prompt':'A book',**bad})


if __name__=='__main__':
    unittest.main()
