from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from caption_output import CaptionOutputError, image_caption, display_caption


class CaptionOutputTests(unittest.TestCase):
    def test_refusal_and_alternative_are_never_forwarded(self):
        for text in [
            'I cannot generate that content. I can, however, help with a different scene.',
            "I can't create this. Let's try a watercolor landscape instead.",
            "I'm unable to help. A cabin stands beside a lake.",
            'I must decline this request. A blue sailing boat on a lake.',
            'Here is the caption: I cannot provide that description.',
        ]:
            with self.subTest(text=text), self.assertRaises(CaptionOutputError):
                image_caption(text)
            self.assertEqual(display_caption(text),'')

    def test_chat_commentary_is_not_an_image_prompt(self):
        for text in ['', '  ', "Let's try something else.", 'Would you like a different image?',
                     '<think>unfinished reasoning', '```json\n{}\n```']:
            with self.subTest(text=text), self.assertRaises(CaptionOutputError):
                image_caption(text)

    def test_actual_scene_description_is_preserved(self):
        text='A watercolor painting of a mountain bookshop at dusk. A sign reads "THE LAST CHAPTER".'
        self.assertEqual(image_caption(text),text)

    def test_quoted_visible_text_is_not_a_refusal(self):
        text='A poster displays "I cannot wait!" in gold letters above a mountain sunrise.'
        self.assertEqual(image_caption(text),text)

    def test_harmless_wrapper_removed(self):
        text='A red boat floats on a quiet blue lake.'
        self.assertEqual(image_caption('Here is your caption:\n'+text),text)
        self.assertEqual(image_caption('```text\n'+text+'\n```'),text)


if __name__=='__main__':
    unittest.main()
