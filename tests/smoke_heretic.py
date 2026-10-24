"""One neutral end-to-end local encoder trial. Does not save app settings."""
from pathlib import Path
import contextlib
import sys
import time

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import worker
from prepare_heretic_encoder import OUTPUT
from custom_models import check

if __name__ == '__main__':
    path,note = check(str(OUTPUT),'text_encoders')
    if not path:
        raise RuntimeError(note)
    request = {
        'prompt': 'A cheerful silver robot painting a bright mountain landscape in a sunlit workshop. A small sign reads "GET GOING FAST".',
        'width': 768, 'height': 768, 'seed': 0, 'enhance': True,
        'generation_mode': 'fast4',
        '_custom_models': {'text_encoders': path},
        'output': str(APP / 'outputs' / ('heretic-encoder-trial-' + str(time.time_ns()) + '.png')),
    }
    with contextlib.redirect_stdout(sys.stderr):
        worker.main(request)
