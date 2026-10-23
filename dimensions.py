"""MOSS uses a 16-pixel latent grid; keep production sizes in its tested pixel range."""
import math

DEFAULT_SIZE = 768
MIN_EDGE, MAX_EDGE, ALIGNMENT = 256, 4096, 16
MIN_PIXELS, MAX_PIXELS = 512 * 512, 2048 * 2048
SHAPES = {
    'Square · 512 × 512 (quick)': (512, 512),
    'Square · 768 × 768 (default)': (768, 768),
    'Square · 1024 × 1024': (1024, 1024),
    'Square · 1536 × 1536': (1536, 1536),
    'Square · 2048 × 2048': (2048, 2048),
    'Landscape · 1024 × 576': (1024, 576),
    'Landscape · 1280 × 720': (1280, 720),
    'Landscape · 1344 × 768': (1344, 768),
    'Landscape · 1536 × 864': (1536, 864),
    'Landscape · 1920 × 1088': (1920, 1088),
    'Landscape · 2048 × 1152': (2048, 1152),
    'Portrait · 576 × 1024': (576, 1024),
    'Portrait · 720 × 1280': (720, 1280),
    'Portrait · 768 × 1344': (768, 1344),
    'Portrait · 864 × 1536': (864, 1536),
    'Portrait · 1088 × 1920': (1088, 1920),
    'Portrait · 1152 × 2048': (1152, 2048),
    'Wide banner · 1536 × 640': (1536, 640),
    'Custom': None,
}
DEFAULT_SHAPE = 'Square · 768 × 768 (default)'


def validate_dimensions(width, height):
    for name, value in [('Width', width), ('Height', height)]:
        number = float(value)
        if not math.isfinite(number) or number != int(number):
            raise ValueError(f'{name} must be a whole number.')
        if not MIN_EDGE <= number <= MAX_EDGE or int(number) % ALIGNMENT:
            raise ValueError(f'{name} must be {MIN_EDGE}–{MAX_EDGE} pixels, in multiples of 16.')
    width, height = int(width), int(height)
    if not MIN_PIXELS <= width * height <= MAX_PIXELS:
        raise ValueError('Choose 262,144–4,194,304 total pixels (512×512 to 2048×2048 worth of pixels).')
    return width, height


def preview_dimensions(width, height):
    width, height = validate_dimensions(width, height)
    ratio = max(0.5, math.sqrt(MIN_PIXELS / (width * height)))
    # Round upward so a reduced preview never falls below the supported pixel budget.
    return validate_dimensions(math.ceil(width * ratio / 16) * 16,
                               math.ceil(height * ratio / 16) * 16)
