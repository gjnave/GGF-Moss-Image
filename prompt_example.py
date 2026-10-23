"""A native MOSS layout example, adapted to the actual output canvas."""
import copy
import json

EXAMPLE = {
    'summary': {
        'meta': ['poster'], 'canvas': 'single',
        'caption': 'A polished Get Going Fast poster. Deep navy background, glowing cyan trails, '
                   'gold lettering and a friendly small robot carrying a paintbrush. Clear readable typography.'
    },
    'items': [
        {'label': 'text', 'caption': 'Large bold gold headline centered across the top.',
         'text': 'GET GOING FAST', 'bbox': [-320, -350, -210, 350]},
        {'label': 'object', 'caption': 'A cheerful little silver robot holding a paintbrush, '
         'with swirling cyan and gold light trails.', 'bbox': [-160, -240, 180, 240]},
        {'label': 'text', 'caption': 'Crisp white subtitle centered near the bottom.',
         'text': 'YOUR IDEAS. BROUGHT TO LIFE.', 'bbox': [210, -350, 270, 350]},
        {'label': 'text', 'caption': 'Small gold website address below the subtitle.',
         'text': 'GetGoingFast.pro', 'bbox': [290, -250, 330, 250]},
    ],
    'bbox_coord_space': 'abs_centralized', 'bbox_format': 'yxyx'
}
DEFAULT_PROMPT = json.dumps(EXAMPLE, indent=2)


def layout_prompt(prompt, width, height, source_size=None):
    """Do not send structured layouts through a conversational caption expander."""
    if not prompt.lstrip().startswith('{'):
        return prompt, False
    try:
        value = json.loads(prompt)
    except ValueError as error:
        raise ValueError('The JSON prompt is not valid. Check its quotes, commas and brackets.') from error
    if not isinstance(value, dict) or not isinstance(value.get('summary'), dict) or not isinstance(value.get('items'), list):
        raise ValueError('A MOSS JSON layout needs a summary object and an items list.')
    if value.get('bbox_format') != 'yxyx' or value.get('bbox_coord_space') != 'abs_centralized':
        raise ValueError('Use bbox_format "yxyx" and bbox_coord_space "abs_centralized".')
    if source_size is not None:
        from dimensions import validate_dimensions
        source_width, source_height = validate_dimensions(*source_size)
        # Validate on the full-size canvas before resizing a custom Turbo layout.
        original, _ = layout_prompt(prompt, source_width, source_height)
        value = json.loads(original)
        for item in value['items']:
            item['bbox'] = [round(n * (height / source_height if i % 2 == 0 else width / source_width))
                            for i, n in enumerate(item['bbox'])]
    elif value == EXAMPLE:
        value = copy.deepcopy(value)
        for item in value['items']:
            item['bbox'] = [round(n * (height if i % 2 == 0 else width) / 768)
                            for i, n in enumerate(item['bbox'])]
    for item in value['items']:
        box = item.get('bbox') if isinstance(item, dict) else None
        if not isinstance(box, list) or len(box) != 4 or any(type(n) not in (int, float) for n in box):
            raise ValueError('Each JSON layout item needs four numeric bbox coordinates.')
        y0, x0, y1, x1 = box
        if not (-height/2 <= y0 < y1 <= height/2 and -width/2 <= x0 < x1 <= width/2):
            raise ValueError('A JSON layout box falls outside the chosen canvas. Adjust its coordinates or choose a larger size.')
    return json.dumps(value, ensure_ascii=False), True
