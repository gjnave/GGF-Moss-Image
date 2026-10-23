"""Validate caption-model output before any text reaches image conditioning.

Do not fall back to the original prompt after a refusal or extract an alternate
caption from a refusal. Keep the full response in the job diagnostic log only.
"""
from __future__ import annotations

import re

DECLINED_MESSAGE = 'The caption model declined this request. No image was generated.'
INVALID_MESSAGE = 'The caption model returned commentary instead of an image description. No image was generated.'


class CaptionOutputError(ValueError):
    def __init__(self, message=INVALID_MESSAGE, code='invalid_caption'):
        super().__init__(message)
        self.code = code


_QUOTED = re.compile(r'"[^"\n]*"|“[^”\n]*”')
_REFUSAL = re.compile(
    r"\b(?:i|we)\s+(?:(?:am|are)\s+(?:sorry\s*,?\s*)?)?"
    r"(?:cannot|can['’]t|will\s+not|won['’]t|do\s+not|don['’]t|"
    r"(?:am\s+)?(?:unable|not\s+able)|(?:must|have\s+to)\s+(?:decline|refuse))\b"
    r"|\b(?:i|we)['’](?:m|re)\s+(?:sorry\s*,?\s*)?(?:unable|not\s+able)\b"
    r"|\b(?:i|we)\s+(?:must|have\s+to)\s+(?:decline|refuse)\b"
    r"|\b(?:i|we)\s+can\s*,?\s+however\b", re.I)
_COMMENTARY = re.compile(
    r"^\s*(?:sorry\b|i\s+(?:can\s+help|can\s+offer|understand|apologize)\b|"
    r"let['’]s\s+(?:try|create|make)|would\s+you\s+like\b|"
    r"here(?:['’]s|\s+is)\s+(?:an?\s+)?(?:alternative|safer|different)\b)", re.I)
_PREAMBLE = re.compile(
    r"^\s*(?:(?:sure|certainly)[,!\.\s—-]*)?"
    r"here(?:['’]s|\s+is)\s+(?:the|your|an?)\s+"
    r"(?:expanded\s+|detailed\s+|image\s+)?(?:caption|description|prompt)\s*:\s*", re.I)


def image_caption(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise CaptionOutputError()
    text = raw.strip()
    if any(token in text.lower() for token in ['<think>', '</think>', '<|im_start|>', '<|im_end|>']):
        raise CaptionOutputError()
    # Quoted visible lettering can legitimately contain words such as "I can't".
    prose = _QUOTED.sub('"[visible text]"', text)
    if _REFUSAL.search(prose):
        raise CaptionOutputError(DECLINED_MESSAGE, 'caption_declined')
    if _COMMENTARY.search(prose):
        raise CaptionOutputError()
    text = _PREAMBLE.sub('', text, count=1).strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != '```' or lines[0].strip() not in ['```', '```text']:
            raise CaptionOutputError()
        text = '\n'.join(lines[1:-1]).strip()
    prose = _QUOTED.sub('"[visible text]"', text)
    if not text or _COMMENTARY.search(prose):
        raise CaptionOutputError()
    if _REFUSAL.search(prose):
        raise CaptionOutputError(DECLINED_MESSAGE, 'caption_declined')
    return text


def display_caption(raw: str) -> str:
    """Hide stale model messages in restored workspaces and historical PNGs."""
    try:
        return image_caption(raw)
    except CaptionOutputError:
        return ''
