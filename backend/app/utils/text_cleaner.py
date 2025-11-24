from __future__ import annotations

import re

from typing import Callable

_STAGE_KEYWORDS = [
    "小声",
    "悄悄",
    "低声",
    "压低",
    "调皮",
    "翻白眼",
    "挑眉",
    "眨眼",
    "坏笑",
    "偷笑",
    "凑近",
    "靠近",
    "叹气",
    "撒娇",
    "转圈",
    "转身",
    "耸肩",
    "抱臂",
    "伸舌",
    "轻笑",
    "轻声",
    "大笑",
    "眯眼",
    "wink",
    "sigh",
]

_PAREN_PATTERN = re.compile(r"[（(]([^（）()\n]{1,60})[）)]")
_STAR_PATTERN = re.compile(r"\*([^*\n]{1,120})\*")


def _contains_stage_keyword(text: str) -> bool:
    lowered = text.lower()
    for keyword in _STAGE_KEYWORDS:
        if keyword in text or keyword.lower() in lowered:
            return True
    return False


def _sanitize_parenthetical(text: str, replacer: Callable[[re.Match[str]], str]) -> str:
    return _PAREN_PATTERN.sub(replacer, text)


def _sanitize_star(text: str, replacer: Callable[[re.Match[str]], str]) -> str:
    return _STAR_PATTERN.sub(replacer, text)


def strip_stage_directions(text: str) -> str:
    if not text:
        return text

    def remove_parenthetical(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if _contains_stage_keyword(inner):
            return ""
        return match.group(0)

    def clean_star(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if _contains_stage_keyword(inner):
            return ""
        return inner

    cleaned = _sanitize_parenthetical(text, remove_parenthetical)
    cleaned = _sanitize_star(cleaned, clean_star)
    return cleaned
