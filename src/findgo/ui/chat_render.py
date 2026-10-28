"""对话消息渲染：路径高亮、图片网格、气泡 HTML。"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Callable

from .thumbnail_cache import thumb_path

_PATH_PATTERN = re.compile(r"([A-Za-z]:\\[^\s<>\"']+|(?<![\w])/[\w./-]{3,})")


def highlight_paths(text: str) -> str:
    """把路径这类技术文本做淡绿高亮。"""
    escaped = html.escape(str(text))
    return _PATH_PATTERN.sub(
        '<span style="color:#4E7A63;background:#EAF3EE;'
        'padding:1px 4px;border-radius:3px;">' + chr(92) + "1</span>",
        escaped,
    )


def message_html(
    text: str,
    images: list | None,
    galleries: dict,
    thumb_fn: Callable[[str], str] = thumb_path,
    max_thumbs: int = 20,
) -> str:
    """气泡内部内容：文本 + 路径高亮 + 图片网格 + 超出入口。"""
    safe = highlight_paths(text).replace("\n", "<br/>")
    if not images:
        return safe

    parts = []
    shown = 0
    for img in images[:max_thumbs]:
        if isinstance(img, str):
            path = img
        elif isinstance(img, dict):
            path = img.get("path", "")
        else:
            path = getattr(img, "path", "")
        if not path or not Path(path).exists():
            continue
        thumb = thumb_fn(path)
        if not thumb:
            continue
        try:
            full_uri = Path(path).resolve().as_uri()
            thumb_uri = Path(thumb).resolve().as_uri()
        except ValueError:
            continue
        parts.append(
            '<a href="' + full_uri + '">'
            '<img src="' + thumb_uri + '" width="112" height="112" '
            'style="margin:5px;border-radius:8px;vertical-align:middle;"></a>'
        )
        shown += 1

    if not parts:
        return safe

    thumbs = '<div style="margin-top:10px;margin-bottom:6px;">' + "".join(parts) + "</div>"
    rest = len(images) - shown
    if rest > 0:
        gid = f"g{len(galleries)}"
        galleries[gid] = list(images)
        thumbs += (
            '<div style="text-align:right;margin-top:4px;">'
            '<a href="gallery:' + gid + '" style="color:#4E7A63;'
            'text-decoration:none;font-size:15px;font-weight:600;">…</a></div>'
        )
    return safe + thumbs
