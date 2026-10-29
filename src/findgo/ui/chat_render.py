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
    """气泡内部内容：文本 + 路径高亮 + 图片网格 + 超出入口 + 选图理由。"""
    safe = highlight_paths(text).replace("\n", "<br/>")
    if not images:
        return safe

    parts = []
    reasons: list[str] = []
    shown = 0
    for img in images[:max_thumbs]:
        if isinstance(img, str):
            path = img
            reason = ""
        elif isinstance(img, dict):
            path = img.get("path", "")
            reason = str(img.get("reason", "") or "")
        else:
            path = getattr(img, "path", "")
            reason = str(getattr(img, "reason", "") or "")
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

        title = f' title="{html.escape(reason)}"' if reason else ""
        parts.append(
            '<a href="' + full_uri + '"' + title + ">"
            '<img src="' + thumb_uri + '" width="112" height="112" '
            'style="margin:5px;border-radius:8px;vertical-align:middle;"></a>'
        )
        if reason and reason not in reasons:
            reasons.append(reason)
        shown += 1

    if not parts:
        return safe

    thumbs = '<div style="margin-top:10px;margin-bottom:6px;">' + "".join(parts) + "</div>"

    # 前几条理由直接用文字说出来，比只放在 tooltip 里更易读
    if reasons:
        bullets = "".join(
            '<div style="color:#6B6B6B;font-size:11.5px;line-height:1.7;">· '
            + html.escape(r)
            + "</div>"
            for r in reasons[:3]
        )
        thumbs += '<div style="margin-top:2px;">' + bullets + "</div>"

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
