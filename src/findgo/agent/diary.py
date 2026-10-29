"""日记读写。

`/diary` 生成的是**助手视角**的工作日记：
- 写的是「我今天帮主人做了什么」
- 不是主人的日记，用助手的口吻

这里同时负责把最近几天的日记读回来，让 Agent 记得自己写过什么。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_CHARS = 2000


def diary_dir(root: Path | None = None) -> Path:
    if root is None:
        try:
            from ..retriever.runtime import PROJECT_ROOT

            root = Path(PROJECT_ROOT)
        except Exception:
            root = Path.cwd()
    return Path(root) / "data" / "diary"


def recent_diary_text(days: int = 3, max_chars: int = MAX_CHARS) -> str:
    """取最近 N 天的日记，按日期从旧到新拼接。没有内容返回空字符串。"""
    folder = diary_dir()
    if not folder.is_dir():
        return ""

    today = datetime.now().date()
    picked: list[tuple[str, str]] = []

    for offset in range(days - 1, -1, -1):
        day = today - timedelta(days=offset)
        path = folder / f"{day.isoformat()}.md"
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.debug("读取日记失败 %s: %s", path, e)
            continue
        if text:
            picked.append((day.isoformat(), text))

    if not picked:
        return ""

    blocks = []
    budget = max_chars
    # 从最新的往前装，超预算就停，保证最近几天优先
    for day, text in reversed(picked):
        body = text[:budget]
        blocks.append(f"[{day}]\n{body}")
        budget -= len(body)
        if budget <= 0:
            break

    blocks.reverse()
    return "\n\n".join(blocks)


def diary_summary_line(days: int = 3) -> str:
    """一句话概括最近几天写过日记，供状态栏显示。"""
    folder = diary_dir()
    if not folder.is_dir():
        return ""

    today = datetime.now().date()
    found = []
    for offset in range(days):
        day = today - timedelta(days=offset)
        if (folder / f"{day.isoformat()}.md").is_file():
            found.append(day.isoformat())

    if not found:
        return ""
    return f"最近 {len(found)} 天有日记记录"


__all__ = ["diary_dir", "diary_summary_line", "recent_diary_text"]
