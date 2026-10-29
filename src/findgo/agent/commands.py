"""斜杠命令的实际执行。

命令不是填提示词，而是直接干活：
/diary 生成日记文件、/compact 压缩上下文、/xing 提炼技能草稿、/loop 重复执行。
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CommandError(RuntimeError):
    """命令执行失败，消息直接展示给用户。"""


def split_command(text: str) -> tuple[str, str]:
    """把 `/diary 今天心情不错` 拆成 ('/diary', '今天心情不错')。"""
    stripped = text.strip()
    if not stripped.startswith("/"):
        return "", stripped

    parts = stripped.split(maxsplit=1)
    name = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ""
    return name, args


def is_command(text: str) -> bool:
    from .ui_command_names import COMMAND_NAMES

    name, _ = split_command(text)
    return name in COMMAND_NAMES


# ── 各命令实现 ────────────────────────────────────────────────────────────


def run_diary(agent, args: str, session_id: str = "") -> str:
    """/diary：以助手的第一人称写今天的工作日记，写入 data/diary/日期.md"""
    history = _today_history(agent)
    if not history:
        raise CommandError("今天还没有聊过什么，先聊几句再写日记。")

    transcript = _format_transcript(history, limit=200)
    extra = f"\n\n补充：{args}" if args else ""

    prompt = (
        "下面是我（助手）今天和主人一起做事的对话记录。\n"
        "请以**我（助手）的第一人称视角**写一篇简短的日记。\n\n"
        "视角要求（很重要）：\n"
        "1. 你是助手，不是主人。写的是「我帮主人做了什么」，不是「主人今天做了什么」。\n"
        "2. 称呼对方为「主人」或「你」，不要自称主人，不要用「我（用户）」的口吻。\n"
        "3. 可以写自己的观察和感受，比如某个需求不好处理、某个结果比预期好。\n\n"
        "内容硬规则：\n"
        "4. 只能写记录里真实出现过的事。记录里没有的天气、地点、人物、心情、"
        "买了什么、遇到谁、看到什么，一律不能写。\n"
        "5. 不允许虚构情节，也不要用套话填字数。记录少就写短，一句话也可以。\n"
        "6. 结尾不要写「明天要努力」这类空话，除非记录里确实提到了后续打算。\n\n"
        "格式：不要标题、不要分条、不要小标题，一段话写到底。\n\n"
        f"对话记录：\n{transcript}{extra}"
    )

    text = _ask_llm(agent, prompt, max_tokens=800)
    if not text:
        raise CommandError("生成日记失败，请检查模型是否可用。")

    today = datetime.now().strftime("%Y-%m-%d")
    path = _data_dir(agent) / "diary" / f"{today}.md"
    path.parent.mkdir(parents=True, exist_ok=True)

    # 同一天多次写就追加，不覆盖之前的内容
    stamp = datetime.now().strftime("%H:%M")
    block = f"\n\n## {stamp}\n\n{text.strip()}\n"
    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8") as f:
        if mode == "w":
            f.write(f"# {today}\n")
        f.write(block)

    return f"日记已写入 {path.relative_to(_project_root(agent))}\n\n{text.strip()}"


def run_compact(agent, args: str, session_id: str = "") -> str:
    """/compact：把当前会话压缩成摘要，写进记忆的 summaries 表。"""
    history = _safe_history(agent)
    if len(history) < 4:
        raise CommandError("对话太短，没什么可压缩的。")

    transcript = _format_transcript(history, limit=200)
    prompt = (
        "把下面这段对话压缩成摘要，用于后续对话的上下文。\n"
        "保留：用户的核心需求、已经确定的约束、已经完成的动作、还没解决的问题。\n"
        "丢掉：寒暄、重复、失败的尝试细节。\n"
        "300 字以内，中文，直接给摘要正文。\n\n"
        f"对话记录：\n{transcript}"
    )

    summary = _ask_llm(agent, prompt, max_tokens=600)
    if not summary:
        raise CommandError("压缩失败，请检查模型是否可用。")

    try:
        agent.memory.set_summary(summary.strip(), session_id=session_id or None)
    except Exception as e:
        raise CommandError(f"写入摘要失败：{e}") from e

    return f"上下文已压缩为 {len(summary)} 字摘要，后续对话会带上它。"


def run_xing(agent, args: str, session_id: str = "") -> str:
    """/xing：从对话里提炼一个可复用的技能草稿，存到 data/skills/draft-*.md"""
    history = _safe_history(agent)
    if len(history) < 4:
        raise CommandError("对话太短，提炼不出技能。多完成几轮任务再试。")

    transcript = _format_transcript(history, limit=120)
    focus = f"\n\n重点关注：{args}" if args else ""

    prompt = (
        "从下面这段对话里，提炼一个可复用的操作技能。\n"
        "输出格式（Markdown）：\n"
        "---\n"
        "name: 简短的英文标识\n"
        "description: 一句话说明这个技能解决什么问题\n"
        "---\n\n"
        "## 什么时候用\n\n## 具体步骤\n\n## 注意事项\n\n"
        "只写从对话里真实出现过的方法，不要编造。中文正文。\n\n"
        f"对话记录：\n{transcript}{focus}"
    )

    draft = _ask_llm(agent, prompt, max_tokens=900)
    if not draft:
        raise CommandError("提炼失败，请检查模型是否可用。")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _data_dir(agent) / "skills" / f"draft_{stamp}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(draft.strip() + "\n", encoding="utf-8")

    return f"技能草稿已保存到 {path.relative_to(_project_root(agent))}\n\n{draft.strip()[:600]}"


def run_loop(agent, args: str, session_id: str = "") -> str:
    """/loop：把后面的内容作为任务重复执行，格式 `/loop 3 找猫的图片`。"""
    if not args:
        raise CommandError("用法：/loop 次数 任务内容，例如 `/loop 3 找猫的图片`")

    count = 3
    target = args
    m = re.match(r"^(\d+)\s+(.+)$", args)
    if m:
        count = max(1, min(int(m.group(1)), 20))
        target = m.group(2).strip()

    if not target:
        raise CommandError("没有指定要循环的任务内容。")

    tasks = agent.plan_tasks(target)
    if not tasks:
        raise CommandError("没能把这个任务拆解成可执行的步骤。")

    done = 0
    paths: set[str] = set()
    for i in range(count):
        try:
            results = agent.run_tasks(tasks, write_memory=False)
        except Exception as e:
            logger.warning("循环第 %d 次失败: %s", i + 1, e)
            break

        got = [img for r in results for img in r.results]
        for img in got:
            p = img.get("path", "") if isinstance(img, dict) else getattr(img, "path", "")
            if p:
                paths.add(p)
        if got:
            done += 1

    if not paths:
        return f"循环 {count} 次，没有新增结果。"

    return f"循环 {count} 次，{done} 次有新结果，累计 {len(paths)} 张不重复的图片。"


def run_skill_command(agent, name: str, args: str, session_id: str = "") -> str:
    """skills 类命令：目前只给指引，等 skills 系统接通后替换成实际调用。"""
    if not args:
        return f"{name} 需要一段需求描述。\n" f"例如：`{name} 帮我做一个批量重命名的技能`"

    return (
        f"{name} 收到需求：{args}\n\n"
        "当前版本还没接通 skill 执行链路，"
        "可以先用 /xing 把相关操作提炼成技能草稿。"
    )


# ── 分派 ──────────────────────────────────────────────────────────────────


def dispatch(text: str, agent, session_id: str = "") -> str | None:
    """命中命令就执行并返回结果，不是命令返回 None。"""
    name, args = split_command(text)
    if not name:
        return None

    handlers = {
        "/diary": run_diary,
        "/compact": run_compact,
        "/xing": run_xing,
        "/loop": run_loop,
    }

    handler = handlers.get(name)
    if handler is not None:
        return handler(agent, args, session_id)

    if name in ("/hana-plugin-creator", "/office-documents"):
        return run_skill_command(agent, name, args, session_id)

    return None


# ── 工具函数 ──────────────────────────────────────────────────────────────


def _safe_history(agent) -> list[dict[str, str]]:
    try:
        return agent.memory.history()
    except Exception:
        return []


def _today_history(agent) -> list[dict[str, str]]:
    """取今天所有会话的消息。

    一天可能开几次应用，每次是独立会话。日记要记的是这一天，
    不是最后一次打开。读不到数据库时退回当前会话。
    """
    try:
        start = datetime.now().strftime("%Y-%m-%d") + "T00:00:00"
        rows = agent.memory.messages_since(start)
        if rows:
            return [{"role": r.get("role", ""), "content": r.get("content", "")} for r in rows]
    except Exception as e:
        logger.debug("读取今日消息失败，退回当前会话: %s", e)

    return _safe_history(agent)


def _format_transcript(history: list[dict[str, str]], limit: int = 60) -> str:
    lines = []
    for msg in history[-limit:]:
        role = "我" if msg.get("role") == "user" else "助手"
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"{role}：{content[:500]}")
    return "\n".join(lines)


def _ask_llm(agent, prompt: str, max_tokens: int = 800) -> str:
    try:
        client = agent.client
    except Exception as e:
        raise CommandError(f"没有可用的模型：{e}") from e

    try:
        reply = client.chat(
            [
                {"role": "system", "content": "你是一个简洁、务实的中文助手。"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
        )
    except Exception as e:
        raise CommandError(f"模型调用失败：{e}") from e

    return str(reply or "").strip()


def _project_root(agent) -> Path:
    try:
        from ..retriever.runtime import PROJECT_ROOT

        return Path(PROJECT_ROOT)
    except Exception:
        return Path.cwd()


def _data_dir(agent) -> Path:
    return _project_root(agent) / "data"


__all__ = [
    "CommandError",
    "dispatch",
    "is_command",
    "run_compact",
    "run_diary",
    "run_loop",
    "run_skill_command",
    "run_xing",
    "split_command",
]

_ = Any
