"""Responder：把工具结果合成自然语言回答。

有 LLM 就用 LLM 总结，没有或失败则用模板兜底。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("findgo.responder")


class Responder:
    def __init__(self, client=None, prompts: dict[str, str] | None = None):
        self.client = client
        self.prompts = prompts or {}

    def compose(
        self,
        query: str,
        result: dict[str, Any],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        images = result.get("images") or []
        summary = result.get("summary") or ""

        if self.client is None:
            return self._fallback(images, summary)

        count = len(images)
        if count == 0:
            prompt = (
                f"用户想找：{query}\n"
                "系统没有找到相关图片。\n"
                "请用一句自然、口语化的中文告诉用户，并建议换个说法再试。"
                "不要提召回数量、耗时等技术细节。"
            )
        else:
            prompt = (
                f"用户想找：{query}\n"
                f"系统找到了 {count} 张相关图片。\n"
                "请用一句自然、口语化的中文告诉用户结果，像朋友聊天一样。"
                "不要提召回数量、候选数、精排、耗时等技术细节。"
            )

        try:
            reply = self.client.chat(
                [
                    {"role": "system", "content": "你是简洁的检索助手，用中文回答。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=128,
            ).strip()
            return reply or self._fallback(images, summary)
        except Exception as e:
            logger.warning("回复合成失败: %s", e)
            return self._fallback(images, summary)

    @staticmethod
    def _fallback(images: list, summary: str) -> str:
        if not images:
            return "没有找到相关图片，可以换个描述再试。"
        text = f"找到 {len(images)} 张相关图片。"
        if summary:
            text += f"（{summary}）"
        return text
