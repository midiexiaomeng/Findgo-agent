"""Responder：把工具结果合成自然语言回答。

有 LLM 就用 LLM 总结，没有或失败则用模板兜底。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("imgsearch_agent.responder")


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

        template = self.prompts.get("summarize_template", "")
        if template:
            try:
                prompt = template.format(
                    query=query, summary=summary or f"{len(images)} 张图片"
                )
            except (KeyError, IndexError):
                prompt = f"用户请求：{query}\n检索结果：{summary}"
        else:
            prompt = (
                f"用户请求：{query}\n"
                f"检索结果：{summary or f'{len(images)} 张图片'}\n"
                "请用一两句中文总结。"
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
