"""Critic：检索结果置信校验与查询改写。

职责单一：拿到工具返回后判断"够不够好"，不好就给出改写建议。
不调用检索工具、不生成最终回答。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("imgsearch_agent.critic")


@dataclass
class Critique:
    passed: bool
    confidence: float
    reason: str


class Critic:
    def __init__(self, client=None, threshold: float = 0.25, max_retries: int = 2):
        self.client = client
        self.threshold = threshold
        self.max_retries = max_retries

    # ── 校验 ──────────────────────────────────────────────────────────────

    def check(self, result: dict[str, Any]) -> Critique:
        """判断检索结果是否达到置信阈值。"""
        images = result.get("images") or []
        if not images:
            return Critique(False, 0.0, "没有检索到任何图片")

        confidence = float(result.get("confidence", 0.0))
        if confidence < self.threshold:
            return Critique(
                False,
                confidence,
                f"平均相似度 {confidence:.3f} 低于阈值 {self.threshold}",
            )
        return Critique(True, confidence, f"平均相似度 {confidence:.3f}，通过")

    # ── 改写 ──────────────────────────────────────────────────────────────

    def rewrite(self, query: str, reason: str) -> str:
        """低置信时改写查询。没有 LLM 客户端则原样返回。"""
        if self.client is None:
            return query

        try:
            prompt = (
                f"原始查询：{query}\n"
                f"检索问题：{reason}\n"
                "请改写这个查询，使其更容易检索到相关图片。"
                "只输出改写后的查询文本，不要解释。"
            )
            new_query = self.client.chat(
                [{"role": "user", "content": prompt}], max_tokens=64
            ).strip()
            return new_query or query
        except Exception as e:
            logger.warning("查询改写失败: %s", e)
            return query
