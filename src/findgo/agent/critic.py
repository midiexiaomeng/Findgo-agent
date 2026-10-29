"""Critic：检索结果的多维度校验与查询改写。

三个维度分别看：
- 数量：用户要 5 张，只给了 2 张
- 约束：must 里的条件是否命中
- 置信：平均相似度是否过低

只做判断，不调工具、不生成回答。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("findgo.critic")

#: 数量缺口容忍度：差 1 张不算问题
COUNT_TOLERANCE = 1


@dataclass
class Critique:
    passed: bool
    confidence: float
    reason: str
    #: 具体问题，按严重程度排序
    issues: list[str] = field(default_factory=list)
    #: 少了多少张
    missing_count: int = 0
    #: 没有满足的 must 约束
    violated: list[str] = field(default_factory=list)
    #: 是否值得改写重试
    retryable: bool = True

    @property
    def severity(self) -> str:
        if self.passed:
            return "ok"
        if not self.retryable:
            return "fatal"
        return "warn"


class Critic:
    def __init__(self, client=None, threshold: float = 0.25, max_retries: int = 2):
        self.client = client
        self.threshold = threshold
        self.max_retries = max_retries

    # ── 校验 ──────────────────────────────────────────────────────────────

    def check(self, result: dict[str, Any], task=None) -> Critique:
        """按数量、约束、置信三个维度判断结果够不够好。

        Args:
            result: 工具返回，含 images / confidence
            task: TaskSpec，用于取要几张、有哪些 must 约束
        """
        images = result.get("images") or []
        confidence = float(result.get("confidence", 0.0))

        if not images:
            return Critique(
                passed=False,
                confidence=0.0,
                reason="没有检索到任何图片",
                issues=["零结果"],
                retryable=True,
            )

        issues: list[str] = []
        violated: list[str] = []
        missing_count = 0

        # ── 维度一：数量 ──
        want = self._wanted_count(task)
        if want > 0:
            gap = want - len(images)
            if gap > COUNT_TOLERANCE:
                missing_count = gap
                issues.append(f"要 {want} 张，实际 {len(images)} 张，差 {gap} 张")

        # ── 维度二：must 约束 ──
        must = self._must_terms(task)
        if must:
            captions = " ".join(
                str(img.get("caption", "") or "") for img in images if isinstance(img, dict)
            )
            if captions.strip():
                from .constraints import matches_term

                violated = [t for t in must if not matches_term(captions, t)]
                if violated:
                    issues.append(f"没有命中必需条件：{'、'.join(violated)}")

        # ── 维度三：置信度 ──
        if confidence < self.threshold:
            issues.append(f"平均相似度 {confidence:.3f} 低于阈值 {self.threshold}")

        if not issues:
            return Critique(
                passed=True,
                confidence=confidence,
                reason=f"平均相似度 {confidence:.3f}，数量与约束均满足",
            )

        # 只有「数量不够」时改写查询有用；约束不命中改写也很难救
        retryable = bool(missing_count) or confidence < self.threshold
        return Critique(
            passed=False,
            confidence=confidence,
            reason="；".join(issues),
            issues=issues,
            missing_count=missing_count,
            violated=violated,
            retryable=retryable,
        )

    # ── 取参数 ────────────────────────────────────────────────────────────

    @staticmethod
    def _wanted_count(task) -> int:
        if task is None:
            return 0
        try:
            return int(getattr(task, "top_k", 0) or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _must_terms(task) -> list[str]:
        if task is None:
            return []
        constraints = getattr(task, "constraints", None) or {}
        if not isinstance(constraints, dict):
            return []
        must = constraints.get("must") or []
        return [str(x).strip() for x in must if str(x).strip()]

    # ── 改写 ──────────────────────────────────────────────────────────────

    def rewrite(self, query: str, reason: str, critique: Critique | None = None) -> str:
        """低置信或数量不足时改写查询。没有 LLM 客户端则原样返回。"""
        if self.client is None:
            return query

        # 数量不够时，扩大描述比换词更有效
        hint = ""
        if critique is not None and critique.missing_count > 0:
            hint = (
                "\n这次只找到部分结果，请给出更宽泛的同义描述，"
                "覆盖更多可能的说法，不要限定过窄的场景。"
            )
        elif critique is not None and critique.violated:
            hint = (
                f"\n用户要求包含：{'、'.join(critique.violated)}。"
                "请把条件写得更明确，但不要堆砌无关修饰。"
            )

        try:
            prompt = (
                f"原始查询：{query}\n"
                f"检索问题：{reason}\n"
                "请改写这个查询，使其更容易检索到相关图片。"
                "只输出改写后的查询文本，不要解释。" + hint
            )
            new_query = self.client.chat(
                [{"role": "user", "content": prompt}], max_tokens=64
            ).strip()
            return new_query or query
        except Exception as e:
            logger.warning("查询改写失败: %s", e)
            return query
