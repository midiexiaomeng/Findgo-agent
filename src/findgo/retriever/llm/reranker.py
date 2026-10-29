"""第三级：LLM 语义重排。

向量检索解决"长得像"，LLM 重排解决"语义对"。

例如查询"一个人在室内工作"，CLIP 可能把"室内摆着一台电脑"排得很高，
但它没有人。BLIP 描述 + Qwen 判断能把这种偏差纠正过来。

工程上的三个关键取舍：
1. 进 LLM 的候选必须先截断（默认 16 个）。50 个候选塞进 7B 模型，prompt
   太长、太慢，而且小模型在长列表上的排序能力会明显下降。
2. 分批打分而不是让模型直接输出整个排序。多候选排序对 7B 模型不稳定，
   绝对分数打分的鲁棒性高得多。
3. 向量分做 min-max 归一化后再和 LLM 分融合。CLIP 的余弦相似度实际分布
   很窄（0.15~0.45），不归一化的话 0~1 的 LLM 分会完全压倒向量信号。
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Sequence

from ..config import Config
from ..vectorstore.base import VectorHit, normalize_scores
from .client import LLMClient, LLMError

logger = logging.getLogger(__name__)


@dataclass
class ScoredCandidate:
    """重排后的单个候选，保留两路分数与理由，便于解释和调试。"""

    hit: VectorHit
    vec_score: float  # 原始余弦相似度
    vec_norm: float  # min-max 归一化后的向量分
    llm_score: float | None  # 0~1，None 表示这一条没拿到 LLM 分
    final_score: float  # 融合分
    reason: str = ""  # LLM 给出的简短理由
    dropped_by: str = ""  # 被哪条硬约束淘汰

    @property
    def id(self) -> str:
        return self.hit.id

    @property
    def metadata(self) -> dict:
        return self.hit.metadata


@dataclass
class RerankReport:
    used: bool = False
    backend: str = ""
    model: str = ""
    candidates: int = 0
    llm_scored: int = 0
    batches: int = 0
    elapsed: float = 0.0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.used:
            return "未使用 LLM 重排"
        return (
            f"LLM 重排 {self.llm_scored}/{self.candidates} 条，"
            f"{self.batches} 批，耗时 {self.elapsed:.1f}s"
        )


class LLMReranker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = LLMClient(cfg)

    # ── 对外 ────────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        hits: Sequence[VectorHit],
        top_k: int | None = None,
        use_llm: bool = True,
        should_stop=None,
        constraints: dict | None = None,
    ) -> tuple[list[ScoredCandidate], RerankReport]:
        """对候选做约束过滤与语义重排。

        Args:
            query: 语义查询。以文搜图时是 core_query；以图搜图时是查询图的 BLIP 描述。
            hits: 向量检索结果，按相似度降序。
            top_k: 最终返回数量，默认取配置。
            constraints: 结构化约束（must / must_not / should）。
        """
        top_k = int(top_k or self.cfg.search.top_k)
        started = time.time()
        report = RerankReport(
            backend=self.client.backend,
            model=self.client.model,
        )

        # 硬约束先淘汰，被扣掉的候选不参与后续打分
        kept = list(hits)
        dropped: list[ScoredCandidate] = []
        if constraints:
            must = [str(x) for x in (constraints.get("must") or []) if str(x).strip()]
            must_not = [str(x) for x in (constraints.get("must_not") or []) if str(x).strip()]
            if must or must_not:
                kept, dropped = self._apply_hard_constraints(hits, must, must_not)
                if dropped:
                    report.errors.append(f"约束淘汰 {len(dropped)} 张")

        if not kept:
            # 约束过严会把候选全灭，宁可不过滤也不要给出空结果
            logger.info("约束淘汰了全部 %d 个候选，回退到不过滤", len(dropped))
            report.errors.append("约束淘汰了全部候选，已回退")
            kept = list(hits)
            dropped = []

        logger.info(
            "重排输入: 共 %d 条，硬过滤保留 %d，淘汰 %d",
            len(hits),
            len(kept),
            len(dropped),
        )

        if not kept:
            report.elapsed = time.time() - started
            return [], report

        # 重排范围：候选不多时全部精排，多时只排头部。
        # 不写死固定值，避免「正好 17 条时第 17 条永远拿不到语义分」。
        base = max(1, int(self.cfg.rerank.max_candidates))
        head = len(kept) if len(kept) <= base * 2 else base
        candidates = list(kept)[:head]
        report.candidates = len(candidates)

        tail = []
        if top_k > head:
            tail = [
                ScoredCandidate(
                    hit=h,
                    vec_score=float(h.score),
                    vec_norm=0.0,
                    llm_score=None,
                    final_score=0.0,
                )
                for h in list(kept)[head:top_k]
            ]

        raw_scores = [float(h.score) for h in candidates]
        norms = normalize_scores(raw_scores)
        scored = [
            ScoredCandidate(
                hit=h,
                vec_score=raw,
                vec_norm=norm,
                llm_score=None,
                final_score=norm,
            )
            for h, raw, norm in zip(candidates, raw_scores, norms)
        ]

        # ── 不该/不能重排的情况：退化为纯向量排序 ──
        if not use_llm:
            report.errors.append("已关闭 LLM 重排，按向量相似度排序")
            report.elapsed = time.time() - started
            return self._finish(scored, top_k, tail), report

        if not self.cfg.rerank.enabled or self.client.backend in ("none", ""):
            report.elapsed = time.time() - started
            return self._finish(scored, top_k, tail), report

        if not query or not query.strip():
            report.errors.append("查询语义为空，跳过 LLM 重排")
            report.elapsed = time.time() - started
            return self._finish(scored, top_k, tail), report

        # 没有任何一条有描述时，LLM 只能看文件名，重排基本是噪声
        with_caption = sum(1 for c in candidates if str(c.metadata.get("caption", "")).strip())
        if with_caption == 0:
            report.errors.append("候选图片没有描述（索引时未启用 BLIP），跳过 LLM 重排")
            report.elapsed = time.time() - started
            return self._finish(scored, top_k, tail), report

        # 服务根本没起来时，不要逐批重试。一次探测就能确定，
        # 否则 3 批候选会白等十几秒。
        available, reason = self.client.is_available()
        if not available:
            report.errors.append(f"LLM 不可用: {reason}")
            report.elapsed = time.time() - started
            return self._finish(scored, top_k, tail), report

        # ── 分批打分 ──
        llm_scores, used, errors = self._score_batches(
            query, candidates, should_stop, constraints=constraints
        )
        report.errors.extend(errors)
        report.llm_scored = len(llm_scores)
        report.used = used
        report.batches = (
            len(candidates) + self.cfg.rerank.batch_size - 1
        ) // self.cfg.rerank.batch_size

        if llm_scores:
            for cand in scored:
                entry = llm_scores.get(cand.hit.id)
                if entry is not None:
                    value = float(entry.get("score", 0.0))
                    cand.llm_score = value
                    cand.reason = str(entry.get("reason", ""))
                    cand.final_score = self._fuse(cand.vec_norm, value)

        report.elapsed = time.time() - started
        if report.used:
            logger.info(report.summary())
        return self._finish(scored, top_k, tail), report

    # ── 约束过滤 ────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_hard_constraints(
        hits: Sequence[VectorHit],
        must: Sequence[str],
        must_not: Sequence[str],
    ) -> tuple[list[VectorHit], list[ScoredCandidate]]:
        """按 must_not 硬淘汰，must 只记录命中情况。

        must 不淘汰的理由：BLIP 描述经常不写颜色、动作等属性，
        用它做硬条件会把大量合格图误杀。must 命中会在重排 prompt
        里作为强提示，由模型综合判断。

        没有描述的图片不参与硬过滤：无法判断就不淘汰。
        """
        from ...agent.constraints import matches_term

        kept: list[VectorHit] = []
        dropped: list[ScoredCandidate] = []

        for hit in hits:
            caption = str(hit.metadata.get("caption", "") or "")
            if not caption.strip():
                kept.append(hit)
                continue

            blocked = next((t for t in must_not if matches_term(caption, t)), None)
            if blocked:
                dropped.append(
                    ScoredCandidate(
                        hit=hit,
                        vec_score=float(hit.score),
                        vec_norm=0.0,
                        llm_score=0.0,
                        final_score=0.0,
                        reason=f"命中排除条件：{blocked}",
                        dropped_by=blocked,
                    )
                )
                continue

            kept.append(hit)

        return kept, dropped

    # ── 打分 ────────────────────────────────────────────────────────────────

    def _score_batches(
        self,
        query: str,
        candidates: Sequence[VectorHit],
        should_stop=None,
        constraints: dict | None = None,
    ) -> tuple[dict[str, dict], bool, list[str]]:
        scores: dict[str, dict] = {}
        errors: list[str] = []
        bs = max(1, int(self.cfg.rerank.batch_size))
        consecutive_failures = 0

        for start in range(0, len(candidates), bs):
            if should_stop is not None and should_stop():
                errors.append("已手动停止")
                break
            batch = candidates[start : start + bs]
            payload = self._build_prompt(query, batch, constraints)
            answer = self._call_with_retry(payload)

            if answer is None:
                consecutive_failures += 1
                errors.append(f"第 {start // bs + 1} 批调用失败")
                # 连续两批失败，说明服务本身有问题，继续调用只是浪费时间
                if consecutive_failures >= 2:
                    break
                continue

            consecutive_failures = 0
            parsed = self._parse_scores(answer, len(batch))
            for local_idx, value in parsed.items():
                if 0 <= local_idx < len(batch):
                    scores[batch[local_idx].id] = value

        return scores, bool(scores), errors

    def _call_with_retry(self, messages: list[dict[str, str]]) -> str | None:
        attempts = max(1, int(self.cfg.rerank.retries) + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return self.client.chat(
                    messages,
                    json_mode=True,
                    max_tokens=int(self.cfg.rerank.max_tokens),
                )
            except LLMError as e:
                last_error = e
                logger.warning("LLM 调用失败（第 %d/%d 次）: %s", attempt + 1, attempts, e)
                if attempt + 1 < attempts:
                    time.sleep(1.0)
            except Exception as e:  # 防御性兜底，避免整条搜索链路挂掉
                last_error = e
                logger.warning("LLM 调用异常: %s", e)
                break
        if last_error:
            logger.warning("本批重排放弃: %s", last_error)
        return None

    def _build_prompt(
        self,
        query: str,
        batch: Sequence[VectorHit],
        constraints: dict | None = None,
    ) -> list[dict[str, str]]:
        lines = []
        for i, hit in enumerate(batch, start=1):
            filename = str(hit.metadata.get("filename", f"image_{i}"))
            caption = str(hit.metadata.get("caption", "") or "").strip()
            if not caption:
                caption = "（无描述）"
            lines.append(f"[{i}] 文件：{filename}\n    描述：{caption}")

        must = [str(x) for x in (constraints or {}).get("must", []) if str(x).strip()]
        must_not = [str(x) for x in (constraints or {}).get("must_not", []) if str(x).strip()]
        should = [str(x) for x in (constraints or {}).get("should", []) if str(x).strip()]

        rule_lines = []
        if must:
            rule_lines.append("必须满足（不满足就淘汰，给 0 分）：" + "、".join(must))
        if must_not:
            rule_lines.append("必须排除（命中就给 0 分）：" + "、".join(must_not))
        if should:
            rule_lines.append("加分项（有则提分，没有不淘汰）：" + "、".join(should))

        system = (
            "你是图像检索系统的相关性评估器。你只输出 JSON，不输出任何解释、"
            "markdown 代码块或多余文字。"
        )
        user = (
            f"用户想找的图片：{query}\n"
            + ("\n".join(rule_lines) + "\n" if rule_lines else "")
            + "\n下面是候选图片的内容描述（由图像描述模型生成，可能是英文，请照常理解）：\n\n"
            + "\n".join(lines)
            + "\n\n请先检查约束，再按语义相关性打 0-100 分：\n"
            "  90-100 主体和场景都高度吻合，且满足全部约束\n"
            "  60-89  主体相关，但场景/动作/细节有偏差\n"
            "  30-59  只有部分元素相关\n"
            "  0-29   基本无关，或违反了必须排除的条件\n"
            "  0      违反了必须满足的条件\n\n"
            "同时用 10 个字以内说明理由，指出是否命中约束。\n"
            '只输出这个格式的 JSON：{"scores":[{"index":1,"score":85,"reason":"白色猫躺着"}]}，'
            f"必须覆盖全部 {len(batch)} 张图，index 从 1 开始。"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # ── 解析 ────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_scores(text: str, expected: int) -> dict[int, dict]:
        """把模型输出解析成 {批内下标(0-based): {"score": 0~1, "reason": str}}。"""
        if not text:
            return {}

        cleaned = text.strip()
        # 去掉 ```json ``` 围栏
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        # 先尝试整体解析（模型通常只输出 JSON），失败再提取片段。
        # 顺序很重要：纯数组必须以数组解析，否则 [{"id":1}] 会先被 \{.*\}
        # 抓成单个对象，丢掉外层结构。
        data = None
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            candidates = []
            if cleaned.lstrip().startswith("["):
                arr_match = re.search(r"\[.*\]", cleaned, re.S)
                if arr_match:
                    candidates.append(arr_match.group(0))
            obj_match = re.search(r"\{.*\}", cleaned, re.S)
            if obj_match:
                candidates.append(obj_match.group(0))
            arr_match = re.search(r"\[.*\]", cleaned, re.S)
            if arr_match:
                candidates.append(arr_match.group(0))

            for raw in candidates:
                try:
                    data = json.loads(raw)
                    break
                except json.JSONDecodeError:
                    continue
        if data is None:
            # 退路：直接抓 "index": n, "score": s 这种片段
            pairs = re.findall(r'"index"\s*:\s*(\d+)\s*,\s*"score"\s*:\s*(\d+(?:\.\d+)?)', cleaned)
            if not pairs:
                logger.warning("无法解析 LLM 打分结果: %s", cleaned[:200])
                return {}
            data = [{"index": int(i), "score": float(s)} for i, s in pairs]

        out: dict[int, dict] = {}

        def put(idx: int, score: float, reason: str = "") -> None:
            if idx < 1 or idx > expected:
                return
            if score > 1.0:  # 0-100 制
                score = score / 100.0
            out[idx - 1] = {
                "score": max(0.0, min(1.0, float(score))),
                "reason": str(reason or "").strip()[:40],
            }

        if isinstance(data, dict):
            # 单条对象形式 {"index":1,"score":85,"reason":"..."}
            if "score" in data or "relevance" in data:
                try:
                    put(
                        int(data.get("index", data.get("id", data.get("rank", 1)))),
                        float(data.get("score", data.get("relevance"))),
                        str(data.get("reason", data.get("why", ""))),
                    )
                except (TypeError, ValueError):
                    pass
                return out

            items = data.get("scores") or data.get("results") or data.get("data")
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        idx = item.get("index", item.get("id", item.get("rank")))
                        score = item.get("score", item.get("relevance"))
                        if idx is None or score is None:
                            continue
                        try:
                            put(
                                int(idx),
                                float(score),
                                str(item.get("reason", item.get("why", ""))),
                            )
                        except (TypeError, ValueError):
                            continue
            else:
                # {"1": 85, "2": 70} 形式
                for key, value in data.items():
                    try:
                        put(int(key), float(value))
                    except (TypeError, ValueError):
                        continue
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    idx = item.get("index", item.get("id", item.get("rank")))
                    score = item.get("score", item.get("relevance"))
                    if idx is None or score is None:
                        continue
                    try:
                        put(
                            int(idx),
                            float(score),
                            str(item.get("reason", item.get("why", ""))),
                        )
                    except (TypeError, ValueError):
                        continue

        if not out:
            logger.warning("LLM 返回了 JSON 但没有可用分数: %s", cleaned[:200])
        return out

    # ── 融合 ────────────────────────────────────────────────────────────────

    def _fuse(self, vec_norm: float, llm_score: float) -> float:
        fusion = self.cfg.search.fusion
        wv = max(0.0, float(fusion.vector_weight))
        wl = max(0.0, float(fusion.llm_weight))
        total = wv + wl
        if total <= 0:
            return llm_score
        return (wv * vec_norm + wl * llm_score) / total

    @staticmethod
    def _finish(
        scored: list[ScoredCandidate],
        top_k: int,
        tail: list[ScoredCandidate] | None = None,
    ) -> list[ScoredCandidate]:
        ordered = sorted(
            scored,
            key=lambda c: (
                c.final_score,
                c.llm_score if c.llm_score is not None else -1.0,
                c.vec_score,
            ),
            reverse=True,
        )
        out = ordered[:top_k]
        if tail and len(out) < top_k:
            out.extend(tail[: top_k - len(out)])
        return out
