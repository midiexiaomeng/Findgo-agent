"""图像检索工具：把 retriever 的多模态检索封装成标准 Tool。

Agent 只看到 name / description / args_schema / run，
不关心底层是 CLIP + 向量库 + 重排。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..utils.path_guard import PathGuard, PathSecurityError
from ..utils.schemas import RetrievalResult, RetrievedImage
from .base_tool import BaseTool

logger = logging.getLogger("findgo.image_retrieval")


class ImageRetrievalArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="图片内容描述")
    top_k: int = Field(8, ge=1, le=1000, description="返回数量，最大 1000")
    folder_path: str = Field("", description="限定检索的文件夹路径，空表示全库")
    all_results: bool = Field(False, description="返回全部匹配结果，忽略 top_k")
    exclude_paths: list[str] = Field(
        default_factory=list, description="本轮要排除的图片路径（上一轮已返回过）"
    )
    dedupe: bool = Field(True, description="过滤掉彼此高度相似的重复图")
    dedupe_threshold: float = Field(0.95, ge=0.5, le=1.0, description="相似度阈值，超过视为重复")
    path: str = Field("", description="（兼容）同 folder_path")
    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="结构化约束：core_query / must / must_not / should / retrieve_count",
    )
    auto_index: bool = Field(
        False,
        description="目录未索引时是否先建库。默认关闭，建库应该单独触发",
    )
    count_only: bool = Field(
        False,
        description="只统计匹配数量，不返回图片列表",
    )


class ImageRetrievalTool(BaseTool):
    name = "image_retrieval"
    description = "用自然语言描述搜索本地图片库，返回图片路径、描述与相似度"
    args_schema = ImageRetrievalArgs

    #: 实时检索的图片数上限，超过就建库
    LIVE_LIMIT = 200
    #: 超过这个数建库明显划算（一次编码，后续检索秒级）
    INDEX_MIN = 200

    def __init__(self, pipeline=None, guard: PathGuard | None = None):
        self._pipeline = pipeline
        self._indexed: set[str] = set()
        self.guard = guard or PathGuard()
        # 建库进度回调，由界面注入；签名 (done, total, name)
        self._progress_hook = None
        self._stop_hook = None
        self._pause_hook = None

    def set_hooks(self, progress=None, should_stop=None, should_pause=None) -> None:
        """界面层注入进度、停止与暂停回调。"""
        self._progress_hook = progress
        self._stop_hook = should_stop
        self._pause_hook = should_pause

    @property
    def pipeline(self):
        if self._pipeline is None:
            # 延迟加载：只有真正执行检索时才初始化模型与向量库
            from ..retriever import build_pipeline

            self._pipeline = build_pipeline()
        return self._pipeline

    def _ensure_indexed(self, folder) -> None:
        """显式建库时才调用，不再由检索链路隐式触发。"""
        key = str(folder.resolve()).lower()
        if key in self._indexed:
            return

        logger.info("开始增量索引: %s", folder)
        last = {"pct": -1}

        def progress(done: int, total: int, name: str) -> None:
            pct = int(done * 100 / max(total, 1))
            if pct >= last["pct"] + 10 or done == total:
                last["pct"] = pct
                logger.info("索引进度 %d/%d (%d%%) %s", done, total, pct, name)
            # 转给界面进度条
            if self._progress_hook is not None:
                try:
                    self._progress_hook(done, total, name)
                except Exception:
                    pass

        try:
            stats = self.pipeline.index_folder(
                folder,
                recursive=True,
                skip_existing=True,
                progress=progress,
                should_stop=self._stop_hook,
                should_pause=self._pause_hook,
            )
            if stats.locked:
                logger.info("目录正在索引中，本次跳过")
                return
            logger.info("索引完成: %s", stats.summary())
        except Exception as e:
            logger.warning("索引 %s 失败: %s", folder, e)
            return
        self._indexed.add(key)

    # ── 实时检索（不入库）────────────────────────────────────────────────

    def _indexed_folders(self) -> set[str]:
        """当前向量库里已覆盖的目录（小写）。"""
        try:
            hits = self.pipeline.store.get(limit=50000)
        except Exception:
            return set()

        folders: set[str] = set()
        for hit in hits:
            path = getattr(hit, "path", "") or ""
            if path:
                folders.add(str(Path(path).parent).lower())
        return folders

    def _is_indexed(self, folder: Path) -> bool:
        """目录本身或其子目录里已有索引条目就算已覆盖。"""
        try:
            prefix = str(folder.resolve()).lower()
        except OSError:
            prefix = str(folder).lower()

        for known in self._indexed_folders():
            if (
                known == prefix
                or known.startswith(prefix + os.sep)
                or known.startswith(prefix + "/")
            ):
                return True
        return False

    def _live_search(
        self,
        folder: Path,
        query: str,
        top_k: int,
        should_stop=None,
        paths: list | None = None,
    ) -> dict[str, Any] | None:
        """小目录直接现算向量，不写库。

        适合临时一次性查询。超过 LIVE_LIMIT 张就返回 None，
        由调用方改走建库路径。
        """
        if paths is None:
            from ..utils.image_scan import scan_images

            try:
                paths = scan_images(folder, recursive=True)
            except Exception as e:
                logger.warning("实时扫描失败 %s: %s", folder, e)
                return None

        if not paths:
            return {
                "query": query,
                "images": [],
                "confidence": 0.0,
                "summary": f"目录里没有图片：{folder}",
            }

        if len(paths) > self.LIVE_LIMIT:
            return None

        logger.info("实时检索 %d 张（不入库）", len(paths))

        vectors = []
        kept: list[str] = []
        bs = 16
        for i in range(0, len(paths), bs):
            if should_stop is not None and should_stop():
                break
            batch = paths[i : i + bs]
            try:
                feats = self.pipeline.encoder.encode_images(batch)
            except Exception as e:
                logger.warning("实时编码失败: %s", e)
                continue
            for path, vec in zip(batch, feats):
                vectors.append(vec)
                kept.append(str(path))

        if not vectors:
            return {
                "query": query,
                "images": [],
                "confidence": 0.0,
                "summary": "实时编码失败，没有可用结果",
            }

        import numpy as np

        q = self.pipeline.encoder.encode_text(query)
        mat = np.stack(vectors)
        scores = mat @ q.reshape(-1)
        order = np.argsort(-scores)[: max(1, top_k)]

        images = [
            RetrievedImage(
                rank=i + 1,
                path=kept[idx],
                caption="",
                score=round(float(scores[idx]), 3),
                reason="实时检索（未入库）",
            )
            for i, idx in enumerate(order)
        ]
        avg = sum(i.score for i in images) / len(images) if images else 0.0
        return RetrievalResult(
            query=query,
            images=images,
            confidence=round(avg, 3),
            summary=(
                f"实时检索 {len(kept)} 张（目录未入库，结果不持久）" f" · 返回 {len(images)} 张"
            ),
        ).model_dump()

    def _dedupe(self, images, threshold: float):
        """按向量相似度去重，保留排名靠前的那张。"""
        import numpy as np

        from ..retriever.indexer import Indexer

        id_map = {Indexer.image_id(i.path): i for i in images}
        try:
            records = self.pipeline.store.get_vectors(ids=list(id_map.keys()))
        except Exception:
            return images

        vectors = {str(r[0]): np.asarray(r[1], dtype=np.float32) for r in records}

        kept = []
        kept_vectors: list[np.ndarray] = []
        for image in images:
            vector = vectors.get(Indexer.image_id(image.path))
            if vector is None:
                kept.append(image)
                continue

            norm = float(np.linalg.norm(vector))
            unit = vector / max(norm, 1e-8)
            if any(float(np.dot(unit, other)) >= threshold for other in kept_vectors):
                continue
            kept.append(image)
            kept_vectors.append(unit)

        return kept

    def _diversify(self, images, limit: int, lam: float = 0.72):
        """MMR 打散：既看相关度，也惩罚与已选图太像的候选。

        去重只能干掉几乎同一张图；这里解决「同一只猫的十个角度」。
        lam 越小越偏向多样性，0.72 是相关性与新颖度的折中。
        """
        import numpy as np

        from ..retriever.indexer import Indexer

        if len(images) <= 2:
            return images

        id_map = {Indexer.image_id(i.path): i for i in images}
        try:
            records = self.pipeline.store.get_vectors(ids=list(id_map.keys()))
        except Exception:
            return images

        vectors = {str(r[0]): np.asarray(r[1], dtype=np.float32) for r in records}

        units: dict[str, np.ndarray] = {}
        for img in images:
            vec = vectors.get(Indexer.image_id(img.path))
            if vec is None:
                continue
            norm = max(float(np.linalg.norm(vec)), 1e-8)
            units[img.path] = vec / norm

        selected: list = []
        remaining = list(images)
        target = min(limit, len(images))

        while remaining and len(selected) < target:
            best = None
            best_score = -1e9

            for img in remaining:
                relevance = float(img.score)
                unit = units.get(img.path)

                if not selected or unit is None:
                    mmr = relevance
                else:
                    sims = [float(np.dot(unit, units[s.path])) for s in selected if s.path in units]
                    max_sim = max(sims) if sims else 0.0
                    mmr = lam * relevance - (1.0 - lam) * max_sim

                if mmr > best_score:
                    best, best_score = img, mmr

            if best is None:
                break
            selected.append(best)
            remaining.remove(best)

        return selected

    def run(
        self,
        query: str,
        top_k: int = 8,
        folder_path: str = "",
        path: str = "",
        all_results: bool = False,
        should_stop=None,
        dedupe: bool = True,
        dedupe_threshold: float = 0.95,
        exclude_paths: list[str] | None = None,
        constraints: dict | None = None,
        auto_index: bool = False,
        count_only: bool = False,
    ) -> dict[str, Any]:
        # 查询为空时用约束里的核心词兜底，全空则直接返回
        if not (query or "").strip() and constraints:
            query = str(constraints.get("core_query", "") or "").strip()
        if not (query or "").strip():
            return RetrievalResult(
                query="",
                images=[],
                confidence=0.0,
                summary="查询内容为空，没能提取出要搜的东西",
            ).model_dump()
        target = folder_path or path
        scope = ""
        if target:
            try:
                folder = self.guard.check(target, "read")
            except PathSecurityError as e:
                logger.warning("路径校验拒绝: %s", e)
                return RetrievalResult(
                    query=query,
                    images=[],
                    confidence=0.0,
                    summary=f"路径不安全，已拒绝：{e}",
                ).model_dump()
            if folder.is_dir():
                scope = str(folder)

                # 由工具自己判断该不该建库：
                # 已索引 → 直接用；未索引 → 看规模，大的自建，小的实时算。
                if not self._is_indexed(folder):
                    from ..utils.image_scan import scan_images

                    try:
                        paths = scan_images(folder, recursive=True)
                    except Exception as e:
                        logger.warning("扫描失败 %s: %s", folder, e)
                        paths = []

                    need_index = auto_index or len(paths) > self.INDEX_MIN

                    if not need_index:
                        live = self._live_search(
                            folder, query, top_k, should_stop=should_stop, paths=paths
                        )
                        if live is not None:
                            return live
                        need_index = True

                    if need_index:
                        logger.info("目录未索引且共 %d 张图片，自动建库后检索", len(paths))
                        self._ensure_indexed(folder)

                        if self._stop_hook is not None and self._stop_hook():
                            return RetrievalResult(
                                query=query,
                                images=[],
                                confidence=0.0,
                                summary="建库被取消，本次没有检索。",
                            ).model_dump()

        resp = self.pipeline.searcher.search_text(
            query,
            top_k=top_k,
            scope=scope or None,
            all_results=all_results,
            should_stop=should_stop,
            exclude_paths=exclude_paths,
            constraints=constraints,
        )
        images = [
            RetrievedImage(
                rank=r.rank,
                path=r.path,
                caption=r.caption or "",
                score=round(float(r.score), 3),
                reason=str(getattr(r, "reason", "") or ""),
            )
            for r in resp.results
        ]
        if dedupe and len(images) > 1:
            images = self._dedupe(images, dedupe_threshold)

        # 多样性打散：去重只能干掉几乎一样的图，
        # 同角度、同构图的扎堆需要另外拉开
        if constraints and constraints.get("diversity", True) and len(images) > 2:
            images = self._diversify(images, top_k)

        for i, image in enumerate(images, start=1):
            image.rank = i

        # 只计数：返回数量，不带图片列表
        if count_only:
            return {
                "count": len(images),
                "query": query,
                "images": [],
                "confidence": 0.0,
                "summary": f"「{query}」共匹配 {len(images)} 张",
            }

        confidence = sum(i.score for i in images) / len(images) if images else 0.0
        result = RetrievalResult(
            query=query,
            images=images,
            confidence=round(confidence, 3),
            summary=resp.status_line(),
        )
        return result.model_dump()
