"""索引器：扫描目录 -> CLIP 编码 -> BLIP 描述 -> 写入向量库。

几个容易踩的坑，这里都处理了：
- 坏图会让整批编码抛异常，所以先 verify 再进模型，失败只记一笔。
- 编码器内部跳过的图会导致向量与路径错位，所以带数量校验和逐张兜底。
- 重复索引要幂等：id 由绝对路径的 uuid5 决定，重启进程也不变。
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image

from .captioner import Captioner
from .clip_encoder import ClipEncoder
from .config import Config
from .runtime import human_time
from .vectorstore.base import VectorStore

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
    ".avif",
}

ProgressFn = Callable[[int, int, str], None]


@dataclass
class IndexStats:
    total: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    captioned: int = 0
    pruned: int = 0
    elapsed: float = 0.0
    locked: bool = False  # 同一目录已有索引任务在跑，本次被跳过
    cancelled: bool = False  # 用户中途停止
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        base = (
            f"扫描 {self.total} 张，索引 {self.indexed} 张，跳过 {self.skipped} 张，"
            f"失败 {self.failed} 张，描述 {self.captioned} 条，耗时 {human_time(self.elapsed)}"
        )
        return f"{base}（已中断）" if self.cancelled else base

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "indexed": self.indexed,
            "skipped": self.skipped,
            "failed": self.failed,
            "captioned": self.captioned,
            "pruned": self.pruned,
            "elapsed": round(self.elapsed, 2),
            "cancelled": self.cancelled,
            "errors": self.errors[:20],
        }


class Indexer:
    def __init__(
        self,
        cfg: Config,
        encoder: ClipEncoder,
        store: VectorStore,
        captioner: Captioner,
    ):
        self.cfg = cfg
        self.encoder = encoder
        self.store = store
        self.captioner = captioner

        # 按文件夹加锁：同一目录同一时间只允许一个索引任务，不同目录互不影响
        self._folder_locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # ── 对外 ────────────────────────────────────────────────────────────────

    def _folder_lock(self, folder: Path) -> threading.Lock:
        key = str(folder.resolve()).lower()
        with self._locks_guard:
            lock = self._folder_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._folder_locks[key] = lock
            return lock

    def index_folder(
        self,
        folder: str | Path,
        recursive: bool = True,
        reset: bool = False,
        skip_existing: bool = True,
        with_caption: bool | None = None,
        progress: ProgressFn | None = None,
        prune_missing: bool = False,
        should_stop: Callable[[], bool] | None = None,
        should_pause: Callable[[], bool] | None = None,
    ) -> IndexStats:
        folder = Path(folder).expanduser()
        lock = self._folder_lock(folder)

        if not lock.acquire(blocking=False):
            logger.info("目录 %s 正在索引中，跳过本次请求", folder)
            return IndexStats(locked=True)

        try:
            return self._index_folder_locked(
                folder,
                recursive=recursive,
                reset=reset,
                skip_existing=skip_existing,
                with_caption=with_caption,
                progress=progress,
                prune_missing=prune_missing,
                should_stop=should_stop,
                should_pause=should_pause,
            )
        finally:
            lock.release()

    def _index_folder_locked(
        self,
        folder: Path,
        recursive: bool = True,
        reset: bool = False,
        skip_existing: bool = True,
        with_caption: bool | None = None,
        progress: ProgressFn | None = None,
        prune_missing: bool = False,
        should_stop: Callable[[], bool] | None = None,
        should_pause: Callable[[], bool] | None = None,
    ) -> IndexStats:
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"文件夹不存在: {folder}")

        started = time.time()
        stats = IndexStats()

        if reset:
            logger.info("按请求清空向量库 collection=%s", self.store.collection)
            self.store.reset()

        paths = self.scan(folder, recursive=recursive)
        stats.total = len(paths)
        if not paths:
            stats.elapsed = time.time() - started
            return stats

        if prune_missing:
            stats.pruned = self._prune_missing(paths)

        existing = set(self.store.get_all_ids()) if skip_existing else set()
        pending = [p for p in paths if self.image_id(p) not in existing]
        stats.skipped = stats.total - len(pending)

        if not pending:
            stats.elapsed = time.time() - started
            return stats

        if progress:
            progress(0, len(pending), f"待索引 {len(pending)} 张")

        bs = max(1, int(self.cfg.embedding.batch_size))
        done = 0

        for start in range(0, len(pending), bs):
            if should_stop is not None and should_stop():
                stats.cancelled = True
                logger.info("索引进度 %d/%d，收到停止信号", done, len(pending))
                break

            # 暂停：原地等，每 0.3 秒看一眼是否恢复或停止
            paused_notified = False
            while should_pause is not None and should_pause():
                if not paused_notified:
                    logger.info("索引进度 %d/%d，已暂停", done, len(pending))
                    if progress:
                        progress(done, len(pending), "已暂停")
                    paused_notified = True
                if should_stop is not None and should_stop():
                    stats.cancelled = True
                    break
                time.sleep(0.3)

            if stats.cancelled:
                break

            batch = pending[start : start + bs]
            batch_stats = self._index_batch(batch, with_caption=with_caption)
            stats.indexed += batch_stats["indexed"]
            stats.failed += batch_stats["failed"]
            stats.captioned += batch_stats["captioned"]
            stats.errors.extend(batch_stats["errors"])
            done += len(batch)
            if progress:
                progress(done, len(pending), batch[-1].name)

        stats.elapsed = time.time() - started
        logger.info("索引完成: %s", stats.summary())

        if self.cfg.caption.unload_after_index:
            self.captioner.unload()

        return stats

    def index_paths(
        self,
        paths: Sequence[str | Path],
        with_caption: bool | None = None,
        progress: ProgressFn | None = None,
    ) -> IndexStats:
        """只索引指定的若干文件（不扫描父目录）。"""
        files = [Path(p) for p in paths if Path(p).is_file()]
        stats = IndexStats(total=len(files))
        if not files:
            return stats

        started = time.time()
        bs = max(1, int(self.cfg.embedding.batch_size))
        done = 0

        for start in range(0, len(files), bs):
            batch = files[start : start + bs]
            result = self._index_batch(batch, with_caption=with_caption)
            stats.indexed += result["indexed"]
            stats.failed += result["failed"]
            stats.captioned += result["captioned"]
            stats.errors.extend(result["errors"])
            done += len(batch)
            if progress:
                progress(done, len(files), batch[-1].name)

        stats.elapsed = time.time() - started
        if self.cfg.caption.unload_after_index:
            self.captioner.unload()
        return stats

    # ── 扫描 ────────────────────────────────────────────────────────────────

    @staticmethod
    def scan(folder: str | Path, recursive: bool = True) -> list[Path]:
        folder = Path(folder)
        iterator = folder.rglob("*") if recursive else folder.glob("*")
        paths: list[Path] = []
        seen: set[str] = set()
        for p in iterator:
            try:
                if not p.is_file():
                    continue
                if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                key = str(p.resolve()).lower()
                if key in seen:
                    continue
                seen.add(key)
                paths.append(p)
            except OSError:
                continue
        paths.sort(key=lambda p: (p.stat().st_mtime, str(p).lower()))
        return paths

    @staticmethod
    def image_id(path: str | Path) -> str:
        """稳定 ID：绝对路径的 uuid5。

        不用 hash()，因为 Python 的字符串 hash 每个进程都会加盐，
        重启后同一张图会得到不同 id，增量索引和删除都会失效。
        """
        absolute = str(Path(path).expanduser().resolve()).lower()
        return str(uuid.uuid5(uuid.NAMESPACE_URL, absolute))

    # ── 批处理 ──────────────────────────────────────────────────────────────

    def _index_batch(self, paths: Sequence[Path], with_caption: bool | None = None) -> dict:
        result = {"indexed": 0, "failed": 0, "captioned": 0, "errors": []}

        # 1) 校验 + 读取元信息
        valid: list[Path] = []
        metas: list[dict] = []
        for p in paths:
            try:
                stat = p.stat()
                with Image.open(p) as im:
                    width, height = im.size
                    im.verify()
                metas.append(
                    {
                        "path": str(p.resolve()),
                        "filename": p.name,
                        "ext": p.suffix.lower(),
                        "size": int(stat.st_size),
                        "mtime": float(stat.st_mtime),
                        "width": int(width),
                        "height": int(height),
                        "indexed_at": float(time.time()),
                    }
                )
                valid.append(p)
            except Exception as e:
                result["failed"] += 1
                msg = f"{p.name}: {e}"
                result["errors"].append(msg)
                logger.warning("跳过无法读取的图片 %s", msg)

        if not valid:
            return result

        # 2) CLIP 编码
        vectors, good_paths, good_metas, encode_errors = self._encode_safe(valid, metas)
        result["failed"] += len(encode_errors)
        result["errors"].extend(encode_errors)
        if not good_paths:
            return result

        ids = [self.image_id(p) for p in good_paths]

        # 3) BLIP 描述
        captions: dict[str, str] = {}
        use_caption = self.cfg.caption.enabled if with_caption is None else with_caption
        if use_caption and self.captioner.available:
            records = [
                {
                    "id": i,
                    "path": m["path"],
                    "mtime": m["mtime"],
                }
                for i, m in zip(ids, good_metas)
            ]
            try:
                captions = self.captioner.captions_for(records)
            except Exception as e:
                logger.warning("描述生成失败，本批不带描述: %s", e)
                result["errors"].append(f"描述生成失败: {e}")
        result["captioned"] += len(captions)

        for image_id, meta in zip(ids, good_metas):
            meta["caption"] = captions.get(image_id, "")

        # 4) 写库
        try:
            self.store.add(ids, vectors, good_metas)
            result["indexed"] += len(ids)
        except Exception as e:
            logger.error("批量写入向量库失败，改为逐条写入: %s", e)
            for i, (image_id, vec, meta) in enumerate(zip(ids, vectors, good_metas)):
                try:
                    self.store.add([image_id], vec.reshape(1, -1), [meta])
                    result["indexed"] += 1
                except Exception as e2:
                    result["failed"] += 1
                    result["errors"].append(f"{meta.get('filename')}: {e2}")

        return result

    def _encode_safe(
        self, paths: Sequence[Path], metas: Sequence[dict]
    ) -> tuple[np.ndarray, list[Path], list[dict], list[str]]:
        """整批编码优先，失败则逐张兜底，保证向量与元信息严格对齐。"""
        try:
            vectors = self.encoder.encode_images([str(p) for p in paths])
            if len(vectors) == len(paths):
                return vectors, list(paths), list(metas), []
            logger.warning(
                "批量编码返回数量不一致（%d != %d），转为逐张编码",
                len(vectors),
                len(paths),
            )
        except Exception as e:
            logger.warning("批量编码失败，转为逐张: %s", e)

        good_paths: list[Path] = []
        good_metas: list[dict] = []
        vectors_list: list[np.ndarray] = []
        errors: list[str] = []

        for p, meta in zip(paths, metas):
            try:
                vec = self.encoder.encode_images([str(p)])
                if len(vec) != 1:
                    errors.append(f"{p.name}: 编码结果为空")
                    continue
                vectors_list.append(vec[0])
                good_paths.append(p)
                good_metas.append(meta)
            except Exception as e:
                errors.append(f"{p.name}: {e}")

        if not vectors_list:
            return np.zeros((0, self.encoder.dim), dtype=np.float32), [], [], errors

        return np.stack(vectors_list).astype(np.float32), good_paths, good_metas, errors

    # ── 清理 ────────────────────────────────────────────────────────────────

    def _prune_missing(self, scanned: Sequence[Path]) -> int:
        """删除库里存在、但磁盘上已经找不到的记录。"""
        current = {self.image_id(p) for p in scanned}
        stale = [i for i in self.store.get_all_ids() if i not in current]
        if stale:
            self.store.delete(stale)
            logger.info("清理失效记录 %d 条", len(stale))
        return len(stale)
