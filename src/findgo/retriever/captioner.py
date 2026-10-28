"""BLIP 图像描述。

为什么要在索引阶段就生成描述，而不是搜索时现算：
- 一张图 0.1~1 秒，假如每次搜索都对 Top-50 现算，光描述就要几十秒。
- 描述是图片的静态属性，算一次存进向量库 metadata，之后直接读。
- 搜索时唯一需要现算的只有"以图搜图"的那张查询图。

描述结果同时落 SQLite 缓存，重建索引时可以直接复用，不必重复跑模型。
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Iterable, Sequence

from .config import Config
from .runtime import pick_device, pick_dtype, resolve_model_source

logger = logging.getLogger(__name__)

_CAPTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS captions (
    id      TEXT PRIMARY KEY,
    caption TEXT NOT NULL,
    model   TEXT,
    mtime   REAL
);
"""


class Captioner:
    """BLIP 描述生成器（带磁盘缓存）。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.enabled = bool(cfg.caption.enabled)
        self.device = "cpu"
        self._model = None
        self._processor = None
        self._load_error: str | None = None
        self._lock = threading.Lock()
        self._db_lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        self._db_path = cfg.resolve(cfg.caption.cache_db)

    # ── 状态 ────────────────────────────────────────────────────────────────

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def available(self) -> bool:
        return self.enabled and self._load_error is None

    # ── 缓存 ────────────────────────────────────────────────────────────────

    def _ensure_db(self) -> sqlite3.Connection | None:
        if not self.enabled:
            return None
        if self._db is not None:
            return self._db
        with self._db_lock:
            if self._db is not None:
                return self._db
            try:
                self._db_path.parent.mkdir(parents=True, exist_ok=True)
                self._db = sqlite3.connect(str(self._db_path), check_same_thread=False)
                self._db.execute(_CAPTION_SCHEMA)
                self._db.commit()
            except Exception as e:
                logger.warning("描述缓存不可用（将每次重新生成）: %s", e)
                self._db = None
        return self._db

    def _cache_get(self, image_id: str, mtime: float | None) -> str | None:
        db = self._ensure_db()
        if db is None:
            return None
        try:
            with self._db_lock:
                row = db.execute(
                    "SELECT caption, mtime FROM captions WHERE id = ?", (image_id,)
                ).fetchone()
            if not row:
                return None
            caption, cached_mtime = row
            # 文件改过就作废
            if mtime is not None and cached_mtime is not None:
                if abs(float(cached_mtime) - float(mtime)) > 1.0:
                    return None
            return str(caption) if caption is not None else None
        except Exception:
            return None

    def _cache_put_many(self, rows: Iterable[tuple[str, str, float | None]]) -> None:
        db = self._ensure_db()
        if db is None:
            return
        model_name = self.cfg.caption.model
        try:
            with self._db_lock:
                db.executemany(
                    "INSERT OR REPLACE INTO captions(id, caption, model, mtime) VALUES (?,?,?,?)",
                    [(i, c, model_name, m) for i, c, m in rows],
                )
                db.commit()
        except Exception as e:
            logger.debug("写入描述缓存失败: %s", e)
            return

        self._evict_cache()

    def _evict_cache(self) -> int:
        """超过条数上限时，从最早写入的记录开始淘汰。"""
        db = self._ensure_db()
        if db is None:
            return 0

        limit = int(getattr(self.cfg.caption, "max_cache_rows", 50000))
        if limit <= 0:
            return 0

        try:
            with self._db_lock:
                total = db.execute("SELECT COUNT(*) FROM captions").fetchone()[0]
                if total <= limit:
                    return 0
                excess = total - limit
                db.execute(
                    "DELETE FROM captions WHERE rowid IN "
                    "(SELECT rowid FROM captions ORDER BY rowid ASC LIMIT ?)",
                    (excess,),
                )
                db.commit()
            logger.info("描述缓存超出上限，淘汰 %d 条", excess)
            return excess
        except Exception as e:
            logger.debug("清理描述缓存失败: %s", e)
            return 0

    def cache_size(self) -> int:
        db = self._ensure_db()
        if db is None:
            return 0
        try:
            with self._db_lock:
                return int(db.execute("SELECT COUNT(*) FROM captions").fetchone()[0])
        except Exception:
            return 0

    # ── 模型 ────────────────────────────────────────────────────────────────

    def _load(self) -> bool:
        if not self.enabled:
            return False
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False

        with self._lock:
            if self._model is not None:
                return True
            try:
                import torch
                from transformers import BlipForConditionalGeneration, BlipProcessor

                self.device = pick_device(self.cfg.device)
                dtype = pick_dtype(self.device, self.cfg.caption.use_fp16)
                source = resolve_model_source(
                    self.cfg.caption.model,
                    self.cfg.models_dir,
                    self.cfg.caption.fallback,
                )
                logger.info("加载 BLIP: %s (device=%s)", source, self.device)
                self._processor = BlipProcessor.from_pretrained(source)
                self._model = BlipForConditionalGeneration.from_pretrained(
                    source, torch_dtype=dtype
                )
                self._model.to(self.device)
                self._model.eval()
                logger.info("BLIP 就绪")
                return True
            except Exception as e:
                self._load_error = str(e)
                logger.warning(
                    "BLIP 加载失败，索引将不包含图像描述（LLM 重排会自动退化为纯向量排序）。%s",
                    e,
                )
                return False

    # ── 生成 ────────────────────────────────────────────────────────────────

    def captions_for(self, records: Sequence[dict]) -> dict[str, str]:
        """给一批记录生成描述。

        Args:
            records: 每项至少含 ``id`` 和 ``path``，可选 ``mtime``。

        Returns:
            {id: caption}，失败或未启用的条目不在结果里。
        """
        if not self.enabled or not records:
            return {}

        result: dict[str, str] = {}
        pending: list[dict] = []

        for rec in records:
            image_id = str(rec.get("id", ""))
            if not image_id:
                continue
            cached = self._cache_get(image_id, rec.get("mtime"))
            if cached is not None:
                result[image_id] = cached
            else:
                pending.append(rec)

        if not pending:
            return result

        if not self._load():
            return result

        bs = max(1, int(self.cfg.caption.batch_size))
        for i in range(0, len(pending), bs):
            batch = pending[i : i + bs]
            generated = self._generate_batch(batch)
            cache_rows = []
            for rec, caption in zip(batch, generated):
                image_id = str(rec["id"])
                if caption:
                    result[image_id] = caption
                cache_rows.append((image_id, caption or "", rec.get("mtime")))
            self._cache_put_many(cache_rows)

        return result

    def caption_image(self, image_path: str | Path) -> str:
        """给单张图生成描述（以图搜图时给查询图用）。"""
        if not self.enabled:
            return ""
        if not self._load():
            return ""
        out = self._generate_batch([{"path": str(image_path)}])
        return out[0] if out else ""

    def _generate_batch(self, records: Sequence[dict]) -> list[str]:
        """返回与输入等长的描述列表，失败的为空字符串。"""
        import torch
        from PIL import Image, ImageOps

        pil_images: list[Image.Image] = []
        slots: list[int] = []

        for idx, rec in enumerate(records):
            path = rec.get("path")
            if not path:
                continue
            try:
                img = Image.open(str(path))
                img = ImageOps.exif_transpose(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                side = self.cfg.embedding.max_image_side
                if side and max(img.size) > side:
                    img.thumbnail((side, side), Image.LANCZOS)
                pil_images.append(img)
                slots.append(idx)
            except Exception as e:
                logger.debug("描述生成跳过坏图 %s: %s", path, e)

        results = ["" for _ in records]
        if not pil_images:
            return results

        try:
            with torch.inference_mode():
                inputs = self._processor(images=pil_images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=int(self.cfg.caption.max_new_tokens),
                    num_beams=max(1, int(self.cfg.caption.beam_size)),
                )
                texts = self._processor.batch_decode(output, skip_special_tokens=True)
        except Exception as e:
            logger.warning("BLIP 批量生成失败，本批描述留空: %s", e)
            return results

        for slot, text in zip(slots, texts):
            results[slot] = str(text).strip()
        return results

    # ── 生命周期 ────────────────────────────────────────────────────────────

    def unload(self) -> None:
        self._model = None
        self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def close(self) -> None:
        self.unload()
        with self._db_lock:
            if self._db is not None:
                try:
                    self._db.close()
                except Exception:
                    pass
                self._db = None

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "loaded": self.loaded,
            "device": self.device,
            "cache_size": self.cache_size(),
            "error": self._load_error,
        }
