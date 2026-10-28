"""第一级：Chinese-CLIP 跨模态编码。

图像和文本经由同一个模型映射到 512 维空间，这是"以图搜图"和"以文搜图"
能共用一条检索链路的前提。

设计要点：
- 模型懒加载：界面启动不碰显存，第一次真正编码时才加载。
- 维度以模型 config 为准，配置值只做校验，避免写死 512 导致换模型出错。
- 图片统一转 RGB + EXIF 校正 + 限长边，防止超大图撑爆显存。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps

from .config import Config
from .runtime import pick_device, pick_dtype, resolve_model_source
from .vectorstore.base import l2_normalize

logger = logging.getLogger(__name__)

ImageLike = str | Path | Image.Image


class ClipEncoder:
    """Chinese-CLIP 编码器。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = "cpu"  # 真正解析推迟到 load()，避免导入时就碰 torch
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._dim = int(cfg.embedding.embedding_dim)
        self._source: str | None = None
        self._torch_dtype = None

    # ── 加载 ────────────────────────────────────────────────────────────────

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def source(self) -> str:
        return self._source or ""

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return

            import torch
            from transformers import AutoModel, AutoProcessor

            self.device = pick_device(self.cfg.embedding.device or self.cfg.device)
            self._torch_dtype = pick_dtype(self.device, self.cfg.embedding.use_fp16)

            source = resolve_model_source(
                self.cfg.embedding.model,
                self.cfg.models_dir,
                self.cfg.embedding.fallback,
            )
            self._source = source
            logger.info(
                "加载 CLIP: %s (device=%s, dtype=%s)", source, self.device, self._torch_dtype
            )

            try:
                self._processor = AutoProcessor.from_pretrained(source)
                self._model = AutoModel.from_pretrained(
                    source,
                    torch_dtype=self._torch_dtype,
                )
            except Exception as e:
                raise RuntimeError(
                    f"CLIP 模型加载失败: {source}\n"
                    f"如果尚未下载，请运行: python scripts/download_models.py\n"
                    f"原始错误: {e}"
                ) from e

            self._model.to(self.device)
            self._model.eval()

            # 维度以模型为准
            model_dim = getattr(self._model.config, "projection_dim", None)
            if model_dim:
                if int(model_dim) != self._dim:
                    logger.warning(
                        "配置维度 %d 与模型实际维度 %d 不一致，以模型为准；"
                        "如果已有索引，请重建。",
                        self._dim,
                        int(model_dim),
                    )
                self._dim = int(model_dim)

            logger.info("CLIP 就绪，嵌入维度 = %d", self._dim)

    # ── 文本 ────────────────────────────────────────────────────────────────

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """批量编码文本，返回 (n, dim) 的归一化 float32 向量。"""
        texts = [str(t) for t in texts if str(t).strip()]
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        self.load()
        import torch

        bs = max(1, self.cfg.embedding.batch_size)
        chunks: list[np.ndarray] = []

        with torch.inference_mode():
            for i in range(0, len(texts), bs):
                batch = texts[i : i + bs]
                inputs = self._processor(
                    text=batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.cfg.embedding.max_text_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                feats = self._text_features(inputs)
                feats = feats.float().cpu().numpy()
                if self.cfg.embedding.normalize:
                    feats = l2_normalize(feats)
                chunks.append(feats.astype(np.float32))

        return np.concatenate(chunks, axis=0)

    def encode_text(self, text: str) -> np.ndarray:
        """编码单条文本，返回 (dim,)。"""
        out = self.encode_texts([text])
        if out.shape[0] == 0:
            return np.zeros(self._dim, dtype=np.float32)
        return out[0]

    # ── 图像 ────────────────────────────────────────────────────────────────

    def _text_features(self, inputs: dict):
        """取文本特征，兼容 Chinese-CLIP 在 transformers 4.57 的回归 bug。

        该版本用 ``add_pooling_layer=False`` 构建 text_model，但
        ``get_text_features`` 仍然读 ``pooler_output``，拿到 None 后在
        ``text_projection`` 抛 TypeError。按官方文档语义，文本特征应当是
        最后一层 [CLS] 过投影层，这里直接这样算。

        其它模型（标准 CLIP / SigLIP 等）没有这个结构，照常走原生接口。
        """
        text_model = getattr(self._model, "text_model", None)
        pooler = getattr(text_model, "pooler", "n/a") if text_model is not None else "n/a"

        if text_model is not None and pooler is None and hasattr(self._model, "text_projection"):
            outputs = text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                token_type_ids=inputs.get("token_type_ids"),
            )
            pooled = outputs.last_hidden_state[:, 0, :]
            return self._model.text_projection(pooled)

        return self._model.get_text_features(**inputs)

    def encode_images(self, images: Sequence[ImageLike]) -> np.ndarray:
        """批量编码图像，返回 (n, dim) 的归一化向量。

        无法读取的图片会被跳过，返回数组长度可能小于输入长度；调用方应
        用 :meth:`load_image` 预先校验，或用 ``encode_images_checked``。
        """
        if not images:
            return np.zeros((0, self._dim), dtype=np.float32)

        pil_images: list[Image.Image] = []
        for img in images:
            try:
                pil_images.append(self.load_image(img))
            except Exception as e:
                logger.warning("跳过无法读取的图片 %s: %s", img, e)

        if not pil_images:
            return np.zeros((0, self._dim), dtype=np.float32)

        self.load()
        import torch

        bs = max(1, self.cfg.embedding.batch_size)
        chunks: list[np.ndarray] = []

        with torch.inference_mode():
            for i in range(0, len(pil_images), bs):
                batch = pil_images[i : i + bs]
                inputs = self._processor(images=batch, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                feats = self._model.get_image_features(**inputs)
                feats = feats.float().cpu().numpy()
                if self.cfg.embedding.normalize:
                    feats = l2_normalize(feats)
                chunks.append(feats.astype(np.float32))

        return np.concatenate(chunks, axis=0)

    def encode_image(self, image: ImageLike) -> np.ndarray:
        out = self.encode_images([image])
        if out.shape[0] == 0:
            return np.zeros(self._dim, dtype=np.float32)
        return out[0]

    @property
    def max_image_side(self) -> int:
        return int(self.cfg.embedding.max_image_side)

    def load_image(self, image: ImageLike) -> Image.Image:
        """读图 + EXIF 方向校正 + 转 RGB + 限制长边。"""
        if isinstance(image, Image.Image):
            img = image
        else:
            img = Image.open(str(image))
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        if getattr(img, "mode", "RGB") != "RGB":
            img = img.convert("RGB")
        else:
            img = img.copy()

        side = self.max_image_side
        if side and max(img.size) > side:
            img.thumbnail((side, side), Image.LANCZOS)
        return img

    # ── 生命周期 ────────────────────────────────────────────────────────────

    def unload(self) -> None:
        """释放显存。索引结束后调用，避免和 7B 模型抢显存。"""
        self._model = None
        self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def info(self) -> dict:
        return {
            "model": self._source or self.cfg.embedding.model,
            "device": self.device,
            "dim": self._dim,
            "loaded": self.loaded,
        }
