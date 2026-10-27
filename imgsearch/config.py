"""配置：YAML <-> dataclass，带默认值与深度合并。

设计原则：
- 所有相对路径都相对项目根目录，而不是当前工作目录。
- 每个字段都有默认值，配置文件缺字段不会炸。
- 环境变量 IMGSRCH_<SECTION>_<KEY> 可覆盖（方便 Docker/脚本临时改）。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .runtime import PROJECT_ROOT, resolve_path

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    title: str = "多模态图像检索"
    host: str = "127.0.0.1"
    port: int = 7860
    open_browser: bool = True
    share: bool = False
    log_level: str = "INFO"


@dataclass
class PathsConfig:
    models_dir: str = "./models"
    data_dir: str = "./data"


@dataclass
class EmbeddingConfig:
    # 短名：优先 models/<短名>，找不到回退 HuggingFace
    model: str = "chinese-clip-vit-base-patch16"
    fallback: str = "OFA-Sys/chinese-clip-vit-base-patch16"
    embedding_dim: int = 512
    normalize: bool = True
    batch_size: int = 32
    max_text_length: int = 52
    # 图片送入编码器前的最长边，太大浪费显存
    max_image_side: int = 1024
    use_fp16: bool = True
    # 空 = 跟随全局 device。显存紧张时给 CLIP 单独指定 cpu，
    # 查询文本的编码在 CPU 上只要几十毫秒，把显存留给重排模型。
    device: str = ""


@dataclass
class ChromaConfig:
    path: str = "./data/chromadb"


@dataclass
class QdrantConfig:
    # local = 嵌入式文件模式，零服务；server = 连远端 Qdrant
    mode: str = "local"
    path: str = "./data/qdrant"
    url: str = "http://localhost:6333"
    api_key: str = ""
    timeout: int = 30


@dataclass
class VectorStoreConfig:
    backend: str = "chromadb"  # chromadb | qdrant
    collection: str = "images"
    chromadb: ChromaConfig = field(default_factory=ChromaConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)


@dataclass
class CaptionConfig:
    enabled: bool = True
    model: str = "blip-image-captioning-base"
    fallback: str = "Salesforce/blip-image-captioning-base"
    batch_size: int = 8
    max_new_tokens: int = 40
    beam_size: int = 3
    cache_db: str = "./data/captions.sqlite"
    use_fp16: bool = False
    # 索引完成后立刻释放 BLIP 显存
    unload_after_index: bool = True


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    timeout: int = 120


@dataclass
class OpenAIConfig:
    base_url: str = "http://localhost:8000/v1"  # vLLM / LM Studio / llama.cpp server
    api_key: str = "not-needed"
    timeout: int = 120


@dataclass
class TransformersLLMConfig:
    # 本地 HuggingFace 模型目录，相对路径按项目根目录解析
    model_path: str = "./models/Qwen2.5-3B-Instruct"
    device: str = "auto"
    dtype: str = "float16"
    max_input_tokens: int = 3072


@dataclass
class RerankConfig:
    enabled: bool = True
    backend: str = "ollama"  # ollama | openai | transformers | none
    model: str = "qwen2.5:7b-instruct"
    # 最多把多少个候选送进 LLM（再多 prompt 太长、太慢）
    max_candidates: int = 16
    # 每次请求评估几张图
    batch_size: int = 8
    max_tokens: int = 512
    temperature: float = 0.0
    retries: int = 1
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    transformers: TransformersLLMConfig = field(default_factory=TransformersLLMConfig)


@dataclass
class FusionConfig:
    vector_weight: float = 0.35
    llm_weight: float = 0.65


@dataclass
class SearchConfig:
    recall_k: int = 50          # 第一级 ANN 召回数，宁多勿漏
    top_k: int = 12              # 最终展示数
    min_score: float = 0.0
    fusion: FusionConfig = field(default_factory=FusionConfig)


@dataclass
class Config:
    device: str = "auto"  # auto | cpu | cuda | cuda:0 | mps
    app: AppConfig = field(default_factory=AppConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    caption: CaptionConfig = field(default_factory=CaptionConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    @property
    def config_path(self) -> str:
        return getattr(self, "_config_path", "") or str(PROJECT_ROOT / "config.yaml")

    # ── 派生路径 ────────────────────────────────────────────────────────────

    @property
    def root(self) -> Path:
        return PROJECT_ROOT

    @property
    def models_dir(self) -> Path:
        return resolve_path(self.paths.models_dir)

    @property
    def data_dir(self) -> Path:
        return resolve_path(self.paths.data_dir)

    def resolve(self, p: str | Path) -> Path:
        return resolve_path(p)

    # ── 加载 ────────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "Config":
        """加载配置。

        优先级：显式 path > 环境变量 IMGSRCH_CONFIG > 项目根目录 config.yaml。
        两套部署（本地模型 / API）靠环境变量指向不同配置文件，互不干扰。
        """
        if path:
            resolved = Path(path)
        else:
            env_path = os.environ.get("IMGSRCH_CONFIG", "").strip()
            resolved = Path(env_path) if env_path else PROJECT_ROOT / "config.yaml"

        path = resolved
        raw: dict[str, Any] = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        else:
            logger.warning("配置文件不存在: %s，使用默认配置", path)

        cfg = _from_dict(cls, raw)
        cfg._config_path = str(path)
        cfg._apply_env_overrides()
        return cfg

    def _apply_env_overrides(self) -> None:
        """IMGSRCH_VECTOR_STORE__BACKEND=qdrant 这种形式覆盖任意字段。"""
        for key, value in os.environ.items():
            if not key.startswith("IMGSRCH_"):
                continue
            parts = key[len("IMGSRCH_"):].lower().split("__")
            if len(parts) < 2:
                continue
            target: Any = self
            ok = True
            for part in parts[:-1]:
                target = getattr(target, part, None)
                if target is None:
                    ok = False
                    break
            if ok and hasattr(target, parts[-1]):
                current = getattr(target, parts[-1])
                try:
                    setattr(target, parts[-1], _coerce(value, current))
                    logger.info("环境变量覆盖配置: %s", key)
                except Exception:
                    logger.warning("无法应用环境变量 %s=%s", key, value)

    def to_dict(self) -> dict:
        return _to_dict(self)


def _coerce(value: str, like: Any) -> Any:
    if isinstance(like, bool):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(like, int):
        return int(value)
    if isinstance(like, float):
        return float(value)
    return value


def _from_dict(cls: type, raw: dict) -> Any:
    """按 dataclass 结构递归实例化，未知键忽略并警告。"""
    if not is_dataclass(cls):
        return raw
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    for key, value in (raw or {}).items():
        key = key.strip()
        if key not in known or key.startswith("_"):
            logger.warning("配置中有未知字段，已忽略: %s.%s", cls.__name__, key)
            continue
        ftype = known[key].type
        # 处理嵌套 dataclass
        nested_cls = _nested_dataclass(ftype)
        if nested_cls is not None and isinstance(value, dict):
            kwargs[key] = _from_dict(nested_cls, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _nested_dataclass(tp: Any) -> type | None:
    """从注解里找出 dataclass 类型（兼容字符串注解）。"""
    if isinstance(tp, type) and is_dataclass(tp):
        return tp
    if isinstance(tp, str):
        mapping = {
            "AppConfig": AppConfig,
            "PathsConfig": PathsConfig,
            "EmbeddingConfig": EmbeddingConfig,
            "ChromaConfig": ChromaConfig,
            "QdrantConfig": QdrantConfig,
            "VectorStoreConfig": VectorStoreConfig,
            "CaptionConfig": CaptionConfig,
            "OllamaConfig": OllamaConfig,
            "OpenAIConfig": OpenAIConfig,
            "RerankConfig": RerankConfig,
            "TransformersLLMConfig": TransformersLLMConfig,
            "FusionConfig": FusionConfig,
            "SearchConfig": SearchConfig,
        }
        return mapping.get(tp)
    return None


def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj
