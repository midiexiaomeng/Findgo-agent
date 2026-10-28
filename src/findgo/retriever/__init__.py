"""findgo — 本地多模态 RAG 图像检索。

三级架构：
  1. 嵌入      Chinese-CLIP 把图像/文本映射到同一个 512 维语义空间
  2. 向量检索  ChromaDB / Qdrant 做 ANN 召回（Top-K，宁多勿漏）
  3. LLM 精排  Qwen2.5 结合 BLIP 图像描述做语义重排

上层只依赖 VectorStore 抽象接口，切换后端改一行配置即可。
"""

from .config import Config
from .runtime import PROJECT_ROOT, describe_device, setup_logging

__version__ = "2.0.0"

__all__ = [
    "Config",
    "PROJECT_ROOT",
    "__version__",
    "describe_device",
    "setup_logging",
    "Pipeline",
    "build_pipeline",
]


def __getattr__(name: str):
    """重模块延迟导入，保证 ``import findgo`` 不会拉起 torch。"""
    if name in ("Pipeline", "build_pipeline"):
        from .pipeline import Pipeline, build_pipeline

        return {"Pipeline": Pipeline, "build_pipeline": build_pipeline}[name]
    raise AttributeError(name)
