"""程序入口：python app.py

只做三件事：修环境变量、解析配置、启动界面。
所有业务逻辑都在 findgo 包里，界面只是壳。
"""

from __future__ import annotations

import os
import sys

# Windows 上必须放在 import torch 之前
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# 允许直接 python app.py（不用先 pip install -e .）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from findgo.retriever.runtime import setup_logging, set_windows_safe_env  # noqa: E402


def main() -> None:
    set_windows_safe_env()

    from findgo.retriever import Config

    cfg = Config.from_yaml()
    setup_logging(cfg.app.log_level)

    from webui.app import launch

    print()
    print("=" * 62)
    print("  多模态图像检索  |  Chinese-CLIP + Qwen2.5 + 双向量库")
    print(f"  界面地址: http://{cfg.app.host}:{cfg.app.port}")
    print("  首次检索会加载 CLIP 模型，请耐心等待")
    print("=" * 62)
    print()

    launch()


if __name__ == "__main__":
    main()
