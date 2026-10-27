"""项目启动入口。

    python run.py

启动对话式 Agent 界面（Gradio），默认地址 http://127.0.0.1:7861
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Windows 上避免 torch 与向量库的 OpenMP 运行库冲突
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from imgsearch_agent.app import main  # noqa: E402


if __name__ == "__main__":
    print()
    print("=" * 56)
    print("  imgsearch Agent  |  ReAct 多模态图像检索")
    print("  http://127.0.0.1:7861")
    print("=" * 56)
    print()
    main()
