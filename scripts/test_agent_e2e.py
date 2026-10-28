"""端到端测试：真实 LLM + 真实检索链路。

前提：
1. 已设置 DEEPSEEK_API_KEY 环境变量（新终端）
2. 已建好图片索引（data/qdrant 或 data/chromadb）
3. configs/agent.yaml 已存在（或使用模板）

    python scripts/test_agent_e2e.py
    python scripts/test_agent_e2e.py "帮我找几张猫的图片"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from findgo.agent import FindgoAgent  # noqa: E402
from findgo.config import AgentConfig  # noqa: E402


def main() -> int:
    query = sys.argv[1] if len(sys.argv) > 1 else "帮我找几张猫的图片"

    config = AgentConfig()
    print(f"config  : {config.config_path}")
    print(f"llm     : {config.llm.get('backend')} / {config.llm.get('model')}")
    print(f"查询    : {query}")
    print("运行中（首次会加载 CLIP 与向量库）...")

    agent = FindgoAgent(config=config)
    started = time.time()
    try:
        result = agent.run(query)
    finally:
        agent.close()

    print()
    print(f"耗时    : {time.time() - started:.1f}s")
    print(f"工具    : {result.tool or '(无)'}")
    print(f"置信度  : {result.confidence}")
    print(f"重试    : {result.retries}")
    print(f"回复    : {result.reply}")
    print(f"结果数  : {len(result.results)}")
    for i, img in enumerate(result.results[:5], 1):
        path = img.get("path") if isinstance(img, dict) else getattr(img, "path", "")
        score = img.get("score") if isinstance(img, dict) else getattr(img, "score", "")
        print(f"  {i}. {path}  ({score})")

    return 0 if result.reply else 1


if __name__ == "__main__":
    raise SystemExit(main())
