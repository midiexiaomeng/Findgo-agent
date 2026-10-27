"""测试 LLM 是否可用。

    python scripts/test_llm.py
    python scripts/test_llm.py --config config.api.yaml

直接发一次最小对话请求，比探测 /models 更可靠：有些代理不实现模型列表接口，
但聊天接口是通的。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from imgsearch.config import Config  # noqa: E402
from imgsearch.llm.client import LLMClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    client = LLMClient(cfg)

    print(f"backend : {client.backend}")
    print(f"model   : {client.model}")
    print(f"endpoint: {client.describe_backend()}")
    print("sending a minimal request ...")

    try:
        reply = client.chat(
            [{"role": "user", "content": "只回复两个字：正常"}],
            max_tokens=16,
        )
    except Exception as e:
        print(f"FAIL: {e}")
        return 1

    text = (reply or "").strip().replace("\n", " ")
    print(f"OK: {text[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
