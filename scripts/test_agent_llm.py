"""验证 agent.yaml 里配置的 LLM 是否可用。

    python scripts/test_agent_llm.py
    python scripts/test_agent_llm.py --config configs/agent.yaml

只发一次最小请求，确认 Key、地址、模型名都对。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from findgo.config import AgentConfig  # noqa: E402
from findgo.llm import ChatClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = AgentConfig(args.config)
    client = ChatClient(cfg.llm)

    print(f"backend : {client.backend}")
    print(f"model   : {client.model}")
    print(f"endpoint: {client.base_url}")
    print(f"api_key : {'已配置' if client.api_key and client.api_key != 'not-needed' else '未配置'}")
    print("sending a minimal request ...")

    try:
        reply = client.chat(
            [{"role": "user", "content": "只回复两个字：正常"}],
            max_tokens=32,
        )
    except Exception as e:
        print(f"FAIL: {e}")
        return 1

    print(f"OK: {reply.strip()[:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
