"""生成部署专用配置。

    python scripts/make_config.py --mode local --out config.local.yaml
    python scripts/make_config.py --mode api --out config.api.yaml \
        --base-url https://api.deepseek.com/v1 --api-key sk-xxx --model deepseek-chat

两种模式的差别只在 LLM 重排这一段，其余（嵌入、向量库、检索参数）完全一致。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 findgo 部署配置")
    parser.add_argument("--mode", choices=["local", "api"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--llm-model", default="qwen2.5:7b-instruct")
    parser.add_argument("--title", default="")
    parser.add_argument("--backend", default="", help="覆盖向量库: chromadb / qdrant")
    args = parser.parse_args()

    source = ROOT / "config.yaml"
    if not source.exists():
        print(f"missing template: {source}")
        return 1

    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    data.setdefault("rerank", {})
    data.setdefault("app", {})

    if args.mode == "local":
        data["rerank"]["backend"] = "ollama"
        data["rerank"]["model"] = args.llm_model
        data["app"]["title"] = args.title or "多模态图像检索（本地模型）"
    else:
        if not args.base_url or not args.api_key:
            print("api mode requires --base-url and --api-key")
            return 1
        data["rerank"]["backend"] = "openai"
        data["rerank"]["model"] = args.model or "qwen-plus"
        data["rerank"].setdefault("openai", {})
        data["rerank"]["openai"]["base_url"] = args.base_url
        data["rerank"]["openai"]["api_key"] = args.api_key
        data["app"]["title"] = args.title or "多模态图像检索（API）"

    if args.backend:
        data.setdefault("vector_store", {})
        data["vector_store"]["backend"] = args.backend

    out = Path(args.out)
    out.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
