"""Agent 层配置加载：读取 configs/agent.yaml 与 configs/tools.yaml。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# src/imgsearch_agent/config.py → 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


class AgentConfig:
    """Agent 行为配置，与代码分离。"""

    def __init__(self, path: str | Path | None = None):
        raw = _load(Path(path) if path else PROJECT_ROOT / "configs" / "agent.yaml")
        self.raw = raw

        agent = raw.get("agent", {}) or {}
        self.max_iterations = int(agent.get("max_iterations", 5))
        self.max_retries = int(agent.get("max_retries", 2))
        self.similarity_threshold = float(agent.get("similarity_threshold", 0.25))
        self.top_k = int(agent.get("top_k", 8))

        self.llm: dict[str, Any] = raw.get("llm", {}) or {}
        self.prompts: dict[str, str] = raw.get("prompts", {}) or {}


def load_tools_config(path: str | Path | None = None) -> list[dict[str, Any]]:
    """读取工具注册配置。"""
    raw = _load(Path(path) if path else PROJECT_ROOT / "configs" / "tools.yaml")
    return raw.get("tools", []) or []
