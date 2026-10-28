"""Agent 层配置加载：读取 configs/agent.yaml 与 configs/tools.yaml。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# src/findgo/config.py → 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


class AgentConfig:
    """Agent 行为配置，与代码分离。"""

    def __init__(self, path: str | Path | None = None):
        if path:
            config_path = Path(path)
        else:
            local = PROJECT_ROOT / "configs" / "agent.yaml"
            example = PROJECT_ROOT / "configs" / "agent.example.yaml"
            config_path = local if local.exists() else example
        raw = _load(config_path)
        self.config_path = str(config_path)
        self.raw = raw

        agent = raw.get("agent", {}) or {}
        self.max_iterations = int(agent.get("max_iterations", 5))
        self.max_retries = int(agent.get("max_retries", 2))
        self.similarity_threshold = float(agent.get("similarity_threshold", 0.25))
        self.top_k = int(agent.get("top_k", 8))

        self.llm: dict[str, Any] = raw.get("llm", {}) or {}
        self.prompts: dict[str, str] = raw.get("prompts", {}) or {}

        security = raw.get("security", {}) or {}
        sandbox = raw.get("sandbox", {}) or {}

        # 只读素材目录：优先 sandbox.image_roots，兼容旧的 security.allowed_roots
        self.image_roots = [
            str(r)
            for r in (sandbox.get("image_roots") or security.get("allowed_roots") or [])
            if r
        ]
        self.workspace_roots = [
            str(r) for r in (sandbox.get("workspace_roots") or []) if r
        ]
        # 旧字段保留，供界面与日志读取
        self.allowed_roots = list(self.image_roots)

        downloads = raw.get("download_cache", {}) or {}
        self.download_path = str(downloads.get("path") or "./data/downloads")
        self.download_max_gb = float(downloads.get("max_total_size_gb", 5.0))

        trash = raw.get("trash_staging", {}) or {}
        self.trash_path = str(trash.get("path") or "./data/trash_staging")
        self.trash_max_days = int(trash.get("max_keep_days", 14))
        self.trash_max_gb = float(trash.get("max_total_size_gb", 10.0))
        self.trash_auto_clean = bool(trash.get("auto_clean", True))

        memory = raw.get("memory", {}) or {}
        self.memory_persist = bool(memory.get("persist", False))
        self.memory_db_path = memory.get("db_path", "./data/agent_sessions.sqlite")
        self.memory_max_sessions = int(memory.get("max_sessions", 20))

    def resolve_db_path(self) -> Path:
        p = Path(self.memory_db_path)
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_tools_config(path: str | Path | None = None) -> list[dict[str, Any]]:
    """读取工具注册配置。"""
    raw = _load(Path(path) if path else PROJECT_ROOT / "configs" / "tools.yaml")
    return raw.get("tools", []) or []
