"""模型配置存储：管理用户添加并测试通过的模型列表。

文件位置：configs/models.json（本地，已被 .gitignore 排除）
只有测试调用成功的模型才会被写入，主界面只展示这份列表。
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 主流 OpenAI 兼容服务预设
PROVIDERS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-flash", "deepseek-chat", "deepseek-reasoner"],
    },
    "dashscope": {
        "name": "阿里百炼",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen-plus", "qwen-turbo", "qwen-max", "qwen-long"],
    },
    "moonshot": {
        "name": "月之暗面",
        "base_url": "https://api.moonshot.cn/v1",
        "models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
    },
    "zhipu": {
        "name": "智谱",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["glm-4-flash", "glm-4-air", "glm-4-plus", "glm-4-long"],
    },
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "o4-mini"],
    },
    "custom": {
        "name": "自定义 / 本地服务",
        "base_url": "http://localhost:8000/v1",
        "models": [],
    },
}


class ModelStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else PROJECT_ROOT / "configs" / "models.json"
        self.models: list[dict[str, Any]] = []
        self.active: str = ""
        self.load()

    # ── 读写 ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.models = [m for m in data.get("models", []) if isinstance(m, dict)]
            self.active = str(data.get("active", ""))
        except Exception:
            self.models = []
            self.active = ""

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"models": self.models, "active": self.active}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 操作 ──────────────────────────────────────────────────────────────

    def add(
        self,
        provider: str,
        model: str,
        base_url: str,
        api_key: str,
        label: str = "",
    ) -> dict[str, Any]:
        entry = {
            "id": uuid.uuid4().hex[:8],
            "provider": provider,
            "model": model,
            "base_url": base_url.rstrip("/"),
            "api_key": api_key,
            "label": label or f"{PROVIDERS.get(provider, {}).get('name', provider)} · {model}",
        }
        # 同一 provider+model+base_url 视为重复，覆盖
        self.models = [
            m
            for m in self.models
            if not (
                m.get("provider") == provider
                and m.get("model") == model
                and m.get("base_url") == entry["base_url"]
            )
        ]
        self.models.append(entry)
        if not self.active:
            self.active = entry["id"]
        self.save()
        return entry

    def remove(self, model_id: str) -> None:
        self.models = [m for m in self.models if m.get("id") != model_id]
        if self.active == model_id:
            self.active = self.models[0]["id"] if self.models else ""
        self.save()

    def set_active(self, model_id: str) -> None:
        if any(m.get("id") == model_id for m in self.models):
            self.active = model_id
            self.save()

    def get_active(self) -> dict[str, Any] | None:
        for m in self.models:
            if m.get("id") == self.active:
                return m
        return self.models[0] if self.models else None

    # ── 转成 ChatClient 配置 ──────────────────────────────────────────────

    @staticmethod
    def to_llm_config(entry: dict[str, Any]) -> dict[str, Any]:
        return {
            "backend": "openai",
            "model": entry.get("model", ""),
            "base_url": entry.get("base_url", ""),
            "api_key": entry.get("api_key", ""),
            "timeout": 120,
        }
