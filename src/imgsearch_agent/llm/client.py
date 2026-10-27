"""Agent 层聊天客户端：支持 OpenAI 兼容 API 与 Ollama。

只做一件事：把 messages 发出去，拿回文本。不关心检索。
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

logger = logging.getLogger("imgsearch_agent.llm")


class ChatClient:
    def __init__(self, cfg: dict[str, Any]):
        self.backend = (cfg.get("backend") or "openai").lower()
        self.model = cfg.get("model") or ""
        self.base_url = (cfg.get("base_url") or "").rstrip("/")
        self.api_key = cfg.get("api_key") or "not-needed"
        self.timeout = int(cfg.get("timeout", 120))

    # ── 对外 ──────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        json_mode: bool = False,
        max_tokens: int = 512,
    ) -> str:
        if self.backend == "openai":
            return self._chat_openai(messages, json_mode, max_tokens)
        if self.backend == "ollama":
            return self._chat_ollama(messages, json_mode, max_tokens)
        raise RuntimeError(f"不支持的 LLM 后端: {self.backend}（可选 openai / ollama）")

    def is_available(self) -> tuple[bool, str]:
        import requests

        try:
            if self.backend == "openai":
                resp = requests.get(
                    self._openai_url().replace("/chat/completions", "/models"),
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=10,
                )
                return resp.status_code < 400, f"{self.base_url} -> {resp.status_code}"
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            return resp.status_code == 200, f"{self.base_url} -> {resp.status_code}"
        except Exception as e:
            return False, f"连接失败: {e}"

    def describe(self) -> str:
        return f"{self.backend} · {self.model} · {self.base_url}"

    # ── OpenAI 兼容 ───────────────────────────────────────────────────────

    def _openai_url(self) -> str:
        base = self.base_url
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _chat_openai(
        self,
        messages: Sequence[dict[str, str]],
        json_mode: bool,
        max_tokens: int,
    ) -> str:
        import requests

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": 0,
            "max_tokens": int(max_tokens),
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        resp = requests.post(
            self._openai_url(),
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"API 返回 {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"无法解析 API 响应: {str(data)[:300]}") from e

    # ── Ollama ────────────────────────────────────────────────────────────

    def _chat_ollama(
        self,
        messages: Sequence[dict[str, str]],
        json_mode: bool,
        max_tokens: int,
    ) -> str:
        import requests

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "options": {"temperature": 0, "num_predict": int(max_tokens)},
        }
        if json_mode:
            payload["format"] = "json"

        resp = requests.post(
            f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Ollama 返回 {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        content = (data.get("message") or {}).get("content", "")
        if not content:
            raise RuntimeError(f"Ollama 返回空内容: {str(data)[:300]}")
        return str(content)
