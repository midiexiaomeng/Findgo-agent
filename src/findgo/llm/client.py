"""Agent 层聊天客户端。

支持：
- OpenAI 兼容 API（DeepSeek V4 Flash、通义、月之暗面、OpenAI 等）
- Ollama 本地服务

针对 DeepSeek V4 系列的兼容处理：
- 模型名可配：deepseek-flash（V4.1 Flash）或 deepseek-v4-flash
- 推理模型可能返回 reasoning_content，content 为空时回退取它
- 部分版本不支持 response_format，JSON mode 被拒时自动降级重试
- 可选透传 reasoning_effort / extra_body
"""

from __future__ import annotations

import logging
import os
from typing import Any, Sequence

logger = logging.getLogger("findgo.llm")


class ChatClient:
    def __init__(self, cfg: dict[str, Any]):
        self.backend = (cfg.get("backend") or "openai").lower()
        self.model = cfg.get("model") or ""
        self.base_url = (cfg.get("base_url") or "").rstrip("/")
        self.api_key = (
            os.environ.get("DEEPSEEK_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
            or cfg.get("api_key")
            or "not-needed"
        )
        self.timeout = int(cfg.get("timeout", 120))
        # DeepSeek V4 可选 low / high / max，留空则不传
        self.reasoning_effort = cfg.get("reasoning_effort") or ""
        # 其他透传参数（例如某些平台的 custom 字段）
        self.extra_body: dict[str, Any] = cfg.get("extra_body") or {}

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
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.extra_body:
            payload.update(self.extra_body)
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        resp = requests.post(
            self._openai_url(), json=payload, headers=headers, timeout=self.timeout
        )

        # 部分 DeepSeek 版本不接受 response_format，去掉后重试一次
        if json_mode and resp.status_code in (400, 422):
            body = (resp.text or "").lower()
            if "response_format" in body or "json" in body:
                logger.info("JSON mode 被拒绝，降级为普通请求重试")
                payload.pop("response_format", None)
                resp = requests.post(
                    self._openai_url(),
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )

        if resp.status_code >= 400:
            raise RuntimeError(f"API 返回 {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"无法解析 API 响应: {str(data)[:300]}") from e

        content = message.get("content")
        if not content:
            # 推理模型（DeepSeek V4 / R1 系列）content 可能为空
            content = message.get("reasoning_content") or ""
        if not content:
            raise RuntimeError(f"模型返回空内容: {str(data)[:300]}")
        return str(content)

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
