"""LLM 客户端：一套接口，三种后端。

- ``ollama``  : 本地 Ollama 服务，最省事，qwen2.5:7b-instruct 默认走这里
- ``openai``  : 任何 OpenAI 兼容端点（vLLM / LM Studio / llama.cpp server / 云端）
- ``none``    : 显式关闭

只依赖 requests，不强制安装 openai SDK。
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Sequence

from ..config import Config

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """LLM 调用失败。"""


class LLMClient:
    def __init__(self, cfg: Config):
        self.full_cfg = cfg
        self.cfg = cfg.rerank
        self.backend = (cfg.rerank.backend or "none").lower()
        self.model = cfg.rerank.model
        self._timeout = int(
            cfg.rerank.ollama.timeout
            if self.backend == "ollama"
            else cfg.rerank.openai.timeout
        )
        # 可用性探测缓存：(时间戳, 结果)，避免每次检索都发探测请求
        self._avail_cache: tuple[float, tuple[bool, str]] = (0.0, (False, "未检查"))
        self._hf = None

    @property
    def api_key(self) -> str:
        """统一取 Key：环境变量优先于配置文件。"""
        return (
            os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or self.cfg.openai.api_key
            or "x"
        )

    # ── 对外 ────────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        if self.backend == "transformers":
            return self._hf_client().chat(messages, max_tokens=max_tokens)
        if self.backend == "ollama":
            return self._chat_ollama(messages, json_mode, max_tokens)
        if self.backend == "openai":
            return self._chat_openai(messages, json_mode, max_tokens)
        raise LLMError(f"未启用 LLM 后端: {self.backend}")

    def _hf_client(self):
        if self._hf is None:
            from .transformers_backend import TransformersChat

            self._hf = TransformersChat(self.full_cfg)
        return self._hf

    def is_available(self, ttl: float = 15.0) -> tuple[bool, str]:
        """返回 (是否可用, 说明)。只做轻量探测，不加载模型。

        结果缓存 ttl 秒。同一轮搜索里不会重复探测，也不会因为服务没启动
        而在毎一批候选上各撞一次超时。
        """
        now = time.time()
        cached_at, cached = self._avail_cache
        if now - cached_at < ttl:
            return cached

        result = self._probe()
        self._avail_cache = (now, result)
        return result

    def _probe(self) -> tuple[bool, str]:
        if self.backend in ("none", ""):
            return False, "已在配置中关闭"

        if self.backend == "transformers":
            return self._hf_client().is_available()

        import requests

        try:
            if self.backend == "ollama":
                base = self.cfg.ollama.base_url.rstrip("/")
                resp = requests.get(f"{base}/api/tags", timeout=5)
                resp.raise_for_status()
                names = [m.get("name", "") for m in resp.json().get("models", [])]
                if not names:
                    return False, "Ollama 已启动，但还没有下载任何模型"
                # qwen2.5:7b-instruct 与 qwen2.5:7b 视为同一模型
                want = self.model
                hit = any(n == want for n in names)
                if not hit:
                    return False, f"Ollama 未找到模型 {want}，请先 ollama pull {want}"
                return True, f"Ollama 就绪，模型 {want}"

            base = self.cfg.openai.base_url.rstrip("/")
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.get(f"{base}/models", headers=headers, timeout=5)
            if resp.status_code >= 400:
                return False, f"OpenAI 兼容端点返回 {resp.status_code}"
            return True, f"OpenAI 兼容端点就绪 ({base})"
        except Exception as e:
            return False, f"无法连接 {self.backend}: {e}"

    # ── Ollama ──────────────────────────────────────────────────────────────

    def _chat_ollama(
        self,
        messages: Sequence[dict[str, str]],
        json_mode: bool,
        max_tokens: int | None,
    ) -> str:
        import requests

        base = self.cfg.ollama.base_url.rstrip("/")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "options": {
                "temperature": float(self.cfg.temperature),
                "num_predict": int(max_tokens or self.cfg.max_tokens),
            },
        }
        if json_mode:
            payload["format"] = "json"

        try:
            resp = requests.post(
                f"{base}/api/chat", json=payload, timeout=self._timeout
            )
        except requests.RequestException as e:
            raise LLMError(f"请求 Ollama 失败: {e}") from e

        if resp.status_code >= 400:
            raise LLMError(f"Ollama 返回 {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        content = (data.get("message") or {}).get("content", "")
        if not content:
            raise LLMError(f"Ollama 返回空内容: {str(data)[:300]}")
        return str(content)

    # ── OpenAI 兼容 ─────────────────────────────────────────────────────────

    def _chat_openai(
        self,
        messages: Sequence[dict[str, str]],
        json_mode: bool,
        max_tokens: int | None,
    ) -> str:
        import requests

        base = self.cfg.openai.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            url = base
        elif base.endswith("/v1"):
            url = f"{base}/chat/completions"
        else:
            url = f"{base}/v1/chat/completions"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": float(self.cfg.temperature),
            "max_tokens": int(max_tokens or self.cfg.max_tokens),
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self._timeout)
        except requests.RequestException as e:
            raise LLMError(f"请求 {url} 失败: {e}") from e

        if resp.status_code >= 400:
            raise LLMError(f"LLM 端点返回 {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"无法解析 LLM 响应: {str(data)[:300]}") from e

    # ── 描述 ────────────────────────────────────────────────────────────────

    def describe_backend(self) -> str:
        if self.backend == "transformers":
            return f"本地 Transformers · {self.cfg.transformers.model_path}"
        if self.backend == "ollama":
            return f"Ollama · {self.model} · {self.cfg.ollama.base_url}"
        if self.backend == "openai":
            return f"OpenAI 兼容 · {self.model} · {self.cfg.openai.base_url}"
        return "未启用"
