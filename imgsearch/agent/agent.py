"""图像检索 Agent：在检索系统之上加一层自然语言决策循环。

工作方式：
    用户自然语言 → 决策（调哪个工具 / 直接回答）→ 执行工具 →
    结果总结（自然语言）→ 返回回复 + 结构化图片结果。

不依赖任何第三方 Agent 框架，决策与总结都复用系统的 LLM 后端，
因此 Ollama / OpenAI 兼容 / 本地 Transformers 三种后端都能直接驱动 Agent。

设计上：
- 决策失败时退化为直接按用户原话做一次文本检索，保证可用性；
- 返回的 AgentResponse 同时携带自然语言回复和 SearchResult 列表，
  界面可以既显示对话气泡，也渲染图片墙。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from .tools import build_tools, tools_to_prompt

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    reply: str
    tool: str = ""
    results: list = field(default_factory=list)
    data: dict = field(default_factory=dict)


SYSTEM = (
    "你是本地图片库的检索助手。根据用户请求判断：如果需要检索图片或查询库信息，"
    "调用工具；如果是闲聊或打招呼，直接回答。你只输出一个 JSON 对象，不要输出其他文字。"
)


class ImageSearchAgent:
    def __init__(self, pipeline, client=None):
        self.pipeline = pipeline
        # 默认复用 pipeline 重排器的 LLM 客户端。若另行新建，会重复加载一份
        # 大模型到显存（transformers 后端尤其明显），因此这里共享同一个实例。
        self.client = client if client is not None else pipeline.reranker.client
        self.tools = {t.name: t for t in build_tools(pipeline)}
        self.history: list[dict[str, str]] = []

    # ── 决策 ──────────────────────────────────────────────────────────────

    def _decision_prompt(self, user_input: str) -> str:
        lines = [
            "可用工具：",
            tools_to_prompt(list(self.tools.values())),
            "",
            f"用户请求：{user_input}",
            "",
            "判断该调用哪个工具，输出 JSON：",
            '{"tool": "search_images", "args": {"query": "海边的日落"}}',
            '或 {"tool": "get_stats", "args": {}}',
            '或直接回答 {"tool": "none", "reply": "..."}',
        ]
        return "\n".join(lines)

    @staticmethod
    def _parse_json(raw: str) -> dict | None:
        text = (raw or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None

    def _decide(self, user_input: str) -> dict:
        try:
            raw = self.client.chat(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": self._decision_prompt(user_input)},
                ],
                json_mode=True,
                max_tokens=256,
            )
            decision = self._parse_json(raw)
            if decision and "tool" in decision:
                return decision
        except Exception as e:
            logger.warning("Agent 工具决策失败: %s", e)

        # 决策失败或格式异常：退化为直接文本检索，保证不冷场
        logger.info("决策回退为直接文本检索")
        return {"tool": "search_images", "args": {"query": user_input}}

    # ── 总结 ──────────────────────────────────────────────────────────────

    def _summarize(self, user_input: str, tool_name: str, summary: str) -> str:
        try:
            prompt = (
                f"用户请求：{user_input}\n"
                f"工具 {tool_name} 的结果：{summary}\n"
                "请用一两句中文，简洁地告诉用户检索结果。"
            )
            reply = self.client.chat(
                [
                    {"role": "system", "content": "你是简洁的检索助手，用中文回答。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=128,
            ).strip()
            return reply or f"已为你完成 {tool_name}。"
        except Exception as e:
            logger.warning("结果总结失败: %s", e)
            return f"已为你完成 {tool_name}：{summary}"

    # ── 主循环 ────────────────────────────────────────────────────────────

    def run(self, user_input: str) -> AgentResponse:
        user_input = (user_input or "").strip()
        if not user_input:
            return AgentResponse(reply="请告诉我你想找什么图片。")

        decision = self._decide(user_input)
        tool_name = decision.get("tool", "none")

        if tool_name == "none":
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": decision.get("reply", "")})
            return AgentResponse(reply=decision.get("reply", "请告诉我你想找什么图片。"))

        tool = self.tools.get(tool_name)
        if tool is None:
            return AgentResponse(reply=f"抱歉，我暂时无法处理这个请求（未知工具 {tool_name}）。")

        args = decision.get("args") or {}
        try:
            payload = tool.func(**args)
        except TypeError:
            # 参数名或数量对不上时，只用第一个必填参数重试一次
            payload = self._retry_call(tool, args)
        except Exception as e:
            logger.warning("工具 %s 执行失败: %s", tool_name, e)
            return AgentResponse(reply=f"执行 {tool_name} 时出错：{e}")

        results = payload.get("results", [])
        summary = payload.get("summary", "")
        reply = self._summarize(user_input, tool_name, summary)

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": reply})

        return AgentResponse(
            reply=reply,
            tool=tool_name,
            results=results,
            data=payload.get("items", []),
        )

    def _retry_call(self, tool, args: dict) -> dict:
        """参数不匹配时的兜底：按工具参数声明的顺序逐个传入。"""
        for key in tool.parameters:
            if key in args:
                try:
                    return tool.func(**{key: args[key]})
                except Exception:
                    continue
        raise RuntimeError("工具参数不匹配")
