"""ReAct 规划器：根据用户意图与工具列表，输出结构化动作。

只负责"想"，不负责"做"：不调用工具、不执行业务逻辑。
所有输出都被解析成 AgentAction，拒绝裸字符串。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Sequence

from ..utils.schemas import AgentAction

logger = logging.getLogger("imgsearch_agent.planner")


class Planner:
    def __init__(self, client, prompts: dict[str, str], max_iterations: int = 5):
        self.client = client
        self.prompts = prompts or {}
        self.max_iterations = max_iterations

    # ── 决策 ──────────────────────────────────────────────────────────────

    def plan(
        self,
        query: str,
        tools_desc: str,
        history: Sequence[dict[str, str]] = (),
    ) -> AgentAction:
        """生成一个动作。

        Args:
            query: 用户本轮输入
            tools_desc: 工具清单文本
            history: 近期对话（可选，供多轮使用）
        """
        template = self.prompts.get("react_template", "")
        if template:
            user_content = template.format(tools=tools_desc, query=query)
        else:
            user_content = f"可用工具：\n{tools_desc}\n\n用户请求：{query}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.prompts.get("system", "")},
        ]
        for item in history[-6:]:
            messages.append(item)
        messages.append({"role": "user", "content": user_content})

        try:
            raw = self.client.chat(messages, json_mode=True, max_tokens=256)
        except Exception as e:
            logger.warning("规划器调用 LLM 失败: %s", e)
            return AgentAction(
                action_type="answer",
                reply="抱歉，我现在无法完成这个请求，请稍后再试。",
            )

        data = self._parse(raw)
        try:
            return AgentAction(**data)
        except Exception as e:
            logger.warning("动作解析失败: %s | raw=%s", e, raw[:200])
            return AgentAction(
                action_type="answer",
                reply="抱歉，我没能理解这个请求。",
            )

    # ── 解析 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse(raw: str) -> dict:
        text = (raw or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, dict) and "action_type" in data:
                    return data
            except json.JSONDecodeError:
                pass
        return {"action_type": "answer", "reply": "抱歉，我没能理解这个请求。"}
