"""ReAct 规划器：根据用户意图与工具列表，输出结构化动作。

只负责"想"，不负责"做"：不调用工具、不执行业务逻辑。
所有输出都被解析成 AgentAction，拒绝裸字符串。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Sequence

from ..utils.schemas import AgentAction, TaskSpec

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



    # ── 任务拆解 ──────────────────────────────────────────────────────────

    def plan_tasks(
        self,
        query: str,
        tools_desc: str,
        history: Sequence[dict[str, str]] = (),
    ) -> list[TaskSpec]:
        """把复杂需求拆成 1-3 个检索任务。"""
        prompt = (
            f"可用工具：\n{tools_desc}\n\n"
            f"用户需求：{query}\n\n"
            "请先理解用户的语言，提取其中的明确约束（数量、场景、颜色、时间等），"
            "再拆解为 1 到 3 个可在本地图片库执行的检索任务。\n"
            "每个任务包含：\n"
            "- description：一句话说明这次找什么\n"
            "- query：用于检索的简短中文描述\n"
            "- top_k：用户明确要求的图片数量。用户说“两张”就填 2，"
            "说“几张”或没提数量就填 0（表示用系统默认值）\n"
            "- path：用户提到的文件夹路径，原样保留（如 D:\\Photos）。"
            "用户没提路径就填空字符串\n\n"
            '只输出 JSON：{"tasks":[{"id":1,"description":"...","query":"...","top_k":2}]}'
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.prompts.get("system", "")},
        ]
        for item in history[-10:]:
            messages.append(item)
        messages.append({"role": "user", "content": prompt})

        try:
            raw = self.client.chat(messages, json_mode=True, max_tokens=384)
        except Exception as e:
            logger.warning("任务拆解调用失败: %s", e)
            return [TaskSpec(id=1, description=query, query=query)]

        data = self._parse(raw)
        tasks = data.get("tasks") if isinstance(data, dict) else None
        if not isinstance(tasks, list) or not tasks:
            return [TaskSpec(id=1, description=query, query=query)]

        result: list[TaskSpec] = []
        for i, item in enumerate(tasks[:3], 1):
            if not isinstance(item, dict):
                continue
            q = str(item.get("query") or item.get("description") or query)
            try:
                top_k = int(item.get("top_k") or 0)
            except (TypeError, ValueError):
                top_k = 0
            task_path = str(item.get("path") or "").strip()
            result.append(
                TaskSpec(
                    id=int(item.get("id", i)),
                    description=str(item.get("description") or q),
                    query=q,
                    top_k=max(0, min(top_k, 50)),
                    path=task_path,
                )
            )
        return result or [TaskSpec(id=1, description=query, query=query)]

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
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
        return {"action_type": "answer", "reply": "抱歉，我没能理解这个请求。"}
