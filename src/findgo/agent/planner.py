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
from .path_extractor import PathExtractor

logger = logging.getLogger("findgo.planner")


RISK_KEYWORDS = ("删除", "删掉", "清理", "移除", "清空", "purge", "trash")

REUSE_KEYWORDS = (
    "刚才",
    "上次",
    "上一次",
    "再发",
    "重新发",
    "重复",
    "还是那",
    "同样那",
    "刚找的",
)


def detect_reuse(query: str) -> bool:
    """用户是否在要求复用上一轮的图片，而不是重新检索。"""
    text = (query or "").lower()
    return any(word in text for word in REUSE_KEYWORDS)


def assess_risk(query: str) -> str:
    """在规划阶段就标出高风险意图，供 UI 提前确认。"""
    text = (query or "").lower()
    return "high" if any(word in text for word in RISK_KEYWORDS) else "normal"


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
        extra_context: str = "",
    ) -> AgentAction:
        """生成一个动作。

        Args:
            query: 用户本轮输入
            tools_desc: 工具清单文本
            history: 近期对话（可选，供多轮使用）
            extra_context: 当前会话的结构化事实，优先级高于历史约束
        """
        template = self.prompts.get("react_template", "")
        if template:
            user_content = template.format(tools=tools_desc, query=query)
        else:
            user_content = f"可用工具：\n{tools_desc}\n\n用户请求：{query}"

        if extra_context.strip():
            user_content += (
                "\n\n【当前会话生效约束（权威源，优先级高于历史对话）】\n" + extra_context.strip()
            )

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
        # 正则优先提取路径，LLM 只做确认与语义拆解
        extractor = PathExtractor()
        candidates = extractor.candidates(query)
        known_path = extractor.resolve_with_llm(query, candidates, self.client)
        clean_query = extractor.strip_paths(query, candidates) if candidates else query

        if known_path:
            path_hint = f"已识别路径：{known_path}（请直接使用）"
        elif candidates:
            path_hint = f"候选路径（未能确认）：{candidates}"
        else:
            path_hint = "用户没有提到文件夹路径"

        prompt = (
            f"可用工具：\n{tools_desc}\n\n"
            f"用户需求：{query}\n"
            f"路径信息：{path_hint}\n\n"
            "请先理解用户的语言，提取其中的明确约束（数量、场景、颜色、时间等），"
            "再拆解为 1 到 3 个可在本地图片库执行的检索任务。\n"
            "现有工具只覆盖：文本搜图、以图搜图、建/删索引、索引统计、导出清单、"
            "找重复图、暂存区管理、单图描述、目录扫描。\n"
            "如果用户需求明显超出这些能力（例如统计图片宽高、按拍摄时间筛选、"
            "按颜色聚类），并且能用简单的只读 Python 实现，"
            "可以在任务里附加 dynamic_tool 字段，格式：\n"
            '{"name":"...","description":"...","tags":["read-only"],'
            '"timeout":60,"args_schema_code":"class Args(BaseModel):\\n    ...",'
            '"run_func_code":"def run(self, args):\\n    ...",'
            '"safety_note":"..."}\n'
            "动态工具禁止硬编码具体文件名或固定路径，路径一律从参数传入，"
            '并用 sandbox_check_path(路径, "read") 校验。\n'
            "能用现有工具完成时不要生成 dynamic_tool。\n"
            "每个任务包含：\n"
            "- description：一句话说明这次找什么\n"
            "- query：用户想找的图片的完整描述，保留数量词和修饰语"
            "（如「两张猫的图片」），不要只留关键词\n"
            "- top_k：用户明确要求的图片数量。用户说“两张”就填 2，"
            "说“几张”或没提数量就填 0（表示用系统默认值）\n"
            "- path：用户提到的文件夹路径，原样保留（如 D:\\Photos）。"
            "用户没提路径就填空字符串\n"
            "- all_results：用户说“所有”“全部”“都要”“都找出来”时填 true，"
            "表示返回全部匹配结果；否则填 false\n\n"
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
            q = str(item.get("query") or item.get("description") or clean_query)
            try:
                top_k = int(item.get("top_k") or 0)
            except (TypeError, ValueError):
                top_k = 0
            task_path = str(item.get("path") or "").strip()
            all_results = bool(item.get("all_results", False))

            dynamic_spec = None
            is_dynamic = False
            raw_dynamic = item.get("dynamic_tool")
            if isinstance(raw_dynamic, dict) and raw_dynamic.get("name"):
                try:
                    from .dynamic import DynamicToolSpec

                    dynamic_spec = DynamicToolSpec(**raw_dynamic).model_dump()
                    is_dynamic = True
                except Exception as e:
                    logger.warning("动态工具定义不合法，已忽略: %s", e)
            result.append(
                TaskSpec(
                    id=int(item.get("id", i)),
                    description=str(item.get("description") or q),
                    query=q,
                    top_k=max(0, min(top_k, 1000)),
                    path=task_path or known_path,
                    all_results=all_results,
                    is_dynamic=is_dynamic,
                    dynamic_spec=dynamic_spec,
                )
            )
        risk = assess_risk(query)
        reuse = detect_reuse(query)
        for task in result:
            task.risk = "high" if task.is_dynamic else risk
            task.reuse_last = reuse

        if not result:
            return [TaskSpec(id=1, description=query, query=query, risk=risk)]

        # 只有一个任务且带显式路径时，拆成「打开目录」「检索图片」两步；
        # 多个任务时每个任务保持一步，路径扫描在检索工具内部完成。
        if len(result) == 1 and result[0].path:
            target = result[0]
            return [
                TaskSpec(
                    id=1,
                    action="scan",
                    description=f"打开 {target.path}",
                    query="",
                    path=target.path,
                    risk=target.risk,
                ),
                TaskSpec(
                    id=2,
                    action="retrieve",
                    description=target.description,
                    query=target.query,
                    top_k=target.top_k,
                    path=target.path,
                    all_results=target.all_results,
                    risk=target.risk,
                ),
            ]

        return result

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
