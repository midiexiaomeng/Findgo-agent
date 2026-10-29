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
from .constraints import Constraints, parse_from_text
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


RESET_KEYWORDS = (
    "从头",
    "重新找",
    "重新搜",
    "重新检索",
    "忽略之前",
    "不管之前",
    "清空排除",
    "重来",
)


def detect_reset(query: str) -> bool:
    """用户是否要求忘掉之前给过的结果，从全库里重新找。"""
    text = (query or "").lower()
    return any(word in text for word in RESET_KEYWORDS)


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
        extra_context: str = "",
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
            "请把用户的话拆成结构化约束。核心原则：**约束优先于关键词**。\n"
            "否定、排除、偏好都要单独列出来，不要塞进 core_query 里。\n\n"
            "字段说明：\n"
            "- core_query：画面里有什么，只写正向内容，**不带数量词、不带否定词**。\n"
            "  用户说「找两张白色的猫，不要黑的」，core_query 是「猫」，\n"
            "  must 里放「白色」，must_not 里放「黑色」。\n"
            "- must：必须满足的条件（颜色、动作、场景、题材），不满足就淘汰。\n"
            "  用户说「一定要」「必须是」「只要」时放这里。\n"
            "- must_not：必须排除的条件，命中就淘汰。\n"
            "  用户说「不要」「别」「排除」「不是」时放这里。\n"
            "  指代之前的图（「刚才那几张不要」）由系统自动处理，不用写进来。\n"
            "- should：加分但非必需的偏好。用户说「最好」「尽量」「希望」时放这里。\n"
            "- retrieve_count：要返回几张图。与图片内容里有几个对象无关。\n"
            "  没提数量填 0。\n"
            "- diversity：需要多样化结果时填 true（默认），\n"
            "  用户说「就要同一种风格」「都差不多的」时填 false。\n"
            "- path：用户提到的文件夹路径，原样保留。没提填空字符串。\n\n"
            "现有工具覆盖：文本搜图、以图搜图、建/删索引、索引统计、导出清单、"
            "找重复图、暂存区管理、单图描述、目录扫描。\n"
            "超出这些能力且能用简单只读 Python 实现时，可附加 dynamic_tool 字段：\n"
            '{"name":"...","description":"...","tags":["read-only"],'
            '"timeout":60,"args_schema_code":"class Args(BaseModel):\\n    ...",'
            '"run_func_code":"def run(self, args):\\n    ...","safety_note":"..."}\n'
            "动态工具禁止硬编码具体路径，路径从参数传入并用 sandbox_check_path 校验。\n\n"
            "只输出 JSON：\n"
            '{"tasks":[{"id":1,"core_query":"猫","must":["白色"],'
            '"must_not":["黑色"],"should":["躺着"],"retrieve_count":2,'
            '"diversity":true,"description":"找两张白色的猫","path":""}]}'
        )

        if extra_context.strip():
            prompt += "\n\n【背景信息（仅供参考，不是本轮指令）】\n" + extra_context.strip()
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

            # 新结构：core_query + 三类约束；旧字段 query 兼容回退
            core = str(item.get("core_query") or item.get("query") or clean_query)
            try:
                count = int(item.get("retrieve_count", item.get("top_k", 0)) or 0)
            except (TypeError, ValueError):
                count = 0

            constraints = Constraints(
                core_query=core,
                must=(
                    [str(x).strip() for x in item.get("must", []) if str(x).strip()]
                    if isinstance(item.get("must"), list)
                    else []
                ),
                must_not=(
                    [str(x).strip() for x in item.get("must_not", []) if str(x).strip()]
                    if isinstance(item.get("must_not"), list)
                    else []
                ),
                should=(
                    [str(x).strip() for x in item.get("should", []) if str(x).strip()]
                    if isinstance(item.get("should"), list)
                    else []
                ),
                diversity=bool(item.get("diversity", True)),
                retrieve_count=max(0, min(count, 1000)),
            )

            # LLM 没给约束时，从原句里兜底抽「不要 X」「最好 X」
            if not constraints.must and not constraints.must_not and not constraints.should:
                fallback = parse_from_text(query)
                constraints.must = fallback.must
                constraints.must_not = fallback.must_not
                constraints.should = fallback.should

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
                    description=str(item.get("description") or core),
                    query=core,
                    top_k=constraints.retrieve_count,
                    path=task_path or known_path,
                    all_results=all_results,
                    is_dynamic=is_dynamic,
                    dynamic_spec=dynamic_spec,
                    constraints=constraints.to_dict(),
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
