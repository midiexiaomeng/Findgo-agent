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


#: 只有出现这些词，才允许生成建库任务
INDEX_INTENT = (
    "建索引",
    "建库",
    "入库",
    "加入图库",
    "加入索引",
    "生成向量库",
    "收进索引",
    "建立索引",
    "更新索引",
    "重建索引",
)


def has_index_intent(query: str) -> bool:
    """用户是不是明确要求建库。没有这个意图就不该出现 index 任务。"""
    text = (query or "").lower()
    return any(word in text for word in INDEX_INTENT)


#: 在上一轮结果里继续筛的表述
REFINE_KEYWORDS = (
    "刚才的",
    "上面这些",
    "这些里面",
    "从里面",
    "在结果里",
    "从中",
    "这几个里",
    "这几张里",
    "里面挑",
    "里面找",
    "再筛",
)


def detect_refine(query: str) -> bool:
    """用户是否要求基于上一轮结果继续筛，而不是重新全库检索。"""
    text = (query or "").lower()
    return any(word in text for word in REFINE_KEYWORDS)


#: 需要多轮探索才知道下一步的任务
EXPLORE_KEYWORDS = (
    "分类",
    "整理",
    "分组",
    "按内容",
    "按类型",
    "逐个",
    "归纳",
    "归档",
    "标注",
    "遍历",
    "汇总",
    "有哪些",
    "分别",
    "一个个",
)


def needs_exploration(query: str) -> bool:
    """任务是否无法一次拆解完，需要看中间结果再决定下一步。"""
    text = (query or "").lower()
    return any(word in text for word in EXPLORE_KEYWORDS)


def assess_risk(query: str) -> str:
    """在规划阶段就标出高风险意图，供 UI 提前确认。"""
    text = (query or "").lower()
    return "high" if any(word in text for word in RISK_KEYWORDS) else "normal"


class Planner:
    def __init__(self, client, prompts: dict[str, str], max_iterations: int = 5, sandbox=None):
        self.client = client
        self.prompts = prompts or {}
        self.max_iterations = max_iterations
        self.sandbox = sandbox

    # ── 决策 ──────────────────────────────────────────────────────────────

    def plan(
        self,
        query: str,
        tools_desc: str,
        history: Sequence[dict[str, str]] = (),
        extra_context: str = "",
        steps: Sequence[dict] = (),
    ) -> AgentAction:
        """生成一个动作。

        Args:
            query: 用户本轮输入
            tools_desc: 工具清单文本
            history: 近期对话（可选，供多轮使用）
            extra_context: 当前会话的结构化事实，优先级高于历史约束
            steps: ReAct 已执行的步骤，每项含 thought / action / observation
        """
        template = self.prompts.get("react_template", "")
        if template:
            user_content = template.format(tools=tools_desc, query=query)
        else:
            user_content = f"可用工具：\n{tools_desc}\n\n用户请求：{query}"

        if steps:
            lines = []
            for i, step in enumerate(steps, 1):
                thought = str(step.get("thought", ""))[:120]
                action = str(step.get("action", ""))[:120]
                observation = str(step.get("observation", ""))[:400]
                lines.append(
                    f"步骤 {i}\n  思考：{thought}\n  动作：{action}\n  观察：{observation}"
                )
            user_content += (
                "\n\n【已执行的步骤】\n"
                + "\n".join(lines)
                + "\n\n基于上面的观察决定下一步。信息够了就直接给 answer，不要重复已经做过的动作。"
            )

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
        extractor = PathExtractor(sandbox=getattr(self, "sandbox", None))
        candidates = extractor.candidates(query)
        known_path = extractor.resolve_with_llm(query, candidates, self.client)
        clean_query = extractor.strip_paths(query, candidates) if candidates else query

        # 最后一道：LLM 确认的路径也要过沙箱
        if known_path and not extractor._allowed(known_path):
            logger.info("LLM 确认的路径被沙箱拒绝: %s", known_path)
            known_path = ""

        if known_path:
            path_hint = f"已识别路径：{known_path}（请直接使用）"
        elif candidates:
            path_hint = f"候选路径（未能确认）：{candidates}"
        else:
            path_hint = "用户没有提到文件夹路径"

        prompt = (
            "【硬性铁则，必须严格遵守】\n"
            "1. 检索任务永远不生成 index 任务。只有用户亲口说出：\n"
            "   「建索引」「入库」「加入图库」「生成向量库」「建库」时才可生成。\n"
            "   用户说「在某个文件夹找图」时，只生成检索任务，不要预判目录状态。\n"
            "   目录有没有索引、要不要建库，由执行层自己判断，不归你管。\n"
            "2. 一条指令只能生成一类任务。检索与建库不能出现在同一个 tasks 里。\n"
            "3. 字段隔离：core_query 只放正向主体名词。\n"
            "   数量→retrieve_count；否定（不要/排除/不是）→must_not；\n"
            "   硬条件（必须/只要/一定要）→must；软偏好（最好/尽量）→should。\n"
            "   严禁把数字、否定词、数量词写进 core_query。\n\n"
            f"可用工具：\n{tools_desc}\n\n"
            f"用户需求：{query}\n"
            f"路径信息：{path_hint}\n\n"
            "字段说明：\n"
            "- core_query：画面里有什么，只写正向内容。\n"
            "  「找两张白色的猫，不要黑的」→ core_query 是「猫」\n"
            "- must：不满足就淘汰的硬条件\n"
            "- must_not：命中就淘汰的排除条件。指代之前的图由系统处理，不用写\n"
            "- should：加分但非必需\n"
            "- retrieve_count：返回几张图，没提填 0\n"
            "- diversity：需要多样化填 true，用户说「就要差不多的」填 false\n"
            "- path：文件夹路径，没提填空字符串\n"
            "- 用户问「有多少」「几张」「统计」「一共几个」时，\n"
            "  把 action 设为 count，这样只统计数量不返回图片列表。\n"
            "  需要同时知道目录总图数和匹配数时，拆成两个任务：\n"
            "  一个 action=scan 统计总数，一个 action=count 统计匹配数。\n"
            "- 其余情况不需要写 action。\n\n"
            "超出工具能力且能用简单只读 Python 实现时，可附加 dynamic_tool：\n"
            '{"name":"...","description":"...","tags":["read-only"],'
            '"timeout":60,"args_schema_code":"class Args(BaseModel):\\n    ...",'
            '"run_func_code":"def run(self, args):\\n    ...","safety_note":"..."}\n'
            "动态工具只能读，不得写文件、删文件；路径从参数传入并用 sandbox_check_path 校验。\n\n"
            "输出规则（违反则无效）：\n"
            "① 只返回纯 JSON，不要解释、不要 markdown、不要 ``` 代码块\n"
            "② 严格按样例字段，不新增字段\n"
            '③ 结构固定为 {"tasks":[...]}\n\n'
            "样例：\n"
            "用户输入：找两张白色的猫，不要黑色的，最好是躺着的\n"
            '输出：{"tasks":[{"id":1,"core_query":"猫","must":["白色"],'
            '"must_not":["黑色"],"should":["躺着"],"retrieve_count":2,'
            '"diversity":true,"description":"查找两张白色猫咪图片","path":""}]}'
        )

        if extra_context.strip():
            prompt += (
                "\n\n【背景信息：仅作上下文参考，不参与任务拆解】\n"
                "下列内容不得转化为额外约束，也不得据此新增任务。\n" + extra_context.strip()
            )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.prompts.get("system", "")},
        ]
        for item in history[-10:]:
            messages.append(item)
        messages.append({"role": "user", "content": prompt})

        # JSON 偶尔带 markdown 围栏或残缺，重试一次；仍失败就降级为正则兜底
        data = None
        for attempt in range(2):
            try:
                raw = self.client.chat(messages, json_mode=True, max_tokens=800)
            except Exception as e:
                logger.warning("任务拆解调用失败（第 %d 次）: %s", attempt + 1, e)
                continue

            parsed = self._parse(raw)
            if isinstance(parsed, dict) and isinstance(parsed.get("tasks"), list):
                data = parsed
                break
            logger.warning("任务 JSON 解析失败（第 %d 次），raw=%s", attempt + 1, str(raw)[:160])

        if data is None:
            logger.info("两次解析均失败，降级为正则拆解")
            fallback = parse_from_text(query)
            return [
                TaskSpec(
                    id=1,
                    description=query,
                    query=fallback.core_query or query,
                    constraints=fallback.to_dict(),
                )
            ]

        tasks = data.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            return [TaskSpec(id=1, description=query, query=query)]

        result: list[TaskSpec] = []
        for i, item in enumerate(tasks[:3], 1):
            if not isinstance(item, dict):
                continue

            # 新结构：core_query + 三类约束；旧字段 query 兼容回退
            core = str(item.get("core_query") or item.get("query") or clean_query)

            # 代码层强制清洗：LLM 把数量词/否定词写进 core_query 时拉到正确字段
            from .constraints import clean_core_query

            cleaned, extra_neg, extra_pref = clean_core_query(core)
            if cleaned:
                core = cleaned

            # LLM 偶尔返回空字符串，兜底用原句，否则检索直接空转
            if not core.strip():
                core = clean_query or query
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

            # 把清洗出来的词补回对应字段，去重
            for term in extra_neg:
                if term not in constraints.must_not:
                    constraints.must_not.append(term)
            for term in extra_pref:
                if term not in constraints.should:
                    constraints.should.append(term)

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
        refine = detect_refine(query)
        index_mode = has_index_intent(query)
        for task in result:
            if task.constraints:
                logger.info(
                    "约束解析: core=%r must=%s must_not=%s should=%s count=%s",
                    task.constraints.get("core_query"),
                    task.constraints.get("must"),
                    task.constraints.get("must_not"),
                    task.constraints.get("should"),
                    task.constraints.get("retrieve_count"),
                )
            task.risk = "high" if task.is_dynamic else risk
            task.reuse_last = reuse

            # 在上一轮结果里继续筛，不重新召回全库
            if refine and not reuse:
                task.input_source = "last_results"
            else:
                task.input_source = "library"

            # 建库是独立动作：只有用户明确要求，且有大目录时才转成 index
            if index_mode and task.path and not task.is_dynamic:
                task.action = "index"
            elif task.action == "index" and not index_mode:
                # LLM 自己编出来的建库任务，一律降回检索
                logger.info("用户未要求建库，已忽略模型生成的 index 任务")
                task.action = "retrieve"

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
