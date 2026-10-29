"""Agent 编排器：把 planner、memory、critic、responder 与工具串起来。

流程：
    用户输入 → Planner 决策 → 调工具 → Critic 校验 → 低置信则改写重试
    → Responder 合成 → 写入 Memory → 返回

这是唯一知道"全流程"的模块，其余模块各自独立、可单独测试。
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..config import AgentConfig
from ..llm import ChatClient
from ..tools import (
    CountIndexedImagesTool,
    DescribeImageTool,
    ExportResultsTool,
    FilterResultsTool,
    FindDuplicatesTool,
    FolderScanTool,
    GetIndexStatsTool,
    ImageRetrievalTool,
    ImageSearchByImageTool,
    IndexFolderTool,
    ListIndexedFoldersTool,
    PreviewStagedTool,
    RemoveIndexTool,
    SwitchVectorBackendTool,
    TrashManageTool,
    WebImageSearchTool,
)
from ..tools.base_tool import BaseTool
from ..utils.exceptions import ValidationError
from ..utils.logger import TraceLogger
from ..utils.schemas import AgentAction, TaskSpec
from .critic import Critic
from .custom_tools import load_custom_tools
from .executor import ToolExecutor
from .memory import Memory
from .planner import Planner
from .responder import Responder

logger = logging.getLogger("findgo.orchestrator")

#: 事实 patch 抽取提示词（可在 agent.yaml 的 prompts.fact_patch 覆盖）
FACT_PATCH_PROMPT = (
    "判断本轮对话需要如何更新当前会话的结构化事实。\n"
    "当前事实：{current}\n"
    "用户：{user}\n助手：{assistant}\n\n"
    "规则：\n"
    "1. 用户修改/删除条件时，直接输出新值或 null，不要保留旧值\n"
    "2. 用户新增条件时，只输出新增字段\n"
    "3. 用户说清空/重置约束时，action 填 clear\n"
    "4. 没有约束变化时，action 填 none\n"
    "5. 不要编造用户没提过的字段\n\n"
    '只输出 JSON：{{"action": "update|clear|none", "patch": {{"count": 3, '
    '"filters": {{"color": null}}}}}}'
)


@dataclass
class AgentResult:
    reply: str
    results: list = field(default_factory=list)
    tool: str = ""
    confidence: float = 0.0
    trace_id: str = ""
    retries: int = 0


class FindgoAgent:
    def __init__(
        self,
        config: AgentConfig | None = None,
        client=None,
        pipeline=None,
        tools: dict[str, BaseTool] | None = None,
    ):
        self.config = config or AgentConfig()
        self.client = client or ChatClient(self.config.llm)
        self.planner = Planner(self.client, self.config.prompts, self.config.max_iterations)
        self.critic = Critic(self.client, self.config.similarity_threshold, self.config.max_retries)
        self.responder = Responder(self.client, self.config.prompts)

        persist = self.config.resolve_db_path() if self.config.memory_persist else None
        self.memory = Memory(persist_path=persist, resume=True)
        self.executor = ToolExecutor()
        # 上一轮返回过的图片，用于「再发一次」
        self._last_paths: set[str] = set()
        self._last_images: list[dict] = []
        # 本会话累计出现过的图片，连续「再找」时全部排除
        self._seen_paths: set[str] = set()

        if tools is not None:
            self.tools = tools
        else:
            from ..utils.sandbox import Sandbox

            # 所有工具共享同一个 Pipeline，避免各自连接导致 Qdrant 文件锁冲突
            if pipeline is None:
                from ..retriever import build_pipeline

                pipeline = build_pipeline()

            guard = Sandbox(
                self.config.image_roots or None,
                self.config.workspace_roots or None,
            )
            self.sandbox = guard
            self.tools = {
                "image_retrieval": ImageRetrievalTool(pipeline, guard=guard),
                "image_search_by_image": ImageSearchByImageTool(pipeline, guard=guard),
                "index_folder": IndexFolderTool(pipeline, guard=guard),
                "get_index_stats": GetIndexStatsTool(pipeline),
                "switch_vector_backend": SwitchVectorBackendTool(pipeline),
                "list_indexed_folders": ListIndexedFoldersTool(pipeline),
                "count_indexed_images": CountIndexedImagesTool(pipeline, guard=guard),
                "filter_results": FilterResultsTool(pipeline),
                "preview_staged": PreviewStagedTool(),
                "describe_image": DescribeImageTool(pipeline, guard=guard),
                "export_results": ExportResultsTool(pipeline, guard=guard),
                "find_duplicates": FindDuplicatesTool(pipeline, guard=guard),
                "remove_index": RemoveIndexTool(pipeline, guard=guard),
                "trash_manage": TrashManageTool(pipeline, guard=guard),
                "web_image_search": WebImageSearchTool(pipeline),
                "folder_scan": FolderScanTool(guard=guard),
            }

            # 自定义全局工具加载在内置之后，同名不覆盖内置
            for name, tool in load_custom_tools(guard).items():
                self.tools.setdefault(name, tool)

        self.trace = TraceLogger()

    # ── 对外 ──────────────────────────────────────────────────────────────

    def run(self, user_input: str) -> AgentResult:
        user_input = (user_input or "").strip()
        if not user_input:
            return AgentResult(reply="请告诉我你想找什么图片。")

        self.memory.add("user", user_input)
        trace_id = uuid.uuid4().hex[:8]

        tools_desc = "\n".join(t.signature() for t in self.tools.values())

        facts = self.memory.get_facts()
        import json as _json

        extra = _json.dumps(facts, ensure_ascii=False) if facts else ""
        action = self.planner.plan(
            user_input, tools_desc, history=self.memory.recent(n=20), extra_context=extra
        )
        self.trace.info(
            "planner", "decide", action_type=action.action_type, thought=action.thought[:80]
        )

        if action.action_type in ("answer", "ask_user"):
            reply = action.reply or "请补充一下你想找的图片内容。"
            self.memory.add("assistant", reply)
            return AgentResult(reply=reply, trace_id=trace_id)

        if action.action_type != "tool_call":
            reply = "抱歉，我没能理解这个请求。"
            self.memory.add("assistant", reply)
            return AgentResult(reply=reply, trace_id=trace_id)

        tool_name = action.tool_name or "image_retrieval"
        tool = self.tools.get(tool_name)
        if tool is None:
            reply = f"没有找到可用的工具：{tool_name}"
            self.memory.add("assistant", reply)
            return AgentResult(reply=reply, trace_id=trace_id)

        args = action.tool_args or {}
        if not args.get("query"):
            args["query"] = user_input
        args.setdefault("top_k", self.config.top_k)

        result, retries = self._execute_with_retry(tool, args)
        reply = self.responder.compose(user_input, result)
        self.memory.add("assistant", reply)
        self.update_facts(user_input, reply)

        return AgentResult(
            reply=reply,
            results=result.get("images", []),
            tool=tool_name,
            confidence=float(result.get("confidence", 0.0)),
            trace_id=trace_id,
            retries=retries,
        )

    # ── 任务规划与分步执行 ────────────────────────────────────────────────

    def plan_tasks(self, user_input: str) -> list[TaskSpec]:
        """让 Planner 拆解任务，供界面展示与人工干预。"""
        tools_desc = "\n".join(t.signature() for t in self.tools.values())
        return self.planner.plan_tasks(
            user_input,
            tools_desc,
            history=self.memory.recent(n=20),
            extra_context=self._memory_context(),
        )

    def reset_recall_memory(self) -> None:
        """切会话或用户要求从头找时清空累计排除。"""
        self._last_paths = set()
        self._last_images = []
        self._seen_paths = set()
        logger.info("已清空本轮会话的排除记录")

    def _memory_context(self) -> str:
        """把结构化事实与最近日记拼成 Planner 的背景信息。"""
        parts: list[str] = []

        try:
            facts = self.memory.get_facts()
            if facts:
                import json

                parts.append("【当前会话事实】\n" + json.dumps(facts, ensure_ascii=False, indent=2))
        except Exception:
            pass

        try:
            from .diary import recent_diary_text

            diary = recent_diary_text(days=3)
            if diary:
                parts.append("【最近几天的日记（用户写过的，可在需要时引用）】\n" + diary)
        except Exception:
            pass

        return "\n\n".join(parts)

    def run_tasks(
        self,
        tasks: list[TaskSpec],
        write_memory: bool = True,
        should_stop=None,
        max_steps: int = 10,
        continue_on_failure: bool = True,
        auto_update_facts: bool = False,
        access_level: str = "auto",
    ) -> list[AgentResult]:
        """按任务列表逐条执行，跳过 skipped。

        write_memory=False 时由调用方统一写入对话记忆，避免一轮多任务产生重复消息。
        """
        results: list[AgentResult] = []
        previous_paths = set(self._last_paths)
        previous_images = list(self._last_images)
        # 累计排除：不只避上一轮，把本会话出现过的都扣掉
        excluded_paths = set(self._seen_paths) | previous_paths
        this_round_paths: set[str] = set()
        steps = 0
        for task in tasks:
            if task.status == "skipped":
                continue
            if steps >= max_steps:
                logger.warning("达到最大执行步数 %d，终止剩余任务", max_steps)
                break
            if should_stop is not None and should_stop():
                logger.info("收到停止信号，剩余任务跳过")
                break

            if task.reuse_last:
                if previous_images:
                    results.append(
                        AgentResult(
                            reply=f"这是上一次找的 {len(previous_images)} 张。",
                            results=list(previous_images),
                            tool="image_retrieval",
                            confidence=1.0,
                            trace_id=uuid.uuid4().hex[:8],
                        )
                    )
                    logger.info("复用上一轮结果 %d 张", len(previous_images))
                    continue
                # 没有可复用的结果，退回正常检索
                logger.info("用户要求重发但上一轮没有结果，改为重新检索")

            if task.is_dynamic:
                name = str((task.dynamic_spec or {}).get("name", ""))
                tool = self.tools.get(name)
                if tool is None:
                    results.append(
                        AgentResult(
                            reply=f"动态工具「{name}」尚未批准，已跳过",
                            results=[],
                            tool="",
                            confidence=0.0,
                            trace_id=uuid.uuid4().hex[:8],
                        )
                    )
                    continue
            elif task.action == "scan":
                tool = self.tools.get("folder_scan")
                if tool is None or not task.path:
                    continue
                args = {"path": task.path}
            else:
                if not task.query.strip():
                    continue
                tool = self.tools.get("image_retrieval")
                if tool is None:
                    break
                if task.reuse_last:
                    if previous_images:
                        results.append(
                            AgentResult(
                                reply=f"这是上一次找的 {len(previous_images)} 张。",
                                results=list(previous_images),
                                tool="image_retrieval",
                                confidence=1.0,
                                trace_id=uuid.uuid4().hex[:8],
                            )
                        )
                        logger.info("复用上一轮结果 %d 张", len(previous_images))
                        continue
                    logger.info("用户要求重发但上一轮没有结果，改为重新检索")

                args = {
                    "query": task.query,
                    "top_k": task.top_k if task.top_k > 0 else self.config.top_k,
                    "folder_path": task.path or "",
                    "all_results": bool(task.all_results),
                    "exclude_paths": list(excluded_paths),
                    "constraints": dict(task.constraints or {}),
                }

            result, retries = self._execute_with_retry(
                tool, args, should_stop=should_stop, access_level=access_level
            )

            if result.get("error") and not continue_on_failure:
                logger.warning(
                    "任务失败且 continue_on_failure=False，终止后续任务：%s",
                    result.get("summary", ""),
                )
                break

            if task.reuse_last:
                if previous_images:
                    results.append(
                        AgentResult(
                            reply=f"这是上一次找的 {len(previous_images)} 张。",
                            results=list(previous_images),
                            tool="image_retrieval",
                            confidence=1.0,
                            trace_id=uuid.uuid4().hex[:8],
                        )
                    )
                    logger.info("复用上一轮结果 %d 张", len(previous_images))
                    continue
                # 没有可复用的结果，退回正常检索
                logger.info("用户要求重发但上一轮没有结果，改为重新检索")

            if task.is_dynamic:
                name = str((task.dynamic_spec or {}).get("name", ""))
                tool = self.tools.get(name)
                if tool is None:
                    results.append(
                        AgentResult(
                            reply=f"动态工具「{name}」尚未批准，已跳过",
                            results=[],
                            tool="",
                            confidence=0.0,
                            trace_id=uuid.uuid4().hex[:8],
                        )
                    )
                    continue
            elif task.action == "scan":
                reply = str(result.get("summary") or "")
                images = []
            else:
                reply = self.responder.compose(task.query, result)
                images = result.get("images", [])

            if write_memory:
                self.memory.add("user", task.query or task.description)
                self.memory.add("assistant", reply)

            # 排除后不够数时补一句，免得用户以为是没找到
            want = task.top_k if task.top_k > 0 else 0
            if want and len(images) < want and excluded_paths:
                reply = (
                    f"{reply}\n\n（本会话已经给过 {len(excluded_paths)} 张，库里符合条件的"
                    f"只剩下 {len(images)} 张。说「从头找」可以重新搜全库。）"
                )

            for item in images:
                path = item.get("path", "") if isinstance(item, dict) else getattr(item, "path", "")
                if path:
                    this_round_paths.add(str(path).lower())

            steps += 1
            results.append(
                AgentResult(
                    reply=reply,
                    results=images,
                    tool=tool.name,
                    confidence=float(result.get("confidence", 0.0)),
                    trace_id=uuid.uuid4().hex[:8],
                    retries=retries,
                )
            )

            # 仅交互式对话开启：批量任务默认关闭，避免多条独立任务互相污染事实
            if auto_update_facts:
                self.update_facts(task.query or task.description, reply)

        self._last_paths = this_round_paths
        self._seen_paths |= this_round_paths
        if this_round_paths:
            self._last_images = [
                item for result in results for item in result.results if isinstance(item, dict)
            ]
        return results

    # ── 结构化事实 ────────────────────────────────────────────────────

    def update_facts(self, user_text: str, assistant_text: str) -> None:
        """让 LLM 输出事实 patch，并应用回当前会话。"""
        import json as _json

        current = self.memory.get_facts()
        prompt = FACT_PATCH_PROMPT.format(
            current=_json.dumps(current, ensure_ascii=False),
            user=(user_text or "")[:2000],
            assistant=(assistant_text or "")[:2000],
        )

        try:
            raw = self.client.chat(
                [{"role": "user", "content": prompt}],
                json_mode=True,
                max_tokens=400,
            )
            data = _json.loads(raw.strip())
        except Exception as e:
            logger.debug("事实 patch 抽取失败: %s", e)
            return

        if not isinstance(data, dict):
            return

        action = str(data.get("action", "none")).lower()
        if action == "clear":
            self.memory.clear_facts()
        elif action == "update" and isinstance(data.get("patch"), dict):
            self.memory.patch_facts(data["patch"])

    # ── 动态工具 ──────────────────────────────────────────────────────────

    def register_dynamic_tool(self, spec: dict):
        """编译并注册一个经用户批准的动态工具。"""
        from .dynamic import DynamicToolSpec, build_dynamic_tool

        parsed = DynamicToolSpec(**spec)
        tool = build_dynamic_tool(parsed, getattr(self, "sandbox", None))
        self.tools[parsed.name] = tool
        logger.info("动态工具已注册: %s", parsed.name)
        return tool

    # ── 运行时切换模型 ────────────────────────────────────────────────────

    def set_llm(self, llm_cfg: dict) -> None:
        """热切换 LLM（主界面模型选择器调用）。"""
        self.client = ChatClient(llm_cfg)
        self.planner.client = self.client
        self.critic.client = self.client
        self.responder.client = self.client

    # ── 内部 ──────────────────────────────────────────────────────────────

    def _execute_with_retry(
        self,
        tool: BaseTool,
        args: dict[str, Any],
        should_stop=None,
        access_level: str = "auto",
    ) -> tuple[dict[str, Any], int]:
        query = str(args.get("query", ""))
        attempt = 0
        result: dict[str, Any] = {}

        while True:
            result, _ = self.executor.execute(
                tool, args, should_stop=should_stop, access_level=access_level
            )
            if result.get("error") in (
                "invalid_args",
                "timeout",
                "need_confirm",
                "readonly_blocked",
            ):
                self.trace.error(
                    "tool", result["error"], tool=tool.name, error=result.get("summary", "")
                )
                return result, 0
            if result.get("error") == "run_failed":
                result = {
                    "images": [],
                    "confidence": 0.0,
                    "summary": result.get("summary", "工具执行失败"),
                }

            critique = self.critic.check(result)
            self.trace.info(
                "critic",
                "check",
                passed=critique.passed,
                confidence=round(critique.confidence, 3),
                reason=critique.reason[:80],
            )
            self.memory.add_trace(
                f"{self.memory.session_id}-{attempt}",
                "tool_call" if attempt == 0 else "critic_rewrite",
                tool.name,
                args,
                {"images": len(result.get("images", [])), "confidence": critique.confidence},
            )

            if critique.passed or attempt >= self.critic.max_retries:
                return result, attempt

            attempt += 1
            new_query = self.critic.rewrite(query, critique.reason)
            self.trace.info(
                "critic",
                "rewrite",
                old_query=query[:60],
                new_query=new_query[:60],
                confidence=round(critique.confidence, 3),
                reason=critique.reason[:120],
            )
            if new_query == query:
                return result, attempt
            query = new_query
            args = dict(args)
            args["query"] = new_query

    def close(self) -> None:
        self.memory.close()
