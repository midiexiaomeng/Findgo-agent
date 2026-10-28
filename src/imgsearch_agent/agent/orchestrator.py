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
    ImageSearchByImageTool,
    ImageRetrievalTool,
    IndexFolderTool,
    ListIndexedFoldersTool,
    PreviewStagedTool,
    RemoveIndexTool,
    SwitchVectorBackendTool,
    TrashManageTool,
    WebImageSearchTool,
)
from ..tools.base_tool import BaseTool
from ..utils.logger import TraceLogger
from ..utils.exceptions import ValidationError
from ..utils.schemas import AgentAction, TaskSpec
from .critic import Critic
from .custom_tools import load_custom_tools
from .executor import ToolExecutor
from .memory import Memory
from .planner import Planner
from .responder import Responder

logger = logging.getLogger("imgsearch_agent.orchestrator")


@dataclass
class AgentResult:
    reply: str
    results: list = field(default_factory=list)
    tool: str = ""
    confidence: float = 0.0
    trace_id: str = ""
    retries: int = 0


class ImageSearchAgent:
    def __init__(
        self,
        config: AgentConfig | None = None,
        client=None,
        pipeline=None,
        tools: dict[str, BaseTool] | None = None,
    ):
        self.config = config or AgentConfig()
        self.client = client or ChatClient(self.config.llm)
        self.planner = Planner(
            self.client, self.config.prompts, self.config.max_iterations
        )
        self.critic = Critic(
            self.client, self.config.similarity_threshold, self.config.max_retries
        )
        self.responder = Responder(self.client, self.config.prompts)

        persist = self.config.resolve_db_path() if self.config.memory_persist else None
        self.memory = Memory(persist_path=persist, resume=True)
        self.executor = ToolExecutor()
        # 上一轮返回过的图片，下一轮默认排除，避免重复推荐
        self._last_paths: set[str] = set()
        self._last_images: list[dict] = []

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
                "image_search_by_image": ImageSearchByImageTool(
                    pipeline, guard=guard
                ),
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
        action = self.planner.plan(
            user_input, tools_desc, history=self.memory.recent(n=20)
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
            user_input, tools_desc, history=self.memory.recent(n=20)
        )

    def run_tasks(
        self,
        tasks: list[TaskSpec],
        write_memory: bool = True,
        should_stop=None,
        max_steps: int = 10,
        continue_on_failure: bool = True,
    ) -> list[AgentResult]:
        """按任务列表逐条执行，跳过 skipped。

        write_memory=False 时由调用方统一写入对话记忆，避免一轮多任务产生重复消息。
        """
        results: list[AgentResult] = []
        previous_paths = set(self._last_paths)
        previous_images = list(self._last_images)
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
                    "exclude_paths": list(previous_paths),
                }

            result, retries = self._execute_with_retry(
                tool, args, should_stop=should_stop
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

        self._last_paths = this_round_paths
        if this_round_paths:
            self._last_images = [
                item
                for result in results
                for item in result.results
                if isinstance(item, dict)
            ]
        return results

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
        self, tool: BaseTool, args: dict[str, Any], should_stop=None
    ) -> tuple[dict[str, Any], int]:
        query = str(args.get("query", ""))
        attempt = 0
        result: dict[str, Any] = {}

        while True:
            result, _ = self.executor.execute(
                tool, args, should_stop=should_stop
            )
            if result.get("error") in ("invalid_args", "timeout"):
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
