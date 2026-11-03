"""一轮对话的应用服务层。

把「用户说一句话之后该做什么」从 Qt 里搬出来：判断命令与探索型任务、
注入默认目录、执行任务链、联网补足、写记忆、打包结果。

界面只负责收集输入和显示输出，不再持有业务分支；这里不 import PyQt，
所以可以直接在测试里跑完整的一轮。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from ..utils.schemas import TaskSpec
from .orchestrator import AgentResult
from .planner import detect_reset, needs_exploration

logger = logging.getLogger("findgo.turn")

#: 观察回调：payload 是任务列表或 ReAct 单步，由界面决定怎么画
ProgressCallback = Callable[[Any], None]


@dataclass
class TurnOutcome:
    """一轮执行的结果。字段与界面需要的四元组对应。"""

    mode: str  # command | react | tasks | image
    tasks: list[TaskSpec] = field(default_factory=list)
    results: list[AgentResult] = field(default_factory=list)
    reply: str = ""
    images: list[dict] = field(default_factory=list)

    def as_payload(self) -> tuple:
        """给界面用的 (tasks, results, reply, images)。"""
        return self.tasks, self.results, self.reply, self.images


class TurnService:
    def __init__(
        self,
        agent,
        on_log: Callable[[str], None] | None = None,
        on_progress: ProgressCallback | None = None,
        on_step: ProgressCallback | None = None,
        on_phase: Callable[[str], None] | None = None,
    ):
        self.agent = agent
        self._on_log = on_log
        self._on_progress = on_progress
        self._on_step = on_step
        self._on_phase = on_phase

    # ── 对外 ──────────────────────────────────────────────────────────────

    def run(
        self,
        text: str,
        *,
        default_scope: str = "",
        search_mode: str = "local",
        access_level: str = "auto",
        should_stop: Callable[[], bool] | None = None,
    ) -> TurnOutcome:
        """正常一轮：探索型走 ReAct，其余走任务链。"""
        text = (text or "").strip()
        self.agent.memory.add("user", text)

        if detect_reset(text):
            self.agent.reset_recall_memory()
            self.log(">>> 已清空排除记录，本次从全库重新检索")

        if self._needs_exploration(text):
            self.log(">>> 检测到探索型任务，使用 ReAct 循环")
            self.phase("react")
            result = self.agent.run_react(
                text,
                should_stop=should_stop,
                access_level=access_level,
                on_step=self._step,
            )
            return TurnOutcome(mode="react", reply=result.reply, images=result.results)

        tasks = self.agent.plan_tasks(text)
        if default_scope:
            for task in tasks:
                if task.action == "retrieve" and not task.path:
                    task.path = default_scope
        self.progress(tasks)
        self.agent.memory.save_tasks(list(tasks))

        if any(task.is_dynamic for task in tasks):
            return TurnOutcome(
                mode="tasks",
                tasks=tasks,
                reply="检测到自定义临时工具，请审核后点「重跑」执行。",
            )

        return self._execute(
            tasks, search_mode=search_mode, access_level=access_level, should_stop=should_stop
        )

    def run_existing(
        self,
        tasks: list[TaskSpec],
        *,
        search_mode: str = "local",
        access_level: str = "auto",
        should_stop: Callable[[], bool] | None = None,
    ) -> TurnOutcome:
        """重跑界面上已有的任务链，不重新规划。"""
        return self._execute(
            tasks, search_mode=search_mode, access_level=access_level, should_stop=should_stop
        )

    def run_command(self, text: str) -> TurnOutcome:
        """斜杠命令：直接执行，不走规划。"""
        from .commands import CommandError, dispatch

        self.agent.memory.add("user", text)
        try:
            reply = dispatch(text, self.agent)
        except CommandError as e:
            reply = str(e)
        except Exception as e:  # 命令实现里的意外错误也要变成一句话
            reply = f"命令执行失败：{e}"

        reply = reply or "命令没有返回内容。"
        self.agent.memory.add("assistant", reply)
        return TurnOutcome(mode="command", reply=reply)

    def run_image(
        self,
        image_path: str,
        text: str = "",
        *,
        top_k: int = 12,
    ) -> TurnOutcome:
        """以图搜图：附件图片作为查询，不再经过 Planner。"""
        query = text or "找相似的图片"
        self.agent.memory.add("user", query)

        tool = self.agent.tools.get("image_search_by_image")
        if tool is None:
            return TurnOutcome(mode="image", reply="没有注册以图搜图工具")

        result, _ = self.agent.executor.execute(tool, {"image_path": image_path, "top_k": top_k})
        reply = self.agent.responder.compose(query, result)
        images = result.get("images", [])
        self.agent.memory.add("assistant", reply, images=self._paths(images))
        return TurnOutcome(mode="image", reply=reply, images=images)

    # ── 内部 ──────────────────────────────────────────────────────────────

    def _execute(self, tasks, *, search_mode, access_level, should_stop) -> TurnOutcome:
        results = self.agent.run_tasks(
            tasks,
            write_memory=False,
            should_stop=should_stop,
            access_level=access_level,
        )
        self.web_fill(tasks, results, search_mode=search_mode, should_stop=should_stop)
        return self.pack(tasks, results)

    def web_fill(self, tasks, results, *, search_mode: str, should_stop=None) -> None:
        """本地结果不够时联网补一批。只补不替，失败也不影响本地结果。"""
        if search_mode not in ("hybrid", "online") or self._stopped(should_stop):
            return

        found = sum(len(r.results) for r in results)
        need = sum(t.top_k for t in tasks if t.top_k > 0)
        short = need - found
        if search_mode != "online" and not (need > 0 and short > 0):
            return

        tool = self.agent.tools.get("web_image_search")
        if tool is None:
            return

        query = tasks[0].query if tasks and tasks[0].query else ""
        count = max(short, 10) if search_mode == "hybrid" else max(need, 20)
        web = tool.run(query=query, top_k=min(count, 100))
        self.log(f">>> 联网补充：{web.get('summary', '')}")

    def pack(self, tasks, results) -> TurnOutcome:
        """把多条任务结果合成一段回复，并按界面需要的形状返回。"""
        replies = [r.reply for r in results if r.reply]
        combined = "\n".join(replies) or "没有找到相关图片。"
        images = [img for result in results for img in result.results]
        self.agent.memory.add("assistant", combined, images=self._paths(images))
        return TurnOutcome(
            mode="tasks",
            tasks=tasks,
            results=results,
            reply=combined,
            images=images,
        )

    @staticmethod
    def _needs_exploration(text: str) -> bool:
        try:
            return needs_exploration(text)
        except Exception:
            return False

    @staticmethod
    def _paths(images) -> list[str]:
        paths = [
            (img.get("path", "") if isinstance(img, dict) else getattr(img, "path", ""))
            for img in images
        ]
        return [p for p in paths if p]

    @staticmethod
    def _stopped(should_stop) -> bool:
        try:
            return bool(should_stop and should_stop())
        except Exception:
            return False

    def log(self, text: str) -> None:
        if self._on_log is not None:
            try:
                self._on_log(text)
            except Exception:
                pass
        else:
            logger.info(text)

    def progress(self, payload) -> None:
        if self._on_progress is not None:
            try:
                self._on_progress(payload)
            except Exception:
                pass

    def _step(self, payload) -> None:
        if self._on_step is not None:
            try:
                self._on_step(payload)
            except Exception:
                pass

    def phase(self, name: str) -> None:
        """阶段变化（react / tasks），给界面切顶部状态文案。"""
        if self._on_phase is not None:
            try:
                self._on_phase(name)
            except Exception:
                pass
