"""Agent 执行：发送、后台线程、任务链重跑、停止、失败处理。"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt

from ..model_store import ModelStore
from .components import Worker
from .dynamic_dialog import DynamicToolDialog


class AgentRunnerMixin:
    """依赖宿主提供：input / send_btn / run_btn / stop_btn / task_list /
    chat_layout / worker / _cancel / agent / bridge / state_label 等属性。"""

    # ── 初始化 ────────────────────────────────────────────────────────────

    def _get_agent(self):
        if self.agent is None:
            from findgo.agent import FindgoAgent
            from findgo.config import AgentConfig

            self.agent = FindgoAgent(config=AgentConfig())
            entry = ModelStore().get_active()
            if entry:
                self.agent.set_llm(ModelStore.to_llm_config(entry))
                self.model_btn.setText(entry.get("label", entry.get("model", "")))
        return self.agent

    def _prepare_agent(self) -> None:
        """主线程创建 Agent 与 Pipeline，避免后台线程碰 SQLite / Qdrant 连接。"""
        try:
            agent = self._get_agent()
            tool = agent.tools.get("image_retrieval")
            if tool is not None:
                _ = tool.pipeline
        except Exception as e:
            self.append_log("初始化检索服务失败: " + str(e))

    def _warmup(self) -> None:
        """后台加载 CLIP，避免第一次检索时再等。"""
        import threading

        def run():
            try:
                agent = self._get_agent()
                tool = agent.tools.get("image_retrieval")
                if tool is None:
                    return
                tool.pipeline.encoder.load()
                self.bridge.message.emit(">>> CLIP 预热完成")
            except Exception as e:
                self.bridge.message.emit(f">>> CLIP 预热失败: {e}")

        threading.Thread(target=run, daemon=True).start()

    # ── 状态 ──────────────────────────────────────────────────────────────

    def _set_busy(self, busy: bool) -> None:
        for btn in (
            self.send_btn,
            self.run_btn,
            self.add_btn,
            self.del_btn,
            self.up_btn,
            self.down_btn,
        ):
            btn.setEnabled(not busy)
        # 设置与停止在任务执行期间仍然可用
        self.settings_btn.setEnabled(True)
        if hasattr(self, "stop_btn"):
            self.stop_btn.setEnabled(busy)
        self.input.setEnabled(not busy)

    # ── 发送 ──────────────────────────────────────────────────────────────

    def on_send(self) -> None:
        text = self.input.toPlainText().strip()
        has_attachment = bool(self._attachment)
        if (not text and not has_attachment) or self.worker is not None:
            return

        self._cancel = False
        if hasattr(self, "resume_btn"):
            self.resume_btn.setVisible(False)
        self.conversation.append(("你", text))
        self._append_chat("你", text)
        self._render_memory()
        self.input.clear()
        self._set_busy(True)
        self.state_label.setText("Agent 正在理解需求并检索…")
        self.append_log(">>> 用户：" + text.replace("\n", " ")[:60])

        if has_attachment:
            worker_fn = self._run_image_turn
            worker_args = (self._attachment, text)
            self._clear_attachment()
        else:
            worker_fn = self._run_turn
            worker_args = (text,)

        self.worker = Worker(worker_fn, *worker_args)
        self.worker.progress.connect(self.on_tasks_ready)
        self.worker.done.connect(self.on_turn_done)
        self.worker.failed.connect(self.on_worker_failed)
        self.worker.finished.connect(self._finish_worker)
        self.worker.start()

    # ── 一轮完整执行 ──────────────────────────────────────────────────────

    def _run_turn(self, text: str):
        agent = self._get_agent()
        agent.memory.add("user", text)

        default_scope = ""
        try:
            if self.tree_root and self.tree_root.resolve() != self._root().resolve():
                default_scope = str(self.tree_root)
        except Exception:
            default_scope = ""

        tasks = agent.plan_tasks(text)
        if default_scope:
            for task in tasks:
                if task.action == "retrieve" and not task.path:
                    task.path = default_scope

        try:
            worker = self.worker
            if worker is not None:
                worker.progress.emit(tasks)
        except Exception:
            pass
        agent.memory.save_tasks(list(tasks))

        pending_dynamic = [t for t in tasks if t.is_dynamic]
        if pending_dynamic:
            return (
                tasks,
                [],
                "检测到自定义临时工具，请审核后点「重跑」执行。",
                [],
            )

        results = agent.run_tasks(
            tasks, write_memory=False, should_stop=lambda: self._cancel
        )
        self._maybe_web_fill(agent, tasks, results)
        return self._pack_results(agent, tasks, results)

    def _run_existing_tasks(self, tasks):
        agent = self._get_agent()
        results = agent.run_tasks(
            tasks, write_memory=False, should_stop=lambda: self._cancel
        )
        self._maybe_web_fill(agent, tasks, results)
        return self._pack_results(agent, tasks, results)

    def _maybe_web_fill(self, agent, tasks, results) -> None:
        """本地不够时按所选模式联网补充。"""
        if self._search_mode not in ("hybrid", "online") or self._cancel:
            return

        found = sum(len(r.results) for r in results)
        need = sum(t.top_k for t in tasks if t.top_k > 0) or 0
        short = need - found

        if self._search_mode == "online" or (need > 0 and short > 0):
            tool = agent.tools.get("web_image_search")
            if tool is None:
                return
            query = tasks[0].query if tasks and tasks[0].query else ""
            count = (
                max(short, 10)
                if self._search_mode == "hybrid"
                else max(need, 20)
            )
            web = tool.run(query=query, top_k=min(count, 100))
            self.append_log(f">>> 联网补充：{web.get('summary', '')}")

    def _pack_results(self, agent, tasks, results):
        replies = [r.reply for r in results if r.reply]
        combined = "\n".join(replies) or "没有找到相关图片。"
        images = [img for result in results for img in result.results]
        image_paths = [
            (img.get("path", "") if isinstance(img, dict) else getattr(img, "path", ""))
            for img in images
        ]
        image_paths = [p for p in image_paths if p]
        agent.memory.add("assistant", combined, images=image_paths)
        return tasks, results, combined, images

    # ── 结束 ──────────────────────────────────────────────────────────────

    def on_turn_done(self, payload) -> None:
        tasks, results, combined, images = payload
        self.conversation.append(("Agent", combined))
        self._append_chat("Agent", combined, images)
        self._render_memory()

        self.task_list.clear()
        for task in tasks:
            self._add_task_item(task)

        self.last_results = images
        self.state_label.setText(
            f"完成 {len(results)} 条任务，共 {len(images)} 张图片"
        )
        self._refresh_sessions()

    def on_worker_failed(self, error: str) -> None:
        self.append_log("!!! 执行失败：" + error)
        self.state_label.setText("执行失败：" + error)
        self.conversation.append(("Agent", "出错了：" + error))
        self._append_chat("Agent", "出错了：" + error)
        self._render_memory()

    def _finish_worker(self) -> None:
        self.worker = None
        self._set_busy(False)
        self.input.setFocus()

    def on_stop(self) -> None:
        self._cancel = True
        self.state_label.setText("正在停止，当前这条检索完成后中断…")

    # ── 任务链重跑 ────────────────────────────────────────────────────────

    def _review_dynamic_tasks(self, tasks) -> bool:
        """逐个审核动态工具，全部允许才继续。"""
        agent = self._get_agent()
        for task in tasks:
            if not task.is_dynamic:
                continue
            spec = task.dynamic_spec or {}
            name = str(spec.get("name", ""))
            if not name:
                task.status = "skipped"
                continue
            if name in agent.tools:
                continue

            try:
                from findgo.agent.recommender import score_dynamic_tool

                recommendation = score_dynamic_tool(
                    spec, agent.memory.list_traces(name)
                )
            except Exception:
                recommendation = {}

            dialog = DynamicToolDialog(spec, self, recommendation)
            dialog.exec_()
            if not dialog.approved:
                task.status = "skipped"
                self.append_log(f">>> 已拒绝动态工具 {name}")
                continue

            try:
                agent.register_dynamic_tool(spec)
                self.append_log(f">>> 已允许动态工具 {name}")
                if dialog.save_to_global:
                    from findgo.agent.custom_tools import save_custom_tool

                    save_custom_tool(spec)
                    self.append_log(f">>> 已保存到全局工具库：{name}")
            except Exception as e:
                task.status = "skipped"
                self.append_log(f"!!! 动态工具加载失败：{e}")
        return True

    def on_run_tasks(self) -> None:
        if hasattr(self, "resume_btn"):
            self.resume_btn.setVisible(False)
        tasks = self._collect_tasks()
        self._review_dynamic_tasks(tasks)
        if not tasks or self.worker is not None:
            return
        if not any(t.status == "pending" for t in tasks):
            self.state_label.setText("没有勾选任何任务。")
            return

        self._set_busy(True)
        self.state_label.setText("正在按任务链重新执行…")
        self.worker = Worker(self._run_existing_tasks, tasks)
        self.worker.done.connect(self.on_turn_done)
        self.worker.failed.connect(self.on_worker_failed)
        self.worker.finished.connect(self._finish_worker)
        self.worker.start()
