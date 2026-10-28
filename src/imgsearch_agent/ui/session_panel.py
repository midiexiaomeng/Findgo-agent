"""会话管理：切换、新建、重命名、删除、历史恢复。"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QInputDialog, QListWidgetItem, QMenu


class SessionMixin:
    """依赖宿主提供：agent / session_list / task_list / state_label / conversation / last_results。"""

    def _session_menu(self, pos) -> None:
        item = self.session_list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        rename = menu.addAction("重命名")
        rename.triggered.connect(lambda: self.on_rename_session(item))
        delete = menu.addAction("删除此会话")
        delete.triggered.connect(lambda: self._delete_session(item))
        menu.exec_(self.session_list.mapToGlobal(pos))

    def _delete_session(self, item: QListWidgetItem) -> None:
        sid = item.data(Qt.UserRole)
        if not sid or self.agent is None:
            return
        self.agent.memory.delete_session(sid)
        if sid == self.agent.memory.session_id:
            self.on_new_chat()
        else:
            self._refresh_sessions()
        self.append_log(">>> 已删除会话 " + sid)

    def _clear_chat_view(self) -> None:
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # 立刻脱离父级，deleteLater 是延迟销毁，不先解绑会残留
                widget.setParent(None)
                widget.deleteLater()

    def _load_chat(self, session_id: str) -> None:
        self._clear_chat_view()
        self._galleries.clear()
        self.conversation = []
        if self.agent is None:
            return
        for message in self.agent.memory.load_messages(session_id):
            role = "你" if message.get("role") == "user" else "Agent"
            content = message.get("content", "")
            self.conversation.append((role, content))
            self._append_chat(role, content, message.get("images") or [])
        self._render_memory()

    def on_clear_chat(self) -> None:
        self._clear_chat_view()
        self.append_log(">>> 对话窗口已清屏（历史仍保存在会话中）")

    def _refresh_sessions(self) -> None:
        self.session_list.clear()
        if self.agent is None:
            return
        keep = max(1, int(self.agent.config.memory_max_sessions))
        for item_data in self.agent.memory.list_sessions(limit=keep):
            mark = (
                "● "
                if item_data["session_id"] == self.agent.memory.session_id
                else "○ "
            )
            line = QListWidgetItem(
                mark
                + item_data["title"][:16]
                + "  ("
                + str(item_data["count"])
                + ")"
            )
            line.setData(Qt.UserRole, item_data["session_id"])
            line.setToolTip(
                item_data["updated_at"] + "\n" + item_data["session_id"]
            )
            self.session_list.addItem(line)

    def on_new_chat(self) -> None:
        try:
            agent = self._get_agent()
            agent.memory.new_session()
            agent.memory.save_tasks([])
            agent.memory.prune_sessions(
                keep=max(1, int(agent.config.memory_max_sessions))
            )
            self.conversation = []
            self.last_results = []
            self._clear_chat_view()
            self.task_list.clear()
            self._render_memory()
            self._refresh_sessions()
            self.append_log(">>> 已开始新对话 " + agent.memory.session_id)
        except Exception as e:
            self.append_log("新建对话失败: " + str(e))

    def on_session_clicked(self, item: QListWidgetItem) -> None:
        sid = item.data(Qt.UserRole)
        if not sid:
            return
        try:
            agent = self._get_agent()
            agent.memory.switch_session(sid)
            self._load_chat(sid)
            self.task_list.clear()
            for task in agent.memory.load_tasks(sid):
                self._add_task_item(task)
            self._refresh_sessions()
            self.append_log(">>> 已切换历史会话 " + sid)
        except Exception as e:
            self.append_log("切换会话失败: " + str(e))

    def on_rename_session(self, item: QListWidgetItem) -> None:
        sid = item.data(Qt.UserRole)
        if not sid or self.agent is None:
            return
        current = self.agent.memory.get_title(sid)
        if not current:
            current = item.text().lstrip("●○ ").split("  (")[0]
        title, ok = QInputDialog.getText(
            self, "重命名会话", "会话名称：", text=current
        )
        if not ok:
            return
        self.agent.memory.rename_session(title.strip(), sid)
        self._refresh_sessions()
        self.append_log(">>> 会话已重命名为「" + title.strip() + "」")

    def _restore_history(self) -> None:
        try:
            agent = self._get_agent()
            self._refresh_sessions()
            self._load_chat(agent.memory.session_id)

            # 恢复任务链，但不自动执行；有未完成的任务时给一个「继续」入口
            tasks = agent.memory.load_tasks(agent.memory.session_id)
            self.task_list.clear()
            for task in tasks:
                self._add_task_item(task)

            pending = [t for t in tasks if t.status == "pending" and t.query]
            if pending and hasattr(self, "resume_btn"):
                self.resume_btn.setVisible(True)
                self.state_label.setText(
                    f"有 {len(pending)} 个未完成任务，点「继续」开始执行"
                )
                self.append_log(f">>> 发现 {len(pending)} 个未完成任务，等待确认")

            self.append_log(">>> 已恢复会话 " + agent.memory.session_id)
        except Exception as e:
            self.append_log("加载历史失败: " + str(e))
