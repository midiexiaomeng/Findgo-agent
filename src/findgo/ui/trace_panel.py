"""执行轨迹回放：按时间顺序展示本会话的工具调用、Critic 判定与改写。"""

from __future__ import annotations

import json

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)


class TraceDialog(QDialog):
    """左侧时间线，右侧详情。只读，不做重新执行。"""

    def __init__(self, traces: list[dict], parent=None):
        super().__init__(parent)
        self.traces = traces
        self.setWindowTitle("执行轨迹")
        self.resize(820, 540)
        self.setStyleSheet(
            "QDialog{background:#FBFAF7;}"
            "QListWidget{background:#FFFFFF;border:1px solid #EDEBE6;border-radius:8px;"
            "font-size:12px;padding:4px;}"
            "QListWidget::item{padding:6px 8px;border-radius:6px;}"
            "QListWidget::item:selected{background:#E3EFE7;}"
            "QTextEdit{background:#FFFFFF;border:1px solid #EDEBE6;border-radius:8px;"
            "font-family:Consolas,Monaco,monospace;font-size:11.5px;color:#2E2E2E;}"
        )
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        top = QHBoxLayout()
        title = QLabel(f"共 {len(self.traces)} 条记录")
        title.setStyleSheet("color:#4A4A4A;font-size:12px;font-weight:600;")
        top.addWidget(title)
        top.addStretch(1)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        top.addWidget(close_btn)
        layout.addLayout(top)

        body = QHBoxLayout()
        body.setSpacing(8)

        self.timeline = QListWidget()
        self.timeline.setFixedWidth(280)
        self.timeline.currentRowChanged.connect(self._on_select)
        body.addWidget(self.timeline)

        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        body.addWidget(self.detail, 1)

        layout.addLayout(body, 1)

        self._fill()

    def _fill(self) -> None:
        if not self.traces:
            self.timeline.addItem(QListWidgetItem("（还没有执行记录）"))
            self.detail.setPlainText("聊几句，这里会记录每一步工具调用与判定。")
            return

        for tr in self.traces:
            stamp = str(tr.get("created_at", ""))[-8:]
            action = str(tr.get("action", ""))
            tool = str(tr.get("tool_name") or "")

            mark = {
                "tool_call": "▶",
                "critic_rewrite": "↻",
                "tool": "▶",
            }.get(action, "·")

            text = f"{mark} {stamp} {action}"
            if tool:
                text += f" · {tool}"
            item = QListWidgetItem(text)
            item.setToolTip(text)
            self.timeline.addItem(item)

        self.timeline.setCurrentRow(0)

    def _on_select(self, row: int) -> None:
        if row < 0 or row >= len(self.traces):
            return
        tr = self.traces[row]

        lines = [
            f"时间    {tr.get('created_at', '')}",
            f"动作    {tr.get('action', '')}",
            f"工具    {tr.get('tool_name') or '（无）'}",
            f"trace   {tr.get('trace_id', '')}",
            "",
            "── 参数 ──",
            self._pretty(tr.get("tool_args")),
            "",
            "── 结果 ──",
            self._pretty(tr.get("tool_result")),
        ]
        self.detail.setPlainText("\n".join(lines))

    @staticmethod
    def _pretty(raw) -> str:
        if not raw:
            return "（空）"
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception:
            return str(raw)


__all__ = ["TraceDialog"]
