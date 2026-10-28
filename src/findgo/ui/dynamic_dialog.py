"""动态工具审核弹窗。"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)


class DynamicToolDialog(QDialog):
    """先看源码和安全提示，再决定是否允许执行。"""

    def __init__(self, spec: dict, parent=None, recommendation: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("动态工具审核")
        self.resize(720, 520)
        self.approved = False
        self.save_to_global = False
        self.recommendation = recommendation or {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        title = QLabel(spec.get("name", "未命名工具"))
        title.setStyleSheet("font-weight:600; font-size:14px;")
        layout.addWidget(title)

        desc = QLabel(spec.get("description", ""))
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#4A4A4A; font-size:12px;")
        layout.addWidget(desc)

        note = QLabel(spec.get("safety_note") or "未提供安全说明")
        note.setWordWrap(True)
        note.setStyleSheet(
            "color:#C97B6E; background:#FDF4F2; border-radius:6px;"
            "padding:8px; font-size:12px;"
        )
        layout.addWidget(note)

        code = QPlainTextEdit()
        code.setReadOnly(True)
        code.setStyleSheet(
            "QPlainTextEdit{background:#F8F7F3;border:1px solid #EDEBE6;"
            "border-radius:8px;padding:8px;font-family:Consolas;font-size:12px;}"
        )
        code.setPlainText(
            spec.get("args_schema_code", "")
            + "\n\n"
            + spec.get("run_func_code", "")
        )
        layout.addWidget(code, 1)

        self.save_check = QCheckBox("保存到自定义全局工具库（custom_tools.yaml）")
        if self.recommendation.get("recommend"):
            self.save_check.setChecked(True)
        self.save_check.setToolTip(
            "；".join(self.recommendation.get("reasons", []))
            or "首次生成，默认不推荐保存"
        )
        layout.addWidget(self.save_check)

        row = QHBoxLayout()
        deny = QPushButton("拒绝")
        deny.clicked.connect(self.reject)
        allow = QPushButton("允许执行")
        allow.setStyleSheet(
            "QPushButton{background:#6F9C86;color:white;border:none;"
            "border-radius:6px;padding:7px 18px;}"
        )
        allow.clicked.connect(self._on_allow)
        row.addStretch(1)
        row.addWidget(deny)
        row.addWidget(allow)
        layout.addLayout(row)

    def _on_allow(self) -> None:
        self.approved = True
        self.save_to_global = self.save_check.isChecked()
        self.accept()
