"""无边框窗口的自定义标题栏。"""

from __future__ import annotations

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QToolButton, QWidget


class TitleBar(QWidget):
    """拖拽移动、双击最大化、左右栏开关、最小化/最大化/关闭。"""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self._drag_offset = None
        self.setFixedHeight(38)
        self.setStyleSheet("background:#FBFAF7;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 8, 0)
        layout.setSpacing(6)

        self.left_btn = self._make_toggle("◧", "隐藏/显示左栏（双击恢复默认宽度）")
        self.left_btn.clicked.connect(window.toggle_left)
        self.left_btn.installEventFilter(self)

        title = QLabel("findgo · 多模态图像检索")
        title.setStyleSheet("color:#2E2E2E; font-size:12.5px; font-weight:600;")
        title.setAlignment(Qt.AlignCenter)

        self.right_btn = self._make_toggle("◨", "隐藏/显示右栏（双击恢复默认宽度）")
        self.right_btn.clicked.connect(window.toggle_right)
        self.right_btn.installEventFilter(self)

        self.max_btn = self._make_icon("□", "最大化 / 还原")
        self.max_btn.clicked.connect(window.toggle_maximize)

        self.min_btn = self._make_icon("—", "最小化")
        self.min_btn.clicked.connect(window.showMinimized)

        self.close_btn = self._make_icon("×", "关闭")
        self.close_btn.setStyleSheet(
            "QToolButton{border:none;border-radius:6px;color:#4A4A4A;"
            "font-size:15px;padding:2px 6px;}"
            "QToolButton:hover{background:#F6DED9;color:#C97B6E;}"
        )
        self.close_btn.clicked.connect(window.close)

        layout.addWidget(self.left_btn)
        layout.addStretch(1)
        layout.addWidget(title)
        layout.addStretch(1)
        layout.addWidget(self.right_btn)
        layout.addSpacing(8)
        layout.addWidget(self.min_btn)
        layout.addWidget(self.max_btn)
        layout.addWidget(self.close_btn)

    @staticmethod
    def _make_toggle(text, tip):
        btn = QToolButton()
        btn.setText(text)
        btn.setToolTip(tip)
        btn.setFixedSize(26, 24)
        btn.setStyleSheet(
            "QToolButton{border:1px solid #EDEBE6;border-radius:7px;"
            "background:#FFFFFF;color:#4A4A4A;font-size:13px;}"
            "QToolButton:hover{background:#F2F0EA;}"
        )
        return btn

    @staticmethod
    def _make_icon(text, tip):
        btn = QToolButton()
        btn.setText(text)
        btn.setToolTip(tip)
        btn.setFixedSize(26, 24)
        btn.setStyleSheet(
            "QToolButton{border:none;border-radius:6px;color:#4A4A4A;"
            "font-size:14px;padding:2px 4px;}"
            "QToolButton:hover{background:#F2F0EA;}"
        )
        return btn

    # ── 拖拽移动 ──

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_offset = event.globalPos() - self.window.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_offset and event.buttons() & Qt.LeftButton:
            self.window.move(event.globalPos() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_offset = None
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        if self.window.isMaximized():
            self.window.showNormal()
        else:
            self.window.showMaximized()
        event.accept()

    def eventFilter(self, obj, event) -> bool:
        # 双击左右切换按钮 = 恢复该栏预设宽度
        if event.type() == QEvent.MouseButtonDblClick:
            if obj is self.left_btn:
                self.window.reset_left()
                return True
            if obj is self.right_btn:
                self.window.reset_right()
                return True
        return super().eventFilter(obj, event)
