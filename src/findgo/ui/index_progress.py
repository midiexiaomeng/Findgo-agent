"""建库进度条：显示当前目录、进度、速度，支持取消。"""

from __future__ import annotations

import time

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class IndexProgressPanel(QWidget):
    """建库时浮在输入区上方的进度条。"""

    cancel_requested = pyqtSignal()
    pause_toggled = pyqtSignal(bool)  # True = 请求暂停

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVisible(False)
        self.setStyleSheet("background:transparent;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(4)

        top = QHBoxLayout()
        top.setSpacing(8)

        self.title = QLabel("正在建索引")
        self.title.setStyleSheet("color:#4A4A4A; font-size:12px; font-weight:600;")

        self.detail = QLabel("")
        self.detail.setStyleSheet("color:#8A8A8A; font-size:11.5px;")

        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setFixedSize(52, 24)
        self.pause_btn.setCursor(Qt.PointingHandCursor)
        self.pause_btn.setStyleSheet(
            "QPushButton{border:1px solid #E5E3DD;border-radius:6px;background:#FFFFFF;"
            "color:#4A4A4A;font-size:11.5px;}"
            "QPushButton:hover{background:#F2F0EA;}"
        )
        self.pause_btn.clicked.connect(self._on_pause)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setFixedSize(52, 24)
        self.cancel_btn.setCursor(Qt.PointingHandCursor)
        self.cancel_btn.setStyleSheet(
            "QPushButton{border:1px solid #E5E3DD;border-radius:6px;background:#FFFFFF;"
            "color:#6B6B6B;font-size:11.5px;}"
            "QPushButton:hover{background:#F6DED9;color:#C97B6E;border-color:#E5C4C2;}"
        )
        self.cancel_btn.clicked.connect(self._on_cancel)

        top.addWidget(self.title)
        top.addWidget(self.detail, 1)
        top.addWidget(self.pause_btn)
        top.addWidget(self.cancel_btn)

        self.bar = QProgressBar()
        self.bar.setFixedHeight(6)
        self.bar.setTextVisible(False)
        self.bar.setStyleSheet(
            "QProgressBar{border:none;border-radius:3px;background:#EDEBE6;}"
            "QProgressBar::chunk{background:#6F9C86;border-radius:3px;}"
        )

        layout.addLayout(top)
        layout.addWidget(self.bar)

        self._started = 0.0
        self._last_done = 0
        self._last_time = 0.0
        self._speed = 0.0
        self._paused = False

    # ── 状态 ──

    def start(self, text: str = "正在建索引") -> None:
        self._started = time.time()
        self._last_done = 0
        self._last_time = self._started
        self._speed = 0.0
        self._paused = False
        self.pause_btn.setText("暂停")
        self.pause_btn.setEnabled(True)
        self.title.setText(text)
        self.detail.setText("准备中…")
        self.bar.setRange(0, 0)  # 未知总量时用滚动条
        self.cancel_btn.setEnabled(True)
        self.setVisible(True)

    def update_progress(self, done: int, total: int, name: str = "") -> None:
        if total <= 0:
            return

        self.bar.setRange(0, total)
        self.bar.setValue(min(done, total))

        now = time.time()
        elapsed = now - self._started

        # 速度用最近一段的平均值：第一次更新时时间差接近 0，
        # 直接算会得到几万张/秒的假数据，所以跳过首次
        span = now - self._last_time
        recent = done - self._last_done
        if self._last_done > 0 and span >= 0.5 and recent > 0:
            speed = recent / span
            self._speed = speed
        else:
            speed = getattr(self, "_speed", 0.0)

        self._last_done = done
        self._last_time = now

        parts = [f"{done}/{total}"]
        if speed >= 0.1:
            parts.append(f"{speed:.1f} 张/秒")
            remain = (total - done) / speed
            if remain > 1:
                parts.append(f"剩余 {self._fmt(remain)}")
        elif elapsed > 1:
            parts.append(f"已用 {self._fmt(elapsed)}")

        if name:
            parts.append(name[:24])

        self.detail.setText(" · ".join(parts))

    def finish(self, summary: str = "") -> None:
        self.bar.setRange(0, 1)
        self.bar.setValue(1)
        self.title.setText("索引完成")
        self.detail.setText(summary[:80] if summary else "")
        self.cancel_btn.setEnabled(False)

        # 完成状态留一会儿再收起
        from PyQt5.QtCore import QTimer

        QTimer.singleShot(2600, self.hide)

    def fail(self, message: str) -> None:
        self.bar.setRange(0, 1)
        self.bar.setValue(0)
        self.title.setText("索引失败")
        self.detail.setText(str(message)[:80])
        self.cancel_btn.setEnabled(False)

        from PyQt5.QtCore import QTimer

        QTimer.singleShot(5000, self.hide)

    def _on_cancel(self) -> None:
        self.cancel_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.detail.setText("正在停止…")
        self.cancel_requested.emit()

    def _on_pause(self) -> None:
        self._paused = not self._paused
        if self._paused:
            self.pause_btn.setText("继续")
            self.detail.setText("已暂停，点「继续」恢复")
            self.bar.setRange(0, 0)
        else:
            self.pause_btn.setText("暂停")
            self._last_time = time.time()
            self._last_done = 0
        self.pause_toggled.emit(self._paused)

    @property
    def paused(self) -> bool:
        return self._paused

    @staticmethod
    def _fmt(seconds: float) -> str:
        if seconds < 60:
            return f"{int(seconds)} 秒"
        minutes = int(seconds // 60)
        if minutes < 60:
            return f"{minutes} 分"
        return f"{minutes // 60} 小时 {minutes % 60} 分"


__all__ = ["IndexProgressPanel"]
