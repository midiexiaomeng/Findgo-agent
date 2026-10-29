"""输入区组件：支持粘贴图片的输入框、窗口边缘拖拽手柄。"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QPlainTextEdit, QWidget


class ChatInput(QPlainTextEdit):
    """Enter 发送、Shift+Enter 换行，支持直接粘贴图片。"""

    image_pasted = pyqtSignal(str)
    send_requested = pyqtSignal()
    # 给宿主处理斜杠候选列表：上下选择 / 回车确认 / Esc 关闭
    nav_key = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        # 候选列表打开时，回车用于选中而不是直接发送
        self.popup_open = False

    def keyPressEvent(self, event) -> None:
        key = event.key()

        if self.popup_open and key in (
            Qt.Key_Up,
            Qt.Key_Down,
            Qt.Key_Return,
            Qt.Key_Enter,
            Qt.Key_Escape,
        ):
            self.nav_key.emit(key)
            return

        if key in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
            else:
                self.send_requested.emit()
            return
        super().keyPressEvent(event)

    def insertFromMimeData(self, source) -> None:
        if source is not None and source.hasImage():
            image = source.imageData()
            if isinstance(image, QImage) and not image.isNull():
                path = self._save_image(image)
                if path:
                    self.image_pasted.emit(path)
                    return
        super().insertFromMimeData(source)

    @staticmethod
    def _save_image(image) -> str:
        from datetime import datetime

        out_dir = Path(__file__).resolve().parents[2] / "data" / "attachments"
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"paste_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        if image.save(str(target), "PNG"):
            return str(target)
        return ""


class ResizeHandle(QWidget):
    """贴在窗口四边四角的透明手柄，负责拉伸。"""

    THICK = 6
    CORNER = 16

    CURSORS = {
        "left": Qt.SizeHorCursor,
        "right": Qt.SizeHorCursor,
        "top": Qt.SizeVerCursor,
        "bottom": Qt.SizeVerCursor,
        "topleft": Qt.SizeFDiagCursor,
        "bottomright": Qt.SizeFDiagCursor,
        "topright": Qt.SizeBDiagCursor,
        "bottomleft": Qt.SizeBDiagCursor,
    }

    def __init__(self, window, edge: str):
        super().__init__(window)
        self.window = window
        self.edge = edge
        self.setCursor(self.CURSORS[edge])
        self.setStyleSheet("background: transparent;")
        self._start = None
        self._geometry = None

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._start = event.globalPos()
            self._geometry = self.window.geometry()
            self.window._resizing = True
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._start is None or not (event.buttons() & Qt.LeftButton):
            return

        delta = event.globalPos() - self._start
        geo = self._geometry
        x, y, w, h = geo.x(), geo.y(), geo.width(), geo.height()

        min_w = self.window.minimumWidth()
        min_h = self.window.minimumHeight()

        # 左/上方向要同时移动原点：新尺寸钳制到下限后，位置按差值回推，
        # 否则尺寸卡在旧值不变，鼠标一往回拖就会跳回拉伸前的高度
        if "left" in self.edge:
            new_w = max(min_w, w - delta.x())
            x = geo.x() + (w - new_w)
            w = new_w
        if "right" in self.edge:
            w = max(min_w, w + delta.x())

        if "top" in self.edge:
            new_h = max(min_h, h - delta.y())
            y = geo.y() + (h - new_h)
            h = new_h
        if "bottom" in self.edge:
            h = max(min_h, h + delta.y())

        self.window.setGeometry(x, y, w, h)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._start = None
        self.window._resizing = False
        event.accept()
