"""消息气泡与图库弹窗。"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QVBoxLayout,
)

from ..theme import GLOBAL_QSS


class ImageGalleryDialog(QDialog):
    """全部图片：左侧文件名列表，悬停预览，双击放大。"""

    def __init__(self, paths, parent=None):
        super().__init__(parent)
        self.paths = [str(p) for p in paths if p]
        self.setWindowTitle(f"全部图片（{len(self.paths)} 张）")
        self.resize(720, 480)
        self.setStyleSheet(GLOBAL_QSS)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.list = QListWidget()
        self.list.setFixedWidth(240)
        self.list.setMouseTracking(True)
        for path in self.paths:
            item = QListWidgetItem(Path(path).name)
            item.setData(Qt.UserRole, path)
            item.setToolTip(path)
            self.list.addItem(item)
        self.list.itemEntered.connect(self._show_preview)
        self.list.itemDoubleClicked.connect(self._open_large)

        self.preview = QLabel("把鼠标移到文件名上预览")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(420, 380)
        self.preview.setStyleSheet(
            "background:#F8F7F3;border-radius:8px;color:#A8A8A8;font-size:12px;"
        )

        layout.addWidget(self.list)
        layout.addWidget(self.preview, 1)

    def _show_preview(self, item) -> None:
        path = item.data(Qt.UserRole)
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.preview.setText("无法预览")
            return
        self.preview.setPixmap(
            pixmap.scaled(
                self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def _open_large(self, item) -> None:
        path = item.data(Qt.UserRole)
        dialog = QDialog(self)
        dialog.setWindowTitle(Path(path).name)
        dialog.resize(960, 720)
        dialog.setStyleSheet(GLOBAL_QSS)
        layout = QVBoxLayout(dialog)
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            label.setPixmap(
                pixmap.scaled(920, 680, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        layout.addWidget(label)
        dialog.exec_()


class BubbleLabel(QLabel):
    """圆角气泡：QLabel 支持 border-radius，QTextBrowser 的富文本不支持。"""

    def __init__(self, role: str, text: str, images: list | None, host, parent=None):
        super().__init__(parent)
        self.host = host
        self.setWordWrap(True)
        self.setTextFormat(Qt.RichText)
        self.setOpenExternalLinks(False)
        self.linkActivated.connect(host.on_chat_link)
        self.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse
        )

        body = host._message_html(text, images)
        self.setText(body)

        bg = "#E3EFE7" if role == "你" else "#F5F3EE"
        self.setStyleSheet(
            f"QLabel{{background:{bg};color:#2E2E2E;border-radius:16px;"
            "padding:18px 24px;font-size:13px;}"
        )
        self.setMaximumWidth(640)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
