"""Skills 面板：拖入技能包安装。"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QLabel,
    QListWidget,
    QVBoxLayout,
)


def _skills_root() -> Path:
    """data/skills 位于项目根目录。"""
    return Path(__file__).resolve().parents[3] / "data" / "skills"


class SkillDropZone(QFrame):
    """虚线圆角气泡，接收拖入的技能包。"""

    dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(110)
        self.setStyleSheet(
            "QFrame{background:#F8FBF9;border:2px dashed #A8C9B8;" "border-radius:16px;}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        tip = QLabel("把技能包拖到这里\n支持 .zip / .skill / 文件夹")
        tip.setAlignment(Qt.AlignCenter)
        tip.setStyleSheet("color:#6F9C86; font-size:12px; border:none;")
        layout.addWidget(tip)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self.dropped.emit(path)
        event.acceptProposedAction()


class SkillsPopup(QFrame):
    """挨着 Skills 按钮弹出的小面板。"""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Popup)
        self.setWindowFlags(Qt.Popup)
        self.setFixedSize(330, 400)
        self.setStyleSheet(
            "QFrame{background:#FBFAF7;border:1px solid #EDEBE6;border-radius:14px;}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        title = QLabel("Skills")
        title.setStyleSheet("font-weight:600; font-size:13px; color:#2E2E2E;")
        layout.addWidget(title)

        self.drop_zone = SkillDropZone()
        self.drop_zone.dropped.connect(self.on_dropped)
        layout.addWidget(self.drop_zone)

        self.status = QLabel("尚未安装技能")
        self.status.setStyleSheet("color:#8A8A8A; font-size:11px;")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.list = QListWidget()
        self.list.setStyleSheet(
            "QListWidget{border:1px solid #EDEBE6;border-radius:10px;"
            "background:#FFFFFF;padding:4px;font-size:12px;}"
        )
        layout.addWidget(self.list, 1)

        self._refresh()

    def _refresh(self) -> None:
        root = _skills_root()
        self.list.clear()
        if not root.exists():
            return
        items = sorted(p.name for p in root.iterdir() if p.is_dir())
        for name in items:
            self.list.addItem(name)
        self.status.setText(f"已安装 {len(items)} 个技能" if items else "尚未安装技能")

    def on_dropped(self, path: str) -> None:
        source = Path(path)
        if not source.exists():
            self.status.setText("文件不存在")
            return

        try:
            name = self._install(source)
        except Exception as e:
            self.status.setText(f"安装失败：{e}")
            return

        self.status.setText(f"已安装：{name}")
        self._refresh()

    def _install(self, source: Path) -> str:
        root = _skills_root()
        root.mkdir(parents=True, exist_ok=True)

        if source.is_dir():
            target = root / source.name
            shutil.copytree(source, target, dirs_exist_ok=True)
            return source.name

        if source.suffix.lower() in (".zip", ".skill"):
            target = root / source.stem
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(source) as archive:
                for member in archive.namelist():
                    parts = Path(member).parts
                    if ".." in parts or Path(member).is_absolute():
                        continue  # 防路径穿越
                    archive.extract(member, target)
            return source.stem

        target = root / source.name
        shutil.copy2(source, target)
        return source.name
