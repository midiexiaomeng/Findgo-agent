"""文件树面板：目录浏览、图片预览、检索结果列表。"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QTreeWidgetItem

from ..utils.image_scan import SUPPORTED_EXTENSIONS as IMAGE_EXTS


class FileTreeMixin:
    """依赖宿主提供的属性：tree / path_label / preview / tree_root / results_mode / last_results。"""

    def _go_up(self) -> None:
        if self.results_mode:
            self.results_mode = False
            self._load_file_tree()
            return
        current = self.tree_root.resolve()
        parent = current.parent
        if parent == current:
            return
        self.tree_root = parent
        self._load_file_tree()

    def _load_file_tree(self) -> None:
        self.results_mode = False
        root = self.tree_root
        self.path_label.setText(str(root))
        self.path_label.setToolTip(str(root))
        self.tree.clear()

        try:
            entries = sorted(
                root.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
            )
        except OSError:
            entries = []

        for p in entries:
            if p.name in ("__pycache__", ".git", ".pytest_cache"):
                continue
            label = ("▸ " if p.is_dir() else "   ") + p.name
            item = QTreeWidgetItem([label])
            item.setData(0, Qt.UserRole, str(p))
            item.setToolTip(0, str(p))
            self.tree.addTopLevelItem(item)

    def _on_tree_double_click(self, item: QTreeWidgetItem, column: int) -> None:
        path = item.data(0, Qt.UserRole)
        if not path:
            return
        p = Path(path)
        if p.is_dir():
            self.tree_root = p
            self._load_file_tree()
        elif p.suffix.lower() in IMAGE_EXTS:
            self._show_preview(p)
        else:
            try:
                from PyQt5.QtCore import QUrl
                from PyQt5.QtGui import QDesktopServices

                QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
            except Exception:
                pass

    def _show_preview(self, p: Path) -> None:
        pixmap = QPixmap(str(p))
        if pixmap.isNull():
            return
        self.preview.setPixmap(
            pixmap.scaled(200, 130, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.preview.setToolTip(str(p))

    def _show_results(self) -> None:
        self.results_mode = True
        self.tree.clear()
        self.path_label.setText("最近检索结果")
        if not self.last_results:
            self.tree.addTopLevelItem(QTreeWidgetItem(["（暂无结果）"]))
            return
        for img in self.last_results:
            path = (
                img.get("path", "") if isinstance(img, dict) else getattr(img, "path", "")
            )
            if not path or not Path(path).exists():
                continue
            item = QTreeWidgetItem([Path(path).name])
            item.setData(0, Qt.UserRole, path)
            item.setToolTip(0, path)
            self.tree.addTopLevelItem(item)

    def _refresh_results(self) -> None:
        if self.results_mode:
            self._show_results()

    def on_file_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        path = item.data(0, Qt.UserRole)
        if not path:
            return
        p = Path(path)
        if p.suffix.lower() in IMAGE_EXTS and p.exists():
            self._show_preview(p)
            return
        self.preview.setText(p.name)
        self.preview.setPixmap(QPixmap())
