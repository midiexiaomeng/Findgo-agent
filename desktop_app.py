"""imgsearch Agent 桌面应用（PyQt5）。

布局：
┌──────┬──────────────────────────┬──────┐
│ 日志 │ 对话记录（持续追加）      │ 任务 │
│      ├──────────────────────────┤      │
│ 记忆 │ 输入框（多行 / 模型 / 发送）│ 文件 │
└──────┴──────────────────────────┴──────┘

中间列宽度为 4，左右各 1，左右合计正好是中间的一半。
"""

from __future__ import annotations

import html
import logging
import os
import sys
import warnings
from pathlib import Path

# 避免 transformers 顺带导入 tensorflow（约 15 秒）
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings("ignore", message=".*numexpr.*")
warnings.filterwarnings("ignore", message=".*bottleneck.*")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from PyQt5.QtCore import QEvent, QObject, QPoint, QRect, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QShortcut,
    QSizeGrip,
    QSizePolicy,
    QSplitter,
    QTextBrowser,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from imgsearch_agent.model_store import ModelStore
from imgsearch_agent.settings_dialog import SettingsDialog
from imgsearch_agent.utils.schemas import TaskSpec

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


# ── 日志桥接 ──────────────────────────────────────────────────────────────────


class LogBridge(QObject):
    message = pyqtSignal(str)


class QtLogHandler(logging.Handler):
    def __init__(self, bridge: LogBridge):
        super().__init__()
        self.bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.bridge.message.emit(self.format(record))
        except Exception:
            pass


# ── 后台线程 ──────────────────────────────────────────────────────────────────


from imgsearch_agent.theme import GLOBAL_QSS




class ChatInput(QPlainTextEdit):
    """支持粘贴图片的输入框。"""

    image_pasted = pyqtSignal(str)

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

        from PyQt5.QtGui import QImage

        out_dir = Path(__file__).resolve().parent / "data" / "attachments"
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"paste_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        if image.save(str(target), "PNG"):
            return str(target)
        return ""


class Worker(QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(object)

    def __init__(self, fn, *args):
        super().__init__()
        self.fn = fn
        self.args = args

    def run(self) -> None:
        try:
            self.done.emit(self.fn(*self.args))
        except Exception as e:
            self.failed.emit(str(e))


# ── 主窗口 ────────────────────────────────────────────────────────────────────


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
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._start is None or not (event.buttons() & Qt.LeftButton):
            return

        delta = event.globalPos() - self._start
        geo = self._geometry
        x, y, w, h = geo.x(), geo.y(), geo.width(), geo.height()

        min_w = max(self.window.minimumWidth(), 400)
        min_h = max(self.window.minimumHeight(), 300)

        if "left" in self.edge:
            new_w = w - delta.x()
            if new_w >= min_w:
                x += delta.x()
                w = new_w
        if "right" in self.edge:
            w = max(min_w, w + delta.x())
        if "top" in self.edge:
            new_h = h - delta.y()
            if new_h >= min_h:
                y += delta.y()
                h = new_h
        if "bottom" in self.edge:
            h = max(min_h, h + delta.y())

        self.window.setGeometry(x, y, w, h)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._start = None
        event.accept()


class TitleBar(QWidget):
    """无边框窗口的自定义标题栏：拖拽移动、双击最大化。"""

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

        title = QLabel("imgsearch · 多模态图像检索")
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


class SkillDropZone(QFrame):
    """虚线圆角气泡，接收拖入的技能包。"""

    dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(110)
        self.setStyleSheet(
            "QFrame{background:#F8FBF9;border:2px dashed #A8C9B8;"
            "border-radius:16px;}"
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

    # ── 安装 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _skills_root() -> Path:
        return Path(__file__).resolve().parent / "data" / "skills"

    def _refresh(self) -> None:
        root = self._skills_root()
        self.list.clear()
        if not root.exists():
            return
        items = sorted(p.name for p in root.iterdir() if p.is_dir())
        for name in items:
            self.list.addItem(name)
        self.status.setText(
            f"已安装 {len(items)} 个技能" if items else "尚未安装技能"
        )

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
        import shutil
        import zipfile

        root = self._skills_root()
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


class DynamicToolDialog(QDialog):
    """动态工具审核：先看源码和安全提示，再决定是否允许。"""

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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("imgsearch Agent · 多模态图像检索")
        self.setMinimumSize(880, 560)

        screen = QApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            width = min(1200, int(area.width() * 0.82))
            height = min(760, int(area.height() * 0.82))
            self.resize(max(width, 900), max(height, 600))
            self.move(
                area.x() + (area.width() - max(width, 900)) // 2,
                area.y() + (area.height() - max(height, 600)) // 2,
            )
        else:
            self.resize(1100, 700)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self._left_width = 240
        self._right_width = 300
        self.setStyleSheet(
            "QMainWindow{background:#FBFAF7;}"
            "QScrollBar:vertical{background:transparent;width:8px;}"
            "QScrollBar::handle:vertical{background:#DEDBD3;border-radius:4px;min-height:24px;}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}"
        )

        self.worker: Worker | None = None
        self._cancel = False
        self.agent = None
        self.conversation: list[tuple[str, str]] = []
        self.last_results: list[dict] = []
        self._attachment: str = ""
        self._galleries: dict[str, list] = {}
        self.tree_root: Path = ROOT
        self.results_mode = False

        self._build_ui()
        self._setup_logging()
        self._load_file_tree()
        self._refresh_model_button()
        self._adjust_input_height()
        self._restore_history()
        self._prepare_agent()
        self._warmup()

    # ── 中间上：对话记录 ──────────────────────────────────────────────────

    def _build_chat_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 4)
        layout.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("对话")
        title.setStyleSheet("font-weight:600; color:#4A4A4A; font-size:13px;")
        header.addWidget(title)
        header.addStretch(1)

        self.clear_chat_btn = QPushButton("清屏")
        self.clear_chat_btn.setFixedHeight(22)
        self.clear_chat_btn.setStyleSheet(
            "QPushButton{border:1px solid #d1d5db;border-radius:5px;"
            "padding:2px 10px;background:#fff;font-size:11px;}"
        )
        self.clear_chat_btn.clicked.connect(self.on_clear_chat)

        self.settings_btn = QPushButton("设置")
        self.settings_btn.setFixedHeight(22)
        self.settings_btn.setStyleSheet(
            "QPushButton{border:1px solid #d1d5db;border-radius:5px;"
            "padding:2px 10px;background:#fff;font-size:11px;}"
        )
        self.settings_btn.clicked.connect(self.on_settings)

        header.addWidget(self.clear_chat_btn)
        header.addWidget(self.settings_btn)
        layout.addLayout(header)

        self.chat_view = QTextBrowser()
        self.chat_view.setOpenExternalLinks(False)
        self.chat_view.setOpenLinks(False)
        self.chat_view.anchorClicked.connect(self.on_chat_link)
        self.chat_view.setStyleSheet(
            "QTextBrowser{border:1px solid #EDEBE6; border-radius:12px; "
            "background:#FFFFFF; padding:16px; font-size:13px;}"
        )
        layout.addWidget(self.chat_view, 1)
        return panel

    # ── 中间下：输入框 ────────────────────────────────────────────────────

    def _build_input_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 2, 10, 8)
        layout.setSpacing(4)

        # 附件预览（默认隐藏）
        self.attach_bar = QWidget()
        attach_layout = QHBoxLayout(self.attach_bar)
        attach_layout.setContentsMargins(2, 2, 2, 2)
        attach_layout.setSpacing(8)

        self.attach_thumb = QLabel()
        self.attach_thumb.setFixedSize(40, 40)
        self.attach_thumb.setStyleSheet(
            "border:1px solid #EDEBE6;border-radius:6px;background:#FFFFFF;"
        )
        self.attach_thumb.setAlignment(Qt.AlignCenter)

        self.attach_name = QLabel("")
        self.attach_name.setStyleSheet("color:#4A4A4A; font-size:12px;")

        remove_btn = QToolButton()
        remove_btn.setText("×")
        remove_btn.setToolTip("移除附件")
        remove_btn.setStyleSheet(
            "QToolButton{border:none;color:#8A8A8A;font-size:14px;padding:2px 6px;}"
            "QToolButton:hover{color:#C97B6E;}"
        )
        remove_btn.clicked.connect(self._clear_attachment)

        attach_layout.addWidget(self.attach_thumb)
        attach_layout.addWidget(self.attach_name, 1)
        attach_layout.addWidget(remove_btn)
        self.attach_bar.setVisible(False)
        layout.addWidget(self.attach_bar)

        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        self.input = ChatInput()
        self.input.image_pasted.connect(self._set_attachment)
        self.input.setPlaceholderText(
            "描述你想找的图片，支持多行。Ctrl+Enter 发送，例如：\n找两张猫的图片，再找几张海边的日落"
        )
        self.input.setStyleSheet(
            "QPlainTextEdit{border:1px solid #EDEBE6;border-radius:10px;"
            "padding:10px 12px 46px 12px;font-size:14px;background:#ffffff;}"
        )
        self.input.textChanged.connect(self._adjust_input_height)
        grid.addWidget(self.input, 0, 0)

        overlay = QWidget()
        overlay.setStyleSheet("background: transparent;")
        btn_row = QHBoxLayout(overlay)
        btn_row.setContentsMargins(0, 0, 8, 8)
        btn_row.setSpacing(6)

        self._search_mode = "local"
        self.mode_btn = QToolButton()
        self.mode_btn.setFixedSize(34, 28)
        self.mode_btn.setStyleSheet(
            "QToolButton{border:1px solid #EDEBE6;border-radius:7px;"
            "background:#F8F7F3;font-size:13px;}"
            "QToolButton:hover{background:#F2F0EA;}"
        )
        self.mode_btn.clicked.connect(self._cycle_mode)
        self._refresh_mode_btn()

        self.model_btn = QToolButton()
        self.model_btn.setText("选择模型")
        self.model_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.model_btn.setStyleSheet(
            "QToolButton{border:1px solid #d1d5db;border-radius:7px;"
            "padding:6px 12px;background:#F8F7F3;font-size:12px;}"
        )
        self.model_btn.clicked.connect(self.on_model_menu)

        self.send_btn = QPushButton("发送")
        self.send_btn.setStyleSheet(
            "QPushButton{background:#6F9C86;color:white;border:none;"
            "border-radius:7px;padding:7px 20px;font-size:13px;}"
            "QPushButton:disabled{background:#A8C9B8;}"
        )
        self.send_btn.clicked.connect(self.on_send)

        btn_row.addWidget(self.mode_btn)
        btn_row.addWidget(self.model_btn)
        btn_row.addWidget(self.send_btn)
        grid.addWidget(overlay, 0, 0, Qt.AlignBottom | Qt.AlignRight)

        # 左下角加号：添加附件做以图搜图
        left_overlay = QWidget()
        left_overlay.setStyleSheet("background: transparent;")
        left_row = QHBoxLayout(left_overlay)
        left_row.setContentsMargins(8, 0, 0, 8)
        left_row.setSpacing(6)

        self.attach_btn = QToolButton()
        self.attach_btn.setText("+")
        self.attach_btn.setToolTip("添加图片，按图搜索（也支持 Ctrl+V 粘贴）")
        self.attach_btn.setFixedSize(28, 28)
        self.attach_btn.setStyleSheet(
            "QToolButton{border:1px solid #EDEBE6;border-radius:8px;"
            "background:#FFFFFF;font-size:16px;color:#4A4A4A;}"
            "QToolButton:hover{background:#F2F0EA;}"
        )
        self.attach_btn.clicked.connect(self.on_attach)
        left_row.addWidget(self.attach_btn)
        left_row.addStretch(1)
        grid.addWidget(left_overlay, 0, 0, Qt.AlignBottom | Qt.AlignLeft)

        layout.addWidget(container)

        self.state_label = QLabel("输入需求后点发送，Agent 会自动规划并检索。")
        self.state_label.setStyleSheet("font-size:12px; color:#8A8A8A;")
        self.state_label.setWordWrap(True)
        layout.addWidget(self.state_label)

        QShortcut(QKeySequence("Ctrl+Return"), self.input, activated=self.on_send)
        return panel

    def _adjust_input_height(self) -> None:
        text = self.input.toPlainText() or " "
        width = max(self.input.viewport().width() - 4, 120)
        rect = self.input.fontMetrics().boundingRect(
            QRect(0, 0, width, 10000), Qt.TextWordWrap, text
        )
        content = rect.height() + 46 + 16
        window_h = self.height() or 880
        max_h = max(110, int(window_h * 0.25))
        target = int(min(max(content, 56), max_h))

        if target != self.input.height():
            self.input.setFixedHeight(target)

        # 还没到最大高度时，保持从第一行开始显示；
        # 超过上限后才交给滚动条跟随光标
        if target < max_h:
            self.input.verticalScrollBar().setValue(0)

    def _layout_handles(self) -> None:
        if not hasattr(self, "_resize_handles"):
            return

        w, h = self.width(), self.height()
        t = ResizeHandle.THICK
        c = ResizeHandle.CORNER

        self._resize_handles["left"].setGeometry(0, c, t, max(0, h - 2 * c))
        self._resize_handles["right"].setGeometry(w - t, c, t, max(0, h - 2 * c))
        self._resize_handles["top"].setGeometry(c, 0, max(0, w - 2 * c), t)
        self._resize_handles["bottom"].setGeometry(c, h - t, max(0, w - 2 * c), t)

        self._resize_handles["topleft"].setGeometry(0, 0, c, c)
        self._resize_handles["topright"].setGeometry(w - c, 0, c, c)
        self._resize_handles["bottomleft"].setGeometry(0, h - c, c, c)
        self._resize_handles["bottomright"].setGeometry(w - c, h - c, c, c)

        for handle in self._resize_handles.values():
            handle.raise_()

    def _apply_round_mask(self) -> None:
        """保留接口；圆角由容器的 border-radius 实现。

        不用 setMask：遮罩会把右下角的拖拽手柄一起裁掉，导致窗口无法拉伸。
        透明背景 + 容器圆角在视觉上效果一致，但边缘仍然可交互。
        """
        if self.mask() and not self.mask().isEmpty():
            self.clearMask()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._apply_round_mask()
        self._layout_handles()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._adjust_input_height()
        self._apply_round_mask()
        self._layout_handles()

    # ── 左侧 ──────────────────────────────────────────────────────────────

    def _build_left_column(self) -> QSplitter:
        split = QSplitter(Qt.Vertical)

        log_panel = QWidget()
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(6, 6, 6, 6)
        log_layout.setSpacing(4)

        log_header = QHBoxLayout()
        log_title = QLabel("运行日志")
        log_title.setStyleSheet("font-weight:600; color:#4A4A4A; font-size:12px;")
        self.new_chat_btn = QPushButton("新对话")
        self.new_chat_btn.setFixedHeight(22)
        self.new_chat_btn.setStyleSheet(
            "QPushButton{border:1px solid #d1d5db;border-radius:5px;"
            "padding:2px 8px;background:#fff;font-size:11px;}"
            "QPushButton:hover{background:#F2F0EA;}"
        )
        self.new_chat_btn.clicked.connect(self.on_new_chat)
        log_header.addWidget(log_title)
        log_header.addStretch(1)
        log_header.addWidget(self.new_chat_btn)
        log_layout.addLayout(log_header)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(3000)
        font = QFont("Consolas")
        font.setPointSize(8)
        self.log_view.setFont(font)
        self.log_view.setStyleSheet(
            "QPlainTextEdit{border:1px solid #EDEBE6; border-radius:6px; "
            "background:#F8F7F3; color:#4A4A4A; padding:6px;}"
        )
        log_layout.addWidget(self.log_view, 1)

        self.skills_btn = QPushButton("Skills")
        self.skills_btn.setFixedHeight(24)
        self.skills_btn.setToolTip("安装和管理技能包")
        self.skills_btn.setStyleSheet(
            "QPushButton{border:1px solid #EDEBE6;border-radius:8px;"
            "padding:2px 10px;background:#FFFFFF;font-size:11.5px;color:#2E2E2E;}"
            "QPushButton:hover{background:#F2F0EA;}"
        )
        self.skills_btn.clicked.connect(self.on_skills)
        log_layout.addWidget(self.skills_btn)

        split.addWidget(log_panel)

        mem_panel = QWidget()
        mem_layout = QVBoxLayout(mem_panel)
        mem_layout.setContentsMargins(6, 6, 6, 6)
        mem_layout.setSpacing(4)

        mem_title = QLabel("任务记忆")
        mem_title.setStyleSheet("font-weight:600; color:#4A4A4A; font-size:12px;")
        mem_layout.addWidget(mem_title)

        mem_split = QSplitter(Qt.Vertical)

        self.session_list = QListWidget()
        self.session_list.setMaximumHeight(110)
        self.session_list.setStyleSheet(
            "QListWidget{border:1px solid #EDEBE6;border-radius:6px;"
            "background:#fff;padding:2px;font-size:11px;}"
            "QListWidget::item{padding:3px 4px;}"
            "QListWidget::item:selected{background:#E3EFE7;color:#2E2E2E;}"
        )
        self.session_list.itemClicked.connect(self.on_session_clicked)
        self.session_list.itemDoubleClicked.connect(self.on_rename_session)
        self.session_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.session_list.customContextMenuRequested.connect(self._session_menu)

        self.memory_view = QTextBrowser()
        self.memory_view.setStyleSheet(
            "QTextBrowser{border:1px solid #EDEBE6; border-radius:6px; "
            "background:#ffffff; padding:6px; font-size:11.5px;}"
        )

        mem_split.addWidget(self.session_list)
        mem_split.addWidget(self.memory_view)
        mem_split.setSizes([90, 230])
        mem_layout.addWidget(mem_split, 1)
        split.addWidget(mem_panel)

        split.setSizes([360, 420])
        return split

    # ── 右侧 ──────────────────────────────────────────────────────────────

    def _build_task_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        title = QLabel("推理任务链")
        title.setStyleSheet("font-weight:600; color:#4A4A4A; font-size:12px;")
        layout.addWidget(title)

        self.task_list = QListWidget()
        self.task_list.setStyleSheet(
            "QListWidget{border:1px solid #EDEBE6; border-radius:6px; "
            "background:#ffffff; padding:3px; font-size:11.5px;}"
        )
        self.task_list.setDragDropMode(QListWidget.InternalMove)
        layout.addWidget(self.task_list, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(3)
        self.add_btn = QPushButton("+")
        self.del_btn = QPushButton("−")
        self.up_btn = QPushButton("↑")
        self.down_btn = QPushButton("↓")
        for b in (self.add_btn, self.del_btn, self.up_btn, self.down_btn):
            b.setFixedWidth(28)
            b.setStyleSheet(
                "QPushButton{border:1px solid #d1d5db;border-radius:5px;"
                "padding:3px 0;background:#fff;font-size:12px;}"
            )
        self.add_btn.clicked.connect(self.on_add_task)
        self.del_btn.clicked.connect(self.on_del_task)
        self.up_btn.clicked.connect(lambda: self._move_task(-1))
        self.down_btn.clicked.connect(lambda: self._move_task(1))
        for b in (self.add_btn, self.del_btn, self.up_btn, self.down_btn):
            btn_row.addWidget(b)

        self.run_btn = QPushButton("↻")
        self.run_btn.setFixedWidth(30)
        self.run_btn.setStyleSheet(
            "QPushButton{background:#6F9C86;color:white;border:none;"
            "border-radius:6px;padding:4px 0;font-size:13px;}"
            "QPushButton:disabled{background:#A8C9B8;}"
        )
        self.run_btn.setToolTip("重跑：按当前编辑后的任务链重新执行")
        self.run_btn.clicked.connect(self.on_run_tasks)

        self.stop_btn = QPushButton("■")
        self.stop_btn.setFixedWidth(30)
        self.stop_btn.setStyleSheet(
            "QPushButton{background:#C97B6E;color:white;border:none;"
            "border-radius:6px;padding:4px 0;font-size:13px;}"
            "QPushButton:disabled{background:#E2BCB3;}"
        )
        self.stop_btn.setToolTip("停止：中断后续任务，当前这一条检索会先跑完")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.on_stop)

        # 右栏可能被拖得很窄，按钮用图标 + tooltip，避免文字被裁掉
        self.resume_btn = QPushButton("▶")
        self.resume_btn.setFixedWidth(30)
        self.resume_btn.setStyleSheet(
            "QPushButton{background:#6F9C86;color:white;border:none;"
            "border-radius:6px;padding:4px 0;font-size:13px;}"
        )
        self.resume_btn.setToolTip("继续执行上次未完成的任务")
        self.resume_btn.setVisible(False)
        self.resume_btn.clicked.connect(self.on_run_tasks)

        btn_row.addStretch(1)
        btn_row.addWidget(self.resume_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.run_btn)
        layout.addLayout(btn_row)
        return panel

    def _build_file_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        title = QLabel("文件与资源")
        title.setStyleSheet("font-weight:600; color:#4A4A4A; font-size:12px;")
        layout.addWidget(title)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(4)
        self.up_btn = QPushButton("↑")
        self.up_btn.setFixedWidth(26)
        self.up_btn.setToolTip("返回上级目录")
        self.up_btn.setStyleSheet(
            "QPushButton{border:1px solid #d1d5db;border-radius:5px;"
            "padding:3px 0;background:#fff;font-size:12px;}"
        )
        self.up_btn.clicked.connect(self._go_up)
        self.path_label = QLineEdit(str(ROOT))
        self.path_label.setReadOnly(True)
        self.path_label.setFrame(False)
        self.path_label.setFocusPolicy(Qt.NoFocus)
        self.path_label.setMinimumWidth(0)
        self.path_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.path_label.setStyleSheet(
            "QLineEdit{background:transparent;border:none;"
            "font-size:10.5px;color:#8A8A8A;padding:0;}"
        )
        self.results_btn = QPushButton("结果")
        self.results_btn.setFixedWidth(40)
        self.results_btn.setToolTip("查看最近检索结果")
        self.results_btn.setStyleSheet(
            "QPushButton{border:1px solid #d1d5db;border-radius:5px;"
            "padding:3px 0;background:#fff;font-size:11px;}"
        )
        self.results_btn.clicked.connect(self._show_results)
        nav_row.addWidget(self.up_btn)
        nav_row.addWidget(self.path_label, 1)
        nav_row.addWidget(self.results_btn)
        layout.addLayout(nav_row)

        split = QSplitter(Qt.Vertical)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(False)
        self.tree.setStyleSheet(
            "QTreeWidget{border:1px solid #EDEBE6; border-radius:6px; "
            "background:#ffffff; font-size:11.5px;}"
        )
        self.tree.itemClicked.connect(self.on_file_clicked)
        self.tree.itemDoubleClicked.connect(self._on_tree_double_click)

        self.preview = QLabel("点击图片预览")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(110)
        self.preview.setStyleSheet(
            "QLabel{border:1px solid #EDEBE6; border-radius:6px; "
            "background:#fafafa; color:#A8A8A8; font-size:11px;}"
        )

        split.addWidget(self.tree)
        split.addWidget(self.preview)
        split.setSizes([300, 150])
        layout.addWidget(split, 1)
        return panel

    # ── 组装：1 : 4 : 1 ───────────────────────────────────────────────────

    def _build_ui(self) -> None:
        left = self._build_left_column()

        center = QSplitter(Qt.Vertical)
        center.addWidget(self._build_chat_panel())
        center.addWidget(self._build_input_panel())
        center.setSizes([560, 190])
        center.setChildrenCollapsible(False)

        right = QSplitter(Qt.Vertical)
        right.addWidget(self._build_task_panel())
        right.addWidget(self._build_file_panel())
        right.setSizes([400, 380])
        right.setChildrenCollapsible(False)

        # 三列之间用水平分割器，左右宽度可以自由拖动
        main = QSplitter(Qt.Horizontal)
        main.addWidget(left)
        main.addWidget(center)
        main.addWidget(right)
        main.setSizes([240, 900, 300])
        main.setStretchFactor(0, 0)
        main.setStretchFactor(1, 1)
        main.setStretchFactor(2, 0)
        main.setChildrenCollapsible(False)

        left.setMinimumWidth(160)
        center.setMinimumWidth(320)
        right.setMinimumWidth(200)

        self.main_split = main
        self.title_bar = TitleBar(self)

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addWidget(self.title_bar)
        vbox.addWidget(main, 1)

        self._container = container
        self._resize_handles = {}
        for edge in (
            "left",
            "right",
            "top",
            "bottom",
            "topleft",
            "topright",
            "bottomleft",
            "bottomright",
        ):
            handle = ResizeHandle(self, edge)
            handle.raise_()
            self._resize_handles[edge] = handle

        self.setAttribute(Qt.WA_TranslucentBackground, True)
        container.setObjectName("rootFrame")
        container.setStyleSheet(
            "#rootFrame{background:#FBFAF7; border-radius:14px;}"
        )
        self.setCentralWidget(container)
        self.setStyleSheet("QMainWindow{background:transparent;}")

    # ── 日志 / 记忆 ───────────────────────────────────────────────────────

    def _setup_logging(self) -> None:
        self.bridge = LogBridge()
        self.bridge.message.connect(self.append_log)
        handler = QtLogHandler(self.bridge)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-5s | %(name)-28s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root = logging.getLogger("imgsearch_agent")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        for noisy in ("httpx", "urllib3", "chromadb", "PIL"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    def append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)
        bar = self.log_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _render_memory(self) -> None:
        rows = []
        for who, text in self.conversation[-80:]:
            safe = html.escape(str(text)).replace("\n", "<br/>")
            color = "#6F9C86" if who == "你" else "#2E2E2E"
            rows.append(
                f'<p style="margin:5px 0;color:{color};"><b>{who}：</b>{safe}</p>'
            )
        self.memory_view.setHtml(
            "".join(rows) or "<p style='color:#A8A8A8'>暂无对话</p>"
        )
        bar = self.memory_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _warmup(self) -> None:
        """后台加载 CLIP，避免第一次检索时再等。"""
        import threading

        def run():
            try:
                agent = self._get_agent()
                tool = agent.tools.get("image_retrieval")
                if tool is None:
                    return
                # pipeline 已在主线程创建，这里只加载权重
                tool.pipeline.encoder.load()
                self.bridge.message.emit(">>> CLIP 预热完成")
            except Exception as e:
                self.bridge.message.emit(f">>> CLIP 预热失败: {e}")

        threading.Thread(target=run, daemon=True).start()

    def _prepare_agent(self) -> None:
        """主线程创建 Agent 与 Pipeline，避免后台线程碰 SQLite / Qdrant 连接。"""
        try:
            agent = self._get_agent()
            tool = agent.tools.get("image_retrieval")
            if tool is not None:
                _ = tool.pipeline  # 触发一次构建，连接留在主线程
        except Exception as e:
            self.append_log("初始化检索服务失败: " + str(e))

    # ── 聊天记录 ──────────────────────────────────────────────────────────

    def _thumb_path(self, path: str, size: int = 112) -> str:
        """生成正方形缩略图，保证对话里图片高度一致。"""
        import hashlib

        from PIL import Image, ImageOps

        source = Path(path)
        try:
            stat = source.stat()
        except OSError:
            return ""

        key = hashlib.md5(
            f"{source.resolve()}|{stat.st_mtime}|{size}".encode("utf-8")
        ).hexdigest()
        cache_dir = Path(__file__).resolve().parent / "data" / "thumbs"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / f"{key}.jpg"
        if target.exists():
            return str(target)

        try:
            with Image.open(source) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im = ImageOps.fit(im, (size, size), method=Image.LANCZOS)
                im.save(target, "JPEG", quality=85)
        except Exception:
            return ""
        return str(target)

    def _highlight_paths(self, text: str) -> str:
        """把路径这类技术文本做淡绿高亮，保留可读性。"""
        import re

        escaped = html.escape(str(text))
        pattern = r"([A-Za-z]:\\[^\s<>\"']+|(?<![\w])/[\w./-]{3,})"
        return re.sub(
            pattern,
            '<span style="color:#4E7A63;background:#EAF3EE;'
            'padding:1px 4px;border-radius:3px;">'
            + chr(92) + "1</span>",
            escaped,
        )

    def _chat_html(self, role: str, text: str, images: list | None = None) -> str:
        safe = self._highlight_paths(text).replace("\n", "<br/>")
        if role == "你":
            return (
                '<div style="margin:10px 0;text-align:right;">'
                '<span style="display:inline-block;background:#E3EFE7;color:#2E2E2E;'
                'padding:14px 18px;border-radius:16px;max-width:76%;min-width:120px;'
                'text-align:left;font-size:13px;line-height:1.65;">'
                + safe + "</span></div>"
            )

        thumbs = ""
        if images:
            parts = []
            shown = 0
            for img in images[:20]:
                if isinstance(img, str):
                    path = img
                elif isinstance(img, dict):
                    path = img.get("path", "")
                else:
                    path = getattr(img, "path", "")
                if not path or not Path(path).exists():
                    continue
                thumb = self._thumb_path(path)
                if not thumb:
                    continue
                try:
                    full_uri = Path(path).resolve().as_uri()
                    thumb_uri = Path(thumb).resolve().as_uri()
                except ValueError:
                    continue
                parts.append(
                    '<a href="' + full_uri + '">'
                    '<img src="' + thumb_uri + '" width="112" height="112" '
                    'style="margin:5px;border-radius:8px;vertical-align:middle;"></a>'
                )
                shown += 1

            if parts:
                # 不能用 line-height:0，Qt 会按行高算容器高度，导致图片不占位
                thumbs = (
                    '<div style="margin-top:10px;margin-bottom:6px;">'
                    + "".join(parts)
                    + "</div>"
                )
                rest = len(images) - shown
                if rest > 0:
                    gid = f"g{len(self._galleries)}"
                    self._galleries[gid] = list(images)
                    thumbs += (
                        '<div style="margin-top:8px;font-size:12px;">'
                        '<a href="gallery:' + gid + '">'
                        "查看全部 " + str(len(images)) + " 张</a></div>"
                    )

        if thumbs:
            # 有图片时用 table 布局：Qt 对 inline-block 内嵌块级元素的高度计算不可靠
            return (
                '<div style="margin:10px 0;text-align:left;">'
                '<table cellspacing="0" cellpadding="0">'
                '<tr><td style="background:#F5F3EE;color:#2E2E2E;'
                'padding:14px 18px;border-radius:16px;'
                'font-size:13px;line-height:1.65;">'
                + safe + thumbs
                + "</td></tr></table></div>"
            )

        return (
            '<div style="margin:10px 0;text-align:left;">'
            '<span style="display:inline-block;background:#F5F3EE;color:#2E2E2E;'
            'padding:14px 18px;border-radius:16px;max-width:76%;min-width:120px;'
            'font-size:13px;line-height:1.65;">'
            + safe + "</span></div>"
        )

    def _append_chat(self, role: str, text: str, images: list | None = None) -> None:
        self.chat_view.append(self._chat_html(role, text, images))
        bar = self.chat_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def on_chat_link(self, url) -> None:
        if url.scheme() == "gallery":
            paths = self._galleries.get(url.path(), [])
            if paths:
                ImageGalleryDialog(paths, self).exec_()
            return

        path = url.toLocalFile()
        if path and Path(path).suffix.lower() in IMAGE_EXTS and Path(path).exists():
            self._open_image_dialog(path)

    def _open_image_dialog(self, path: str) -> None:
        from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout

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

    def on_tasks_ready(self, tasks) -> None:
        self.task_list.clear()
        for t in tasks:
            self._add_task_item(t)

        risky = [t for t in tasks if getattr(t, "risk", "normal") == "high"]
        if risky:
            self.state_label.setText(
                f"已规划 {len(tasks)} 条任务，其中 {len(risky)} 条涉及删除/清理，请留意"
            )
            self.append_log(">>> 注意：本轮包含高风险操作，删除需在设置面板确认")
        else:
            self.state_label.setText(f"已规划 {len(tasks)} 条任务，正在检索…")

    def _load_chat(self, session_id: str) -> None:
        self.chat_view.clear()
        self._galleries.clear()
        self.conversation = []
        if self.agent is None:
            return
        for m in self.agent.memory.load_messages(session_id):
            role = "你" if m.get("role") == "user" else "Agent"
            content = m.get("content", "")
            self.conversation.append((role, content))
            self._append_chat(role, content, m.get("images") or [])
        self._render_memory()

    def on_clear_chat(self) -> None:
        self.chat_view.clear()
        self.append_log(">>> 对话窗口已清屏（历史仍保存在会话中）")

    # ── 会话管理 ──────────────────────────────────────────────────────────

    def _refresh_sessions(self) -> None:
        self.session_list.clear()
        if self.agent is None:
            return
        keep = max(1, int(self.agent.config.memory_max_sessions))
        for s in self.agent.memory.list_sessions(limit=keep):
            mark = "● " if s["session_id"] == self.agent.memory.session_id else "○ "
            item = QListWidgetItem(mark + s["title"][:16] + "  (" + str(s["count"]) + ")")
            item.setData(Qt.UserRole, s["session_id"])
            item.setToolTip(s["updated_at"] + "\n" + s["session_id"])
            self.session_list.addItem(item)

    def on_new_chat(self) -> None:
        try:
            agent = self._get_agent()
            agent.memory.new_session()
            agent.memory.save_tasks([])
            agent.memory.prune_sessions(keep=max(1, int(agent.config.memory_max_sessions)))
            self.conversation = []
            self.last_results = []
            self.chat_view.clear()
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
            for t in agent.memory.load_tasks(sid):
                self._add_task_item(t)
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

    # ── 模型选择 ──────────────────────────────────────────────────────────

    def _refresh_model_button(self) -> None:
        entry = ModelStore().get_active()
        self.model_btn.setText(
            entry.get("label", entry.get("model", "选择模型")) if entry else "选择模型"
        )

    def on_model_menu(self) -> None:
        store = ModelStore()
        menu = QMenu(self)
        if not store.models:
            act = menu.addAction("尚未添加模型，点击添加…")
            act.triggered.connect(self.on_add_model)
        else:
            for entry in store.models:
                label = entry.get("label", entry.get("model", ""))
                prefix = "● " if entry.get("id") == store.active else "○ "
                act = menu.addAction(prefix + label)
                act.triggered.connect(
                    lambda checked=False, mid=entry.get("id"): self.on_select_model(mid)
                )
            menu.addSeparator()
            menu.addAction("添加模型…").triggered.connect(self.on_add_model)

        menu.exec_(self.model_btn.mapToGlobal(QPoint(0, self.model_btn.height())))

    def on_select_model(self, model_id: str) -> None:
        store = ModelStore()
        entry = next((m for m in store.models if m.get("id") == model_id), None)
        if entry is None:
            return
        store.set_active(model_id)
        if self.agent is not None:
            self.agent.set_llm(ModelStore.to_llm_config(entry))
        self.model_btn.setText(entry.get("label", entry.get("model", "")))
        self.append_log(">>> 切换模型：" + str(entry.get("label")))

    def toggle_maximize(self) -> None:
        if self.isMaximized():
            self.showNormal()
            self.title_bar.max_btn.setText("□")
        else:
            self.showMaximized()
            self.title_bar.max_btn.setText("❐")
        self._apply_round_mask()

    def toggle_left(self) -> None:
        widget = self.main_split.widget(0)
        sizes = list(self.main_split.sizes())
        if not widget.isHidden():
            if len(sizes) > 0 and sizes[0] > 0:
                self._left_width = sizes[0]
            widget.setHidden(True)
        else:
            widget.setHidden(False)
            if len(sizes) > 0:
                sizes[0] = self._left_width or 240
                self.main_split.setSizes(sizes)

    def toggle_right(self) -> None:
        widget = self.main_split.widget(2)
        sizes = list(self.main_split.sizes())
        if not widget.isHidden():
            if len(sizes) > 2 and sizes[2] > 0:
                self._right_width = sizes[2]
            widget.setHidden(True)
        else:
            widget.setHidden(False)
            if len(sizes) > 2:
                sizes[2] = self._right_width or 300
                self.main_split.setSizes(sizes)

    def reset_left(self) -> None:
        widget = self.main_split.widget(0)
        widget.setHidden(False)
        sizes = list(self.main_split.sizes())
        if len(sizes) > 0:
            sizes[0] = 240
            self.main_split.setSizes(sizes)

    def reset_right(self) -> None:
        widget = self.main_split.widget(2)
        widget.setHidden(False)
        sizes = list(self.main_split.sizes())
        if len(sizes) > 2:
            sizes[2] = 300
            self.main_split.setSizes(sizes)

    def on_skills(self) -> None:
        popup = SkillsPopup(self)
        pos = self.skills_btn.mapToGlobal(self.skills_btn.rect().topRight())
        popup.move(pos.x() - popup.width() + self.skills_btn.width(), pos.y())
        popup.show()

    def on_settings(self) -> None:
        self._open_settings(SettingsDialog.PAGE_MODELS)

    def on_add_model(self) -> None:
        self._open_settings(SettingsDialog.PAGE_MODELS)

    def _open_settings(self, page: int) -> None:
        dialog = SettingsDialog(ModelStore(), self, page=page, agent=self.agent)
        dialog.models_changed.connect(self._on_models_changed)
        dialog.config_changed.connect(self._on_config_changed)
        dialog.exec_()
        self._on_models_changed()

    def _on_models_changed(self) -> None:
        entry = ModelStore().get_active()
        if entry is None:
            self.model_btn.setText("选择模型")
            return
        self.model_btn.setText(entry.get("label", entry.get("model", "")))
        if self.agent is not None:
            self.agent.set_llm(ModelStore.to_llm_config(entry))

    def _on_config_changed(self) -> None:
        if self.agent is None:
            return
        try:
            from imgsearch_agent.config import AgentConfig
            from imgsearch_agent.utils.path_guard import PathGuard

            cfg = AgentConfig()
            self.agent.config.memory_max_sessions = cfg.memory_max_sessions
            self.agent.config.similarity_threshold = cfg.similarity_threshold
            self.agent.config.max_retries = cfg.max_retries
            self.agent.config.allowed_roots = list(cfg.allowed_roots)
            self.agent.critic.threshold = cfg.similarity_threshold
            self.agent.critic.max_retries = cfg.max_retries

            # 安全根目录热更新：所有带 guard 的工具换成同一份新规则
            guard = PathGuard(cfg.allowed_roots or None)
            for tool in self.agent.tools.values():
                if hasattr(tool, "guard"):
                    tool.guard = guard

            # 检索参数热更新：只改已创建的 Pipeline，未创建的等首次检索读新配置
            try:
                from imgsearch_agent.retriever import Config as RetrieverConfig

                rcfg = RetrieverConfig.from_yaml()
                tool = self.agent.tools.get("image_retrieval")
                pipeline = getattr(tool, "_pipeline", None) if tool else None
                if pipeline is not None:
                    pipeline.cfg.search.top_k = rcfg.search.top_k
                    pipeline.cfg.search.recall_k = rcfg.search.recall_k
                    pipeline.cfg.rerank.enabled = rcfg.rerank.enabled
                    pipeline.cfg.caption.enabled = rcfg.caption.enabled
            except Exception:
                pass

            self._refresh_sessions()
            self.append_log(
                f">>> 设置已更新：保留会话 {cfg.memory_max_sessions} 个，"
                f"允许目录 {len(cfg.allowed_roots) or '默认'} 个"
            )
        except Exception as e:
            self.append_log("应用设置失败: " + str(e))

    # ── 任务操作 ──────────────────────────────────────────────────────────

    def _add_task_item(self, task: TaskSpec) -> None:
        if task.is_dynamic:
            name = str((task.dynamic_spec or {}).get("name", "动态工具"))
            label = f"🔧 {name}｜自定义临时工具｜需要审核"
        elif task.action == "scan":
            label = "📁 打开 " + (task.path or task.description)
        else:
            label = task.query or task.description
        item = QListWidgetItem(label)
        item.setFlags(
            item.flags() | Qt.ItemIsEditable | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled
        )
        item.setCheckState(Qt.Checked if task.status != "skipped" else Qt.Unchecked)
        item.setData(Qt.UserRole, task.model_dump())
        item.setToolTip(task.description or task.query or task.path)
        self.task_list.addItem(item)

    def on_add_task(self) -> None:
        self._add_task_item(TaskSpec(id=self.task_list.count() + 1, description="", query=""))

    def on_del_task(self) -> None:
        row = self.task_list.currentRow()
        if row >= 0:
            self.task_list.takeItem(row)

    def _move_task(self, delta: int) -> None:
        row = self.task_list.currentRow()
        target = row + delta
        if row < 0 or target < 0 or target >= self.task_list.count():
            return
        item = self.task_list.takeItem(row)
        self.task_list.insertItem(target, item)
        self.task_list.setCurrentRow(target)

    def _collect_tasks(self) -> list[TaskSpec]:
        tasks: list[TaskSpec] = []
        for i in range(self.task_list.count()):
            item = self.task_list.item(i)
            meta = item.data(Qt.UserRole) or {}
            try:
                task = TaskSpec(**meta)
            except Exception:
                task = TaskSpec(id=i + 1, description=item.text(), query=item.text())

            text = item.text().strip()
            if task.action != "scan" and text and text != task.query:
                task.query = text

            task.id = i + 1
            task.status = "pending" if item.checkState() == Qt.Checked else "skipped"
            tasks.append(task)
        return tasks

    # ── Agent 调用 ────────────────────────────────────────────────────────

    def _get_agent(self):
        if self.agent is None:
            from imgsearch_agent.agent import ImageSearchAgent
            from imgsearch_agent.config import AgentConfig

            self.agent = ImageSearchAgent(config=AgentConfig())
            entry = ModelStore().get_active()
            if entry:
                self.agent.set_llm(ModelStore.to_llm_config(entry))
                self.model_btn.setText(entry.get("label", entry.get("model", "")))
        return self.agent

    def _set_busy(self, busy: bool) -> None:
        for b in (
            self.send_btn,
            self.run_btn,
            self.add_btn,
            self.del_btn,
            self.up_btn,
            self.down_btn,
        ):
            b.setEnabled(not busy)
        # 设置与停止在任务执行期间仍然可用
        self.settings_btn.setEnabled(True)
        if hasattr(self, "stop_btn"):
            self.stop_btn.setEnabled(busy)
        self.input.setEnabled(not busy)

    # ── 附件与以图搜图 ────────────────────────────────────────────────────

    def on_attach(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            str(Path(__file__).resolve().parent / "datasets"),
            "图片 (*.jpg *.jpeg *.png *.webp *.bmp *.gif)",
        )
        if path:
            self._set_attachment(path)

    def _set_attachment(self, path: str) -> None:
        if not Path(path).exists():
            return
        self._attachment = path
        thumb = self._thumb_path(path, 40)
        if thumb:
            self.attach_thumb.setPixmap(QPixmap(thumb))
        else:
            self.attach_thumb.setText("图")
        self.attach_name.setText(Path(path).name)
        self.attach_bar.setVisible(True)
        self.input.setPlaceholderText("可以补充描述，或直接发送按图搜索")
        self.append_log(">>> 已附加图片：" + Path(path).name)

    def _clear_attachment(self) -> None:
        self._attachment = ""
        self.attach_bar.setVisible(False)
        self.attach_thumb.clear()
        self.attach_name.setText("")
        self.input.setPlaceholderText(
            "描述你想找的图片，支持多行。Ctrl+Enter 发送，例如：\n"
            "找两张猫的图片，再找几张海边的日落"
        )

    def _run_image_turn(self, image_path: str, text: str):
        agent = self._get_agent()
        agent.memory.add("user", text or "找相似的图片")

        tool = agent.tools.get("image_search_by_image")
        if tool is None:
            return [], [], "没有注册以图搜图工具", []

        result, _ = agent.executor.execute(
            tool, {"image_path": image_path, "top_k": 12}
        )
        reply = agent.responder.compose(text or "找相似的图片", result)
        images = result.get("images", [])

        image_paths = [
            (img.get("path", "") if isinstance(img, dict) else getattr(img, "path", ""))
            for img in images
        ]
        image_paths = [p for p in image_paths if p]
        agent.memory.add("assistant", reply, images=image_paths)
        return [], [], reply, images

    # ── 联网模式 ──────────────────────────────────────────────────────────

    _MODE_LABELS = {
        "local": ("💻", "仅本地图库"),
        "hybrid": ("🌐💻", "先搜本地，不够再联网"),
        "online": ("🌐", "仅联网下载"),
    }
    _MODE_ORDER = ("local", "hybrid", "online")

    def _refresh_mode_btn(self) -> None:
        text, tip = self._MODE_LABELS[self._search_mode]
        self.mode_btn.setText(text)
        self.mode_btn.setToolTip(f"当前：{tip}（点击切换）")

    def _cycle_mode(self) -> None:
        index = self._MODE_ORDER.index(self._search_mode)
        self._search_mode = self._MODE_ORDER[(index + 1) % len(self._MODE_ORDER)]
        self._refresh_mode_btn()
        self.append_log(f">>> 检索模式切换为：{self._MODE_LABELS[self._search_mode][1]}")

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

    def _run_turn(self, text: str):
        agent = self._get_agent()
        agent.memory.add("user", text)

        # 用户没在话里指定路径时，用文件区当前打开的目录
        default_scope = ""
        try:
            if self.tree_root and self.tree_root.resolve() != ROOT.resolve():
                default_scope = str(self.tree_root)
        except Exception:
            default_scope = ""

        tasks = agent.plan_tasks(text)
        if default_scope:
            for t in tasks:
                if t.action == "retrieve" and not t.path:
                    t.path = default_scope
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

        # 本地不够时按所选模式联网补充
        if self._search_mode in ("hybrid", "online") and not self._cancel:
            found = sum(len(r.results) for r in results)
            need = sum(t.top_k for t in tasks if t.top_k > 0) or 0
            short = need - found

            if self._search_mode == "online" or (need > 0 and short > 0):
                tool = agent.tools.get("web_image_search")
                if tool is not None:
                    query = tasks[0].query if tasks else text
                    count = max(short, 10) if self._search_mode == "hybrid" else max(need, 20)
                    web = tool.run(query=query, top_k=min(count, 100))
                    self.append_log(f">>> 联网补充：{web.get('summary', '')}")
        replies = [r.reply for r in results if r.reply]
        combined = "\n".join(replies) or "没有找到相关图片。"
        images = [img for r in results for img in r.results]
        image_paths = [
            (img.get("path", "") if isinstance(img, dict) else getattr(img, "path", ""))
            for img in images
        ]
        image_paths = [p for p in image_paths if p]
        agent.memory.add("assistant", combined, images=image_paths)
        return tasks, results, combined, images

    def on_turn_done(self, payload) -> None:
        tasks, results, combined, images = payload
        self.conversation.append(("Agent", combined))
        self._append_chat("Agent", combined, images)
        self._render_memory()

        self.task_list.clear()
        for t in tasks:
            self._add_task_item(t)

        self.last_results = images
        self.state_label.setText(
            f"完成 {len(results)} 条任务，共 {len(images)} 张图片"
        )
        self._refresh_sessions()

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
                from imgsearch_agent.agent.recommender import score_dynamic_tool

                traces = agent.memory.list_traces(name)
                recommendation = score_dynamic_tool(spec, traces)
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
                    from imgsearch_agent.agent.custom_tools import save_custom_tool

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

    def _run_existing_tasks(self, tasks: list[TaskSpec]):
        agent = self._get_agent()
        results = agent.run_tasks(
            tasks, write_memory=False, should_stop=lambda: self._cancel
        )

        # 本地不够时按所选模式联网补充
        if self._search_mode in ("hybrid", "online") and not self._cancel:
            found = sum(len(r.results) for r in results)
            need = sum(t.top_k for t in tasks if t.top_k > 0) or 0
            short = need - found

            if self._search_mode == "online" or (need > 0 and short > 0):
                tool = agent.tools.get("web_image_search")
                if tool is not None:
                    query = tasks[0].query if tasks else text
                    count = max(short, 10) if self._search_mode == "hybrid" else max(need, 20)
                    web = tool.run(query=query, top_k=min(count, 100))
                    self.append_log(f">>> 联网补充：{web.get('summary', '')}")
        replies = [r.reply for r in results if r.reply]
        combined = "\n".join(replies) or "没有找到相关图片。"
        images = [img for r in results for img in r.results]
        image_paths = [
            (img.get("path", "") if isinstance(img, dict) else getattr(img, "path", ""))
            for img in images
        ]
        image_paths = [p for p in image_paths if p]
        agent.memory.add("assistant", combined, images=image_paths)
        return tasks, results, combined, images

    def on_stop(self) -> None:
        self._cancel = True
        self.state_label.setText("正在停止，当前这条检索完成后中断…")

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

    def closeEvent(self, event) -> None:
        """关闭窗口时释放向量库文件锁，便于下次启动。"""
        try:
            if self.agent is not None:
                self.agent.close()
        except Exception:
            pass
        super().closeEvent(event)

    # ── 文件树 ────────────────────────────────────────────────────────────

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
            path = img.get("path", "") if isinstance(img, dict) else getattr(img, "path", "")
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


def _setup_file_logging() -> None:
    try:
        log_dir = ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        root = logging.getLogger()
        root.addHandler(handler)
        if root.level > logging.INFO:
            root.setLevel(logging.INFO)
    except Exception:
        pass


def main() -> int:
    _setup_file_logging()
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("imgsearch Agent")
    app.setStyleSheet(GLOBAL_QSS)

    from PyQt5.QtCore import QDir, QLockFile
    from PyQt5.QtWidgets import QMessageBox

    lock = QLockFile(QDir.temp().filePath("imgsearch_agent.lock"))
    if not lock.tryLock(100):
        QMessageBox.warning(
            None,
            "已有实例在运行",
            "imgsearch Agent 已经打开了一个窗口。\n"
            "Qdrant 本地模式只允许单进程访问，请先关闭已有窗口。",
        )
        return 1

    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
