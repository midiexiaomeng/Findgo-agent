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

from PyQt5.QtCore import QObject, QPoint, QRect, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("imgsearch Agent · 多模态图像检索")
        self.resize(1440, 880)
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

        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        self.input = QPlainTextEdit()
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

        btn_row.addWidget(self.model_btn)
        btn_row.addWidget(self.send_btn)
        grid.addWidget(overlay, 0, 0, Qt.AlignBottom | Qt.AlignRight)

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
        self.input.setFixedHeight(int(min(max(content, 56), max_h)))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._adjust_input_height()

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

        self.run_btn = QPushButton("重跑")
        self.run_btn.setStyleSheet(
            "QPushButton{background:#6F9C86;color:white;border:none;"
            "border-radius:5px;padding:4px 10px;font-size:12px;}"
            "QPushButton:disabled{background:#A8C9B8;}"
        )
        self.run_btn.setToolTip("按当前编辑后的任务链重新执行")
        self.run_btn.clicked.connect(self.on_run_tasks)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setStyleSheet(
            "QPushButton{background:#C97B6E;color:white;border:none;"
            "border-radius:5px;padding:4px 10px;font-size:12px;}"
            "QPushButton:disabled{background:#E2BCB3;}"
        )
        self.stop_btn.setToolTip("中断后续任务，当前这一条检索会先跑完")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.on_stop)

        btn_row.addStretch(1)
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
        central = QWidget()
        row = QHBoxLayout(central)
        row.setContentsMargins(6, 6, 6, 6)
        row.setSpacing(6)

        left = self._build_left_column()

        center = QSplitter(Qt.Vertical)
        center.addWidget(self._build_chat_panel())
        center.addWidget(self._build_input_panel())
        center.setSizes([560, 190])

        right = QSplitter(Qt.Vertical)
        right.addWidget(self._build_task_panel())
        right.addWidget(self._build_file_panel())
        right.setSizes([400, 380])

        row.addWidget(left, 1)
        row.addWidget(center, 4)
        row.addWidget(right, 1)

        for w in (left, center, right):
            w.setMinimumWidth(0)

        self.setCentralWidget(central)
        self.setStyleSheet("QMainWindow{background:#F2F0EA;}")

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
                'padding:9px 14px;border-radius:14px;max-width:78%;'
                'text-align:left;font-size:13px;">' + safe + "</span></div>"
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
                    'style="margin:5px;border-radius:8px;"></a>'
                )
                shown += 1

            if parts:
                thumbs = (
                    '<div style="margin-top:8px;line-height:0;">'
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

        return (
            '<div style="margin:10px 0;text-align:left;">'
            '<span style="display:inline-block;background:#F5F3EE;color:#2E2E2E;'
            'padding:9px 14px;border-radius:14px;max-width:78%;font-size:13px;">'
            + safe + thumbs + "</span></div>"
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
        self.state_label.setText(
            f"已规划 {len(tasks)} 条任务，正在检索…"
        )

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
        if task.action == "scan":
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

    def on_send(self) -> None:
        text = self.input.toPlainText().strip()
        if not text or self.worker is not None:
            return

        self._cancel = False
        self.conversation.append(("你", text))
        self._append_chat("你", text)
        self._render_memory()
        self.input.clear()
        self._set_busy(True)
        self.state_label.setText("Agent 正在理解需求并检索…")
        self.append_log(">>> 用户：" + text.replace("\n", " ")[:60])

        self.worker = Worker(self._run_turn, text)
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
        results = agent.run_tasks(
            tasks, write_memory=False, should_stop=lambda: self._cancel
        )
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

    def on_run_tasks(self) -> None:
        tasks = self._collect_tasks()
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
