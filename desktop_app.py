"""Findgo 桌面应用（PyQt5）。

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

from findgo.model_store import ModelStore
from findgo.settings_dialog import SettingsDialog
from findgo.theme import GLOBAL_QSS
from findgo.ui.agent_runner import AgentRunnerMixin
from findgo.ui.bubble import BubbleLabel, ImageGalleryDialog
from findgo.ui.chat_render import highlight_paths, message_html
from findgo.ui.command_panel import (
    CommandMenu,
    PermissionButton,
    SlashPopup,
    command_by_name,
    match_commands,
    popup_position,
)
from findgo.ui.components import LogBridge, QtLogHandler, Worker
from findgo.ui.dynamic_dialog import DynamicToolDialog
from findgo.ui.file_tree import FileTreeMixin
from findgo.ui.index_progress import IndexProgressPanel
from findgo.ui.input_panel import ChatInput, ResizeHandle
from findgo.ui.session_panel import SessionMixin
from findgo.ui.skills_panel import SkillDropZone, SkillsPopup
from findgo.ui.thumbnail_cache import thumb_path
from findgo.ui.title_bar import TitleBar
from findgo.ui.trace_panel import TraceDialog
from findgo.utils.schemas import TaskSpec
from PyQt5.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QRect,
    QSettings,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
)
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
    QScrollArea,
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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


# ── 日志桥接 ──────────────────────────────────────────────────────────────────


class MainWindow(AgentRunnerMixin, SessionMixin, FileTreeMixin, QMainWindow):
    # 工具线程里的建库进度，通过信号转回 UI 线程
    index_progress = pyqtSignal(int, int, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Findgo · 多模态图像检索")
        self.setMinimumSize(760, 560)

        if not self._restore_geometry():
            screen = QApplication.primaryScreen()
            if screen is not None:
                area = screen.availableGeometry()
                width = max(min(1200, int(area.width() * 0.82)), 900)
                height = max(min(760, int(area.height() * 0.82)), 600)
                self.resize(width, height)
                self.move(
                    area.x() + (area.width() - width) // 2,
                    area.y() + (area.height() - height) // 2,
                )
            else:
                self.resize(1100, 700)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self._left_width = 240
        self._right_width = 225
        self.setStyleSheet(
            "QMainWindow{background:#FBFAF7;}"
            "QScrollBar:vertical{background:transparent;width:8px;}"
            "QScrollBar::handle:vertical{background:#DEDBD3;border-radius:4px;min-height:24px;}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}"
        )

        self.worker: Worker | None = None
        self._cancel = False
        self._index_cancel = False
        self.agent = None
        self.conversation: list[tuple[str, str]] = []
        self.last_results: list[dict] = []
        self._attachment: str = ""
        self._galleries: dict[str, list] = {}
        self.tree_root: Path = ROOT
        self.results_mode = False

        self._build_ui()
        self._setup_logging()
        self.index_progress.connect(self._on_index_progress)
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

        self.chat_view = QScrollArea()
        self.chat_view.setWidgetResizable(True)
        self.chat_view.setStyleSheet(
            "QScrollArea{border:1px solid #EDEBE6;border-radius:12px;" "background:#FFFFFF;}"
        )

        self.chat_container = QWidget()
        self.chat_container.setStyleSheet("background:#FFFFFF;")
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(16, 16, 16, 16)
        self.chat_layout.setSpacing(0)
        self.chat_layout.addStretch(1)
        self.chat_view.setWidget(self.chat_container)

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

        # 建库进度条（默认隐藏）
        self.index_panel = IndexProgressPanel()
        self.index_panel.cancel_requested.connect(self._cancel_index)
        layout.addWidget(self.index_panel)

        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        self.input = ChatInput()
        self.input.image_pasted.connect(self._set_attachment)
        self.input.send_requested.connect(self.on_send)
        self.input.setPlaceholderText(
            "描述你想找的图片。Enter 发送，Shift+Enter 换行，例如：\n找两张猫的图片，再找几张海边的日落"
        )
        self.input.setStyleSheet(
            "QPlainTextEdit{border:1px solid #EDEBE6;border-radius:18px;"
            "padding:14px 18px 48px 18px;font-size:14px;background:#FFFFFF;}"
            "QPlainTextEdit:focus{border:1px solid #A8C9B8;}"
        )
        self.input.textChanged.connect(self._adjust_input_height)
        self.input.textChanged.connect(self._on_input_changed)
        self.input.nav_key.connect(self._on_input_nav_key)
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

        # 命令菜单：四角星，点击弹出，也响应输入框里的 /
        self.command_btn = QToolButton()
        self.command_btn.setText("✦")
        self.command_btn.setToolTip("命令菜单（输入 / 也能唤出）")
        self.command_btn.setFixedSize(28, 28)
        self.command_btn.setCursor(Qt.PointingHandCursor)
        self.command_btn.setStyleSheet(
            "QToolButton{border:1px solid #EDEBE6;border-radius:14px;"
            "background:#FFFFFF;font-size:14px;color:#6F9C86;}"
            "QToolButton:hover{background:#E3EFE7;}"
        )
        self.command_btn.clicked.connect(self.on_command_menu)
        left_row.addWidget(self.command_btn)

        # 权限切换：按钮文字就是当前等级符号
        self.perm_btn = PermissionButton(self._load_permission())
        self.perm_btn.changed.connect(self._on_permission_changed)
        left_row.addWidget(self.perm_btn)

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
        # 顶部内边距 14 + 底部按钮区 48，再加两行文本高度
        content = rect.height() + 14 + 48 + 12
        window_h = self.height() or 880
        max_h = max(170, int(window_h * 0.28))
        # 最小 112：能完整显示两行提示语，不压到发送按钮
        target = int(min(max(content, 112), max_h))

        if target != self.input.height():
            self.input.setFixedHeight(target)

        # 还没到最大高度时，保持从第一行开始显示；
        # 超过上限后才交给滚动条跟随光标
        if target < max_h:
            self.input.verticalScrollBar().setValue(0)

    # ── 命令菜单与斜杠候选 ────────────────────────────────────────────────

    def on_command_menu(self) -> None:
        menu = CommandMenu(self)
        menu.command_picked.connect(self._apply_command)
        menu.exec_(self.command_btn.mapToGlobal(QPoint(0, self.command_btn.height() + 4)))

    def _on_input_changed(self) -> None:
        text = self.input.toPlainText()
        # 只在整段以 / 开头、且还没输入空格时弹候选
        if text.startswith("/") and "\n" not in text:
            query_part = text.split(" ")[0]
            if " " in text:
                self._hide_slash_popup()
                return
            items = match_commands(query_part)
            if items:
                self._show_slash_popup(items)
                return
        self._hide_slash_popup()

    def _show_slash_popup(self, items: list[dict]) -> None:
        if not hasattr(self, "slash_popup"):
            self.slash_popup = SlashPopup(self)
            self.slash_popup.command_picked.connect(self._apply_command)

        self.slash_popup.fill(items)
        self.slash_popup.resize(420, min(len(items) * 34 + 14, 6 * 34 + 14))

        cursor = self.input.textCursor()
        rect = self.input.cursorRect(cursor)
        self.slash_popup.move(popup_position(self.input, rect))
        self.slash_popup.show()
        self.slash_popup.raise_()
        self.input.popup_open = True

    def _hide_slash_popup(self) -> None:
        if hasattr(self, "slash_popup") and self.slash_popup.isVisible():
            self.slash_popup.hide()
        if hasattr(self, "input"):
            self.input.popup_open = False

    def _slash_popup_visible(self) -> bool:
        return hasattr(self, "slash_popup") and self.slash_popup.isVisible()

    def _on_input_nav_key(self, key: int) -> None:
        """方向键与回车在候选列表里导航，Esc 关闭。"""
        if not self._slash_popup_visible():
            return

        if key == Qt.Key_Down:
            self.slash_popup.move_selection(1)
        elif key == Qt.Key_Up:
            self.slash_popup.move_selection(-1)
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            name = self.slash_popup.current_command()
            if name:
                self._apply_command(name)
        elif key == Qt.Key_Escape:
            self._hide_slash_popup()

    def _apply_command(self, name: str) -> None:
        """选中命令后直接执行，不往输入框填提示词。"""
        self._hide_slash_popup()
        self.input.setPlainText(name)
        self.on_send()

    def _on_permission_changed(self, key: str) -> None:
        label = {
            "full": "完整权限",
            "auto": "自动审批",
            "ask": "操作前询问",
            "readonly": "只读模式",
        }.get(key, key)
        self._save_permission(key)
        self.append_log(f">>> 权限切换为：{label}")
        self.state_label.setText(f"当前权限：{label}")

    @staticmethod
    def _permission_file() -> Path:
        return ROOT / "data" / "ui_state.json"

    def _load_permission(self) -> str:
        try:
            import json

            data = json.loads(self._permission_file().read_text(encoding="utf-8"))
            key = str(data.get("permission", "ask"))
            return key if key in {"full", "auto", "ask", "readonly"} else "ask"
        except Exception:
            return "ask"

    def _save_permission(self, key: str) -> None:
        try:
            import json

            path = self._permission_file()
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
            data["permission"] = key
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ── 建库 ──────────────────────────────────────────────────────────────

    def _on_tree_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        path = item.data(0, Qt.UserRole) if item is not None else str(self.tree_root)
        if not path:
            return

        target = Path(path)
        if not target.is_dir():
            target = target.parent

        menu = QMenu(self)
        act_index = menu.addAction("建立索引（增量）")
        act_index.triggered.connect(lambda: self._start_index(target, rebuild=False))
        act_rebuild = menu.addAction("全量重建索引")
        act_rebuild.triggered.connect(lambda: self._start_index(target, rebuild=True))
        menu.addSeparator()
        act_open = menu.addAction("在资源管理器中打开")
        act_open.triggered.connect(lambda: self._open_in_explorer(target))

        menu.exec_(self.tree.mapToGlobal(pos))

    @staticmethod
    def _open_in_explorer(path: Path) -> None:
        try:
            from PyQt5.QtCore import QUrl
            from PyQt5.QtGui import QDesktopServices

            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        except Exception:
            pass

    def _cancel_index(self) -> None:
        self._index_cancel = True
        self.append_log(">>> 已请求停止建索引，当前批次完成后退出")

    def _start_index(self, folder: Path, rebuild: bool = False) -> None:
        if self.worker is not None:
            self.state_label.setText("还有任务在跑，等它完成再建索引。")
            return

        self._index_cancel = False
        self.index_panel.start(f"正在建索引：{folder.name}")
        self.append_log(f">>> 开始建索引：{folder}（{'全量' if rebuild else '增量'}）")

        self.worker = Worker(self._run_index, folder, rebuild)
        self.worker.progress.connect(self._on_index_progress)
        self.worker.done.connect(self._on_index_done)
        self.worker.failed.connect(self._on_index_failed)
        self.worker.finished.connect(self._finish_worker)
        self._set_busy(True)
        self.worker.start()

    def _run_index(self, folder: Path, rebuild: bool):
        agent = self._get_agent()
        tool = agent.tools.get("index_folder")
        if tool is None:
            raise RuntimeError("index_folder 工具未注册")

        def report(done: int, total: int, name: str) -> None:
            worker = self.worker
            if worker is not None:
                try:
                    worker.progress.emit((done, total, name))
                except Exception:
                    pass

        return tool.run(
            folder_path=str(folder),
            rebuild=rebuild,
            should_stop=lambda: self._index_cancel,
            progress=report,
        )

    def _index_progress_from_tool(self, done: int, total: int, name: str) -> None:
        """从工具线程回调，转到 UI 线程更新进度条。"""
        try:
            self.index_progress.emit(int(done), int(total), str(name))
        except Exception:
            pass

    def _on_index_progress(self, payload) -> None:
        try:
            done, total, name = payload
        except Exception:
            return
        self.index_panel.update_progress(int(done), int(total), str(name))

    def _on_index_done(self, result) -> None:
        if not isinstance(result, dict):
            self.index_panel.fail("索引返回格式异常")
            return

        summary = str(result.get("summary", ""))
        if result.get("cancelled"):
            self.index_panel.fail("已取消")
        else:
            self.index_panel.finish(summary)

        self.append_log(f">>> {summary}")
        self.state_label.setText(summary or "索引完成")
        self._refresh_results()

    def _on_index_failed(self, error: str) -> None:
        self.index_panel.fail(error)
        self.append_log("!!! 建索引失败：" + error)

    # ── 执行轨迹 ──────────────────────────────────────────────────────────

    def on_show_traces(self) -> None:
        try:
            agent = self._get_agent()
            traces = agent.memory.session_traces(limit=200)
        except Exception as e:
            self.append_log("读取执行轨迹失败: " + str(e))
            return

        dialog = TraceDialog(traces, self)
        dialog.exec_()

    def _layout_handles(self) -> None:
        if not hasattr(self, "_resize_handles"):
            return
        # 正在拖动手柄时不要重排，否则鼠标会脱离手柄导致抖动
        if getattr(self, "_resizing", False):
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
        # 只在侧栏收起（中间宽度锁定）时重算填充块，正常缩放交给布局自适应，
        # 否则每次 resize 都改分割器会让窗口高度抖动
        if getattr(self, "_locked_center", None):
            self._balance_spacer()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_round_mask()

        # 拖拽手柄位置跟着窗口变，但拖动过程中重排会让鼠标脱离手柄
        if not hasattr(self, "_handle_timer"):
            self._handle_timer = QTimer(self)
            self._handle_timer.setSingleShot(True)
            self._handle_timer.timeout.connect(self._layout_handles)
        self._handle_timer.start(40)

        # 只在侧栏收起（中间宽度锁定）时重算填充块，且防抖避免连续触发
        if getattr(self, "_locked_center", None):
            if not hasattr(self, "_balance_timer"):
                self._balance_timer = QTimer(self)
                self._balance_timer.setSingleShot(True)
                self._balance_timer.timeout.connect(self._balance_spacer)
            self._balance_timer.start(80)

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

        # 执行轨迹回放：只看不跑
        self.trace_btn = QPushButton("≡")
        self.trace_btn.setFixedWidth(30)
        self.trace_btn.setStyleSheet(
            "QPushButton{border:1px solid #EDEBE6;border-radius:6px;"
            "background:#FFFFFF;color:#4A4A4A;font-size:13px;padding:4px 0;}"
            "QPushButton:hover{background:#F2F0EA;}"
        )
        self.trace_btn.setToolTip("查看本会话的执行轨迹（工具调用、Critic 判定、改写）")
        self.trace_btn.clicked.connect(self.on_show_traces)

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
        btn_row.addWidget(self.trace_btn)
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
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_tree_menu)

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
        right.setSizes([400, 330])
        right.setChildrenCollapsible(False)

        # 三列之间用水平分割器，左右宽度可以自由拖动
        main = QSplitter(Qt.Horizontal)

        # 左侧占位：收起左栏后用它撑住空白，中间和右栏不会贴到窗口左边
        self.left_spacer = QWidget()
        self.left_spacer.setStyleSheet("background:#FBFAF7;")
        self.left_spacer.setMinimumWidth(0)

        main.addWidget(self.left_spacer)
        main.addWidget(left)
        main.addWidget(center)
        main.addWidget(right)

        # 末尾的填充块：窗口变宽或收起侧栏时吸收多余空间，
        # 中间对话区保持用户拖出来的宽度，不会自动撑开
        self.spacer = QWidget()
        self.spacer.setStyleSheet("background:#FBFAF7;")
        self.spacer.setMinimumWidth(0)
        main.addWidget(self.spacer)

        main.setSizes([0, 240, 900, 225, 0])
        main.setStretchFactor(0, 0)  # 左侧占位
        main.setStretchFactor(1, 0)
        main.setStretchFactor(2, 1)  # 中间跟随窗口自适应
        main.setStretchFactor(3, 0)
        main.setStretchFactor(4, 0)  # 右侧填充块
        main.setChildrenCollapsible(False)

        # 分割条只保留可拖区域，视觉上隐藏
        main.setHandleWidth(5)
        main.setStyleSheet(
            "QSplitter::handle{background:transparent;}"
            "QSplitter::handle:hover{background:#E3EFE7;}"
        )

        left.setMinimumWidth(160)
        center.setMinimumWidth(320)
        right.setMinimumWidth(200)

        self.main_split = main
        self.title_bar = TitleBar(self)

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(8, 0, 8, 0)
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
        container.setStyleSheet("#rootFrame{background:#FBFAF7; border-radius:14px;}")
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
        root = logging.getLogger("findgo")
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
            rows.append(f'<p style="margin:5px 0;color:{color};"><b>{who}：</b>{safe}</p>')
        self.memory_view.setHtml("".join(rows) or "<p style='color:#A8A8A8'>暂无对话</p>")
        bar = self.memory_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    # ── 聊天记录 ──────────────────────────────────────────────────────────

    def _thumb_path(self, path: str, size: int = 112) -> str:
        return thumb_path(path, size)

    def _highlight_paths(self, text: str) -> str:
        return highlight_paths(text)

    def _message_html(self, text: str, images: list | None = None) -> str:
        return message_html(text, images, self._galleries, self._thumb_path)

    def _chat_html(self, role: str, text: str, images: list | None = None) -> str:
        safe = self._highlight_paths(text).replace("\n", "<br/>")
        if role == "你":
            return (
                '<div style="margin:28px 0 10px 0;text-align:right;">'
                '<span style="display:inline-block;background:#E3EFE7;color:#2E2E2E;'
                "padding:18px 24px;border-radius:16px;width:640px;"
                'text-align:left;font-size:13px;line-height:1.65;">' + safe + "</span></div>"
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
                    '<div style="margin-top:10px;margin-bottom:6px;">' + "".join(parts) + "</div>"
                )
                rest = len(images) - shown
                if rest > 0:
                    gid = f"g{len(self._galleries)}"
                    self._galleries[gid] = list(images)
                    thumbs += (
                        '<div style="text-align:right;margin-top:4px;">'
                        '<a href="gallery:' + gid + '" style="color:#4E7A63;'
                        'text-decoration:none;font-size:15px;font-weight:600;">…</a></div>'
                    )

        if thumbs:
            # 有图片时用 table 布局：Qt 对 inline-block 内嵌块级元素的高度计算不可靠
            return (
                '<div style="margin:18px 0;text-align:left;">'
                '<table cellspacing="0" cellpadding="0">'
                '<tr><td style="background:#F5F3EE;color:#2E2E2E;'
                "padding:18px 24px;border-radius:16px;"
                'font-size:13px;line-height:1.65;">' + safe + thumbs + "</td></tr></table></div>"
            )

        return (
            '<div style="margin:10px 0;text-align:left;">'
            '<span style="display:inline-block;background:#F5F3EE;color:#2E2E2E;'
            "padding:18px 24px;border-radius:16px;width:640px;"
            'font-size:13px;line-height:1.65;">' + safe + "</span></div>"
        )

    def _append_chat(self, role: str, text: str, images: list | None = None) -> None:
        row = QWidget()
        row.setStyleSheet("background:transparent;")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(0)

        bubble = BubbleLabel(role, text, images, self, row)
        if role == "你":
            layout.addStretch(1)
            layout.addWidget(bubble)
        else:
            layout.addWidget(bubble)
            layout.addStretch(1)

        self.chat_layout.insertWidget(self.chat_layout.count() - 1, row)
        QTimer.singleShot(
            0,
            lambda: self.chat_view.verticalScrollBar().setValue(
                self.chat_view.verticalScrollBar().maximum()
            ),
        )

    def on_chat_link(self, url) -> None:
        # QLabel 的 linkActivated 传字符串，QTextBrowser 的 anchorClicked 传 QUrl
        from PyQt5.QtCore import QUrl

        if not isinstance(url, QUrl):
            url = QUrl(str(url))

        if url.scheme() == "gallery":
            paths = self._galleries.get(url.path(), [])
            if paths:
                ImageGalleryDialog(paths, self).exec_()
            return

        path = url.toLocalFile()
        if path and Path(path).suffix.lower() in IMAGE_EXTS and Path(path).exists():
            self._open_image_dialog(path)

    def _open_image_dialog(self, path: str) -> None:
        from PyQt5.QtWidgets import QLabel, QVBoxLayout

        dialog = QDialog(self)
        dialog.setWindowTitle(Path(path).name)
        dialog.resize(960, 720)
        dialog.setStyleSheet(GLOBAL_QSS)
        layout = QVBoxLayout(dialog)
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            label.setPixmap(pixmap.scaled(920, 680, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(label)
        dialog.exec_()

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

    # ── 会话管理 ──────────────────────────────────────────────────────────

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

    @staticmethod
    def _state_file() -> Path:
        return Path(__file__).resolve().parent / "data" / "window_state.json"

    def _restore_geometry(self) -> bool:
        try:
            import json

            data = json.loads(self._state_file().read_text(encoding="utf-8"))
            x, y, w, h = (int(v) for v in data["rect"])
            if w < 400 or h < 300:
                return False
            self.setGeometry(x, y, w, h)
            return True
        except Exception:
            return False

    def _save_geometry(self) -> None:
        try:
            import json

            g = self.geometry()
            target = self._state_file()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps({"rect": [g.x(), g.y(), g.width(), g.height()]}),
                encoding="utf-8",
            )
        except Exception:
            pass

    def toggle_maximize(self) -> None:
        if self.isMaximized():
            self.showNormal()
            self.title_bar.max_btn.setText("□")
        else:
            self.showMaximized()
            self.title_bar.max_btn.setText("❐")
        self._apply_round_mask()

    def toggle_left(self) -> None:
        widget = self.main_split.widget(1)
        sizes = list(self.main_split.sizes())
        if len(sizes) < 5:
            return

        if not widget.isHidden():
            left_w = sizes[1] or self._left_width or 240
            self._left_width = left_w
            widget.setHidden(True)
            # 把左栏宽度让给左侧占位块，中间保持原位
            self.main_split.setSizes([left_w, 0, sizes[2], sizes[3], 0])
        else:
            widget.setHidden(False)
            self.main_split.setSizes([0, self._left_width, sizes[2], sizes[3], 0])

    def toggle_right(self) -> None:
        widget = self.main_split.widget(3)
        sizes = list(self.main_split.sizes())
        if len(sizes) < 5:
            return

        if not widget.isHidden():
            self._right_width = sizes[3] or self._right_width or 225
            widget.setHidden(True)
            self.main_split.setSizes([sizes[0], sizes[1], sizes[2], 0, self._right_width])
        else:
            widget.setHidden(False)
            self.main_split.setSizes([sizes[0], sizes[1], sizes[2], self._right_width, 0])

    def _balance_spacer(self) -> None:
        """侧栏收起时把中间宽度锁住，多余空间交给对应占位块。"""
        if not hasattr(self, "spacer"):
            return
        # setSizes 会触发新的 resizeEvent，必须防止递归
        if getattr(self, "_balancing", False):
            return
        self._balancing = True

        try:
            self._do_balance()
        finally:
            self._balancing = False

    def _do_balance(self) -> None:
        sizes = list(self.main_split.sizes())
        if len(sizes) < 5:
            return

        left = self.main_split.widget(1)
        right = self.main_split.widget(3)

        total = sum(sizes)
        left_w = 0 if left.isHidden() else sizes[1]
        right_w = 0 if right.isHidden() else sizes[3]
        center_w = sizes[2]

        left_spacer = max(0, self._left_width) if left.isHidden() else 0
        right_spacer = max(0, self._right_width) if right.isHidden() else 0
        used = left_spacer + left_w + center_w + right_w + right_spacer
        leftover = max(0, total - used)

        self.main_split.setSizes([left_spacer, left_w, center_w, right_w, leftover])

    def reset_left(self) -> None:
        widget = self.main_split.widget(1)
        widget.setHidden(False)
        sizes = list(self.main_split.sizes())
        if len(sizes) >= 5:
            self.main_split.setSizes([0, 240, sizes[2], sizes[3], 0])

    def reset_right(self) -> None:
        widget = self.main_split.widget(3)
        widget.setHidden(False)
        sizes = list(self.main_split.sizes())
        if len(sizes) >= 5:
            self.main_split.setSizes([sizes[0], sizes[1], sizes[2], 225, 0])

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
            from findgo.config import AgentConfig
            from findgo.utils.path_guard import PathGuard

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
                from findgo.retriever import Config as RetrieverConfig

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

    @staticmethod
    def _root() -> Path:
        return ROOT

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

        result, _ = agent.executor.execute(tool, {"image_path": image_path, "top_k": 12})
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

    def closeEvent(self, event) -> None:
        """关闭窗口时保存尺寸并释放向量库文件锁。"""
        self._save_geometry()
        try:
            if self.agent is not None:
                self.agent.close()
        except Exception:
            pass
        super().closeEvent(event)

    # ── 文件树 ────────────────────────────────────────────────────────────


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
    app.setApplicationName("Findgo")
    app.setStyle("Fusion")

    from PyQt5.QtGui import QColor, QPalette

    palette = app.palette()
    palette.setColor(QPalette.ToolTipBase, QColor("#FFFFFF"))
    palette.setColor(QPalette.ToolTipText, QColor("#2E2E2E"))
    palette.setColor(QPalette.Window, QColor("#FBFAF7"))
    palette.setColor(QPalette.Base, QColor("#FFFFFF"))
    palette.setColor(QPalette.Text, QColor("#2E2E2E"))
    palette.setColor(QPalette.WindowText, QColor("#2E2E2E"))
    app.setPalette(palette)
    app.setStyleSheet(GLOBAL_QSS)

    from PyQt5.QtCore import QDir, QLockFile
    from PyQt5.QtWidgets import QMessageBox

    lock = QLockFile(QDir.temp().filePath("findgo.lock"))
    if not lock.tryLock(100):
        QMessageBox.warning(
            None,
            "已有实例在运行",
            "Findgo 已经打开了一个窗口。\n" "Qdrant 本地模式只允许单进程访问，请先关闭已有窗口。",
        )
        return 1

    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
