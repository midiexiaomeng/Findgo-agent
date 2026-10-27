"""imgsearch Agent 桌面应用（PyQt5）—— 三列布局。

┌──────┬────────────────────────────┬──────┐
│ 日志 │ 推理输入（多行 / 模型 / 发送）│ 任务 │
│      ├────────────────────────────┤      │
│ 记忆 │ 最终结果（回复 + 图片墙）    │ 文件 │
└──────┴────────────────────────────┴──────┘

左右宽度比为 1 : 4 : 1，左右合计正好是中间的一半。
"""

from __future__ import annotations

import html
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from PyQt5.QtCore import QObject, QPoint, QRect, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QShortcut,
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
THUMB_W, THUMB_H = 180, 135
GRID_COLUMNS = 4
LOADED_ROLE = Qt.UserRole + 1


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

    def __init__(self, fn, *args):
        super().__init__()
        self.fn = fn
        self.args = args

    def run(self) -> None:
        try:
            self.done.emit(self.fn(*self.args))
        except Exception as e:
            self.failed.emit(str(e))


# ── 图片网格 ──────────────────────────────────────────────────────────────────


class ImageGrid(QWidget):
    def __init__(self):
        super().__init__()
        self.grid = QGridLayout(self)
        self.grid.setSpacing(8)
        self.grid.setContentsMargins(8, 8, 8, 8)
        self.grid.setAlignment(Qt.AlignTop)

    def clear(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def set_results(self, results: list) -> None:
        self.clear()
        index = 0
        for item in results:
            if isinstance(item, dict):
                path = item.get("path", "")
                score = float(item.get("score", 0) or 0)
            else:
                path = getattr(item, "path", "")
                score = float(getattr(item, "score", 0) or 0)

            if not path or not Path(path).exists():
                continue
            pixmap = QPixmap(path)
            if pixmap.isNull():
                continue

            label = QLabel()
            label.setPixmap(
                pixmap.scaled(
                    THUMB_W, THUMB_H, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
            label.setAlignment(Qt.AlignCenter)
            label.setFixedSize(THUMB_W + 6, THUMB_H + 6)
            label.setStyleSheet(
                "border:1px solid #e5e7eb; border-radius:6px; background:#fff;"
            )
            label.setToolTip(f"{path}\n相似度 {score:.3f}")
            self.grid.addWidget(label, index // GRID_COLUMNS, index % GRID_COLUMNS)
            index += 1

        if index == 0:
            empty = QLabel("执行任务后，结果会显示在这里")
            empty.setAlignment(Qt.AlignCenter)
            empty.setStyleSheet("color:#9ca3af; padding:30px;")
            self.grid.addWidget(empty, 0, 0, 1, GRID_COLUMNS)


# ── 主窗口 ────────────────────────────────────────────────────────────────────


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("imgsearch Agent · 多模态图像检索")
        self.resize(1440, 880)

        self.worker: Worker | None = None
        self.agent = None
        self.conversation: list[tuple[str, str]] = []
        self.last_results: list[dict] = []
        self.tree_root: Path = ROOT
        self.results_node: QTreeWidgetItem | None = None

        self._build_ui()
        self._setup_logging()
        self._load_file_tree()
        self._refresh_model_button()
        self._adjust_input_height()

    # ── 中间上：推理输入 ──────────────────────────────────────────────────

    def _build_input_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 6)
        layout.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("推理输入")
        title.setStyleSheet("font-weight:600; color:#374151; font-size:13px;")
        header.addWidget(title)
        header.addStretch(1)
        self.settings_btn = QPushButton("设置")
        self.settings_btn.setStyleSheet(
            "QPushButton{border:1px solid #d1d5db;border-radius:6px;"
            "padding:5px 12px;background:#fff;font-size:12px;}"
        )
        self.settings_btn.clicked.connect(self.on_settings)
        header.addWidget(self.settings_btn)
        layout.addLayout(header)

        # 输入框 + 右下角悬浮按钮
        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        self.input = QPlainTextEdit()
        self.input.setPlaceholderText(
            "描述检索需求，支持多行。Ctrl+Enter 发送，例如：\n找出海边日落的图片，再找几张猫的照片"
        )
        self.input.setStyleSheet(
            "QPlainTextEdit{border:1px solid #d1d5db;border-radius:10px;"
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
            "padding:6px 12px;background:#f9fafb;font-size:12px;}"
            "QToolButton:hover{background:#f3f4f6;}"
        )
        self.model_btn.clicked.connect(self.on_model_menu)

        self.send_btn = QPushButton("发送")
        self.send_btn.setStyleSheet(
            "QPushButton{background:#2563eb;color:white;border:none;"
            "border-radius:7px;padding:7px 18px;font-size:13px;}"
            "QPushButton:disabled{background:#93c5fd;}"
        )
        self.send_btn.clicked.connect(self.on_plan)

        btn_row.addWidget(self.model_btn)
        btn_row.addWidget(self.send_btn)
        grid.addWidget(overlay, 0, 0, Qt.AlignBottom | Qt.AlignRight)

        layout.addWidget(container)

        self.state_label = QLabel("输入需求后点「发送」，确认任务链后点「执行任务」。")
        self.state_label.setStyleSheet("font-size:12px; color:#6b7280;")
        self.state_label.setWordWrap(True)
        layout.addWidget(self.state_label)

        QShortcut(QKeySequence("Ctrl+Return"), self.input, activated=self.on_plan)
        return panel

    def _adjust_input_height(self) -> None:
        text = self.input.toPlainText() or " "
        width = max(self.input.viewport().width() - 4, 120)
        rect = self.input.fontMetrics().boundingRect(
            QRect(0, 0, width, 10000), Qt.TextWordWrap, text
        )
        # 上下 padding + 右下按钮留白
        content = rect.height() + 46 + 16
        window_h = self.height() or 880
        max_h = max(110, int(window_h * 0.25))
        self.input.setFixedHeight(int(min(max(content, 56), max_h)))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._adjust_input_height()

    # ── 中间下：最终结果 ──────────────────────────────────────────────────

    def _build_result_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 2, 10, 10)
        layout.setSpacing(6)

        title = QLabel("最终结果")
        title.setStyleSheet("font-weight:600; color:#374151; font-size:13px;")
        layout.addWidget(title)

        self.reply_view = QTextBrowser()
        self.reply_view.setFixedHeight(86)
        self.reply_view.setStyleSheet(
            "QTextBrowser{border:1px solid #e5e7eb; border-radius:8px; "
            "background:#f8fafc; padding:6px; font-size:12.5px;}"
        )
        layout.addWidget(self.reply_view)

        self.result_scroll = QScrollArea()
        self.result_scroll.setWidgetResizable(True)
        self.result_scroll.setStyleSheet(
            "QScrollArea{border:1px solid #e5e7eb; border-radius:8px; background:#fafafa;}"
        )
        self.result_grid = ImageGrid()
        self.result_scroll.setWidget(self.result_grid)
        layout.addWidget(self.result_scroll, 1)
        return panel

    # ── 左侧 ──────────────────────────────────────────────────────────────

    def _build_left_column(self) -> QSplitter:
        split = QSplitter(Qt.Vertical)

        log_panel = QWidget()
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(6, 6, 6, 6)
        log_layout.setSpacing(4)
        log_title = QLabel("运行日志")
        log_title.setStyleSheet("font-weight:600; color:#374151; font-size:12px;")
        log_layout.addWidget(log_title)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(3000)
        font = QFont("Consolas")
        font.setPointSize(8)
        self.log_view.setFont(font)
        self.log_view.setStyleSheet(
            "QPlainTextEdit{border:1px solid #e5e7eb; border-radius:6px; "
            "background:#0f172a; color:#e2e8f0; padding:4px;}"
        )
        log_layout.addWidget(self.log_view, 1)
        split.addWidget(log_panel)

        mem_panel = QWidget()
        mem_layout = QVBoxLayout(mem_panel)
        mem_layout.setContentsMargins(6, 6, 6, 6)
        mem_layout.setSpacing(4)
        mem_title = QLabel("任务记忆")
        mem_title.setStyleSheet("font-weight:600; color:#374151; font-size:12px;")
        mem_layout.addWidget(mem_title)

        self.memory_view = QTextBrowser()
        self.memory_view.setStyleSheet(
            "QTextBrowser{border:1px solid #e5e7eb; border-radius:6px; "
            "background:#ffffff; padding:6px; font-size:11.5px;}"
        )
        mem_layout.addWidget(self.memory_view, 1)
        split.addWidget(mem_panel)

        split.setSizes([400, 380])
        return split

    # ── 右侧 ──────────────────────────────────────────────────────────────

    def _build_task_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        title = QLabel("推理任务链")
        title.setStyleSheet("font-weight:600; color:#374151; font-size:12px;")
        layout.addWidget(title)

        self.task_list = QListWidget()
        self.task_list.setStyleSheet(
            "QListWidget{border:1px solid #e5e7eb; border-radius:6px; "
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

        self.run_btn = QPushButton("执行")
        self.run_btn.setStyleSheet(
            "QPushButton{background:#16a34a;color:white;border:none;"
            "border-radius:5px;padding:4px 10px;font-size:12px;}"
            "QPushButton:disabled{background:#86efac;}"
        )
        self.run_btn.clicked.connect(self.on_run_tasks)
        btn_row.addStretch(1)
        btn_row.addWidget(self.run_btn)
        layout.addLayout(btn_row)
        return panel

    def _build_file_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        title = QLabel("文件与资源")
        title.setStyleSheet("font-weight:600; color:#374151; font-size:12px;")
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
        self.path_label = QLabel(str(ROOT))
        self.path_label.setStyleSheet("font-size:10.5px;color:#6b7280;")
        self.path_label.setToolTip(str(ROOT))
        nav_row.addWidget(self.up_btn)
        nav_row.addWidget(self.path_label, 1)
        layout.addLayout(nav_row)

        split = QSplitter(Qt.Vertical)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setStyleSheet(
            "QTreeWidget{border:1px solid #e5e7eb; border-radius:6px; "
            "background:#ffffff; font-size:11.5px;}"
        )
        self.tree.itemClicked.connect(self.on_file_clicked)
        self.tree.itemExpanded.connect(self._on_tree_expand)

        self.preview = QLabel("点击图片预览")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(110)
        self.preview.setStyleSheet(
            "QLabel{border:1px solid #e5e7eb; border-radius:6px; "
            "background:#fafafa; color:#9ca3af; font-size:11px;}"
        )

        split.addWidget(self.tree)
        split.addWidget(self.preview)
        split.setSizes([300, 150])
        layout.addWidget(split, 1)
        return panel

    # ── 组装 ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        row = QHBoxLayout(central)
        row.setContentsMargins(6, 6, 6, 6)
        row.setSpacing(6)

        left = self._build_left_column()

        center = QSplitter(Qt.Vertical)
        center.addWidget(self._build_input_panel())
        center.addWidget(self._build_result_panel())
        center.setSizes([190, 600])

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
        self.setStyleSheet("QMainWindow{background:#f3f4f6;}")

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
        for who, text in self.conversation[-30:]:
            color = "#2563eb" if who == "你" else "#111827"
            safe = html.escape(text).replace("\n", "<br/>")
            rows.append(
                f'<p style="margin:5px 0;color:{color};"><b>{who}：</b>{safe}</p>'
            )
        self.memory_view.setHtml(
            "".join(rows) or "<p style='color:#9ca3af'>暂无交互记录</p>"
        )
        bar = self.memory_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    # ── 模型选择 ──────────────────────────────────────────────────────────

    def _refresh_model_button(self) -> None:
        store = ModelStore()
        entry = store.get_active()
        if entry:
            self.model_btn.setText(entry.get("label", entry.get("model", "选择模型")))
        else:
            self.model_btn.setText("选择模型")

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

        # 贴近按钮向上弹出，避免跳到屏幕中央
        pos = self.model_btn.mapToGlobal(QPoint(0, self.model_btn.height()))
        menu.exec_(pos)

    def on_select_model(self, model_id: str) -> None:
        store = ModelStore()
        entry = next((m for m in store.models if m.get("id") == model_id), None)
        if entry is None:
            return
        store.set_active(model_id)
        cfg = ModelStore.to_llm_config(entry)
        if self.agent is not None:
            self.agent.set_llm(cfg)
        self.model_btn.setText(entry.get("label", entry.get("model", "")))
        self.append_log(f">>> 切换模型：{entry.get('label')}")

    def on_settings(self) -> None:
        self._open_settings(SettingsDialog.PAGE_MODELS)

    def on_add_model(self) -> None:
        """从模型菜单直接跳到「添加模型」页。"""
        self._open_settings(SettingsDialog.PAGE_MODELS)

    def _open_settings(self, page: int) -> None:
        dialog = SettingsDialog(ModelStore(), self, page=page)
        dialog.models_changed.connect(self._on_models_changed)
        dialog.exec_()
        self._on_models_changed()

    def _on_models_changed(self) -> None:
        store = ModelStore()
        entry = store.get_active()
        if entry is None:
            self.model_btn.setText("选择模型")
            return
        self.model_btn.setText(entry.get("label", entry.get("model", "")))
        if self.agent is not None:
            self.agent.set_llm(ModelStore.to_llm_config(entry))

    # ── 任务操作 ──────────────────────────────────────────────────────────

    def _add_task_item(self, task: TaskSpec) -> None:
        item = QListWidgetItem(task.query or task.description)
        item.setFlags(
            item.flags() | Qt.ItemIsEditable | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled
        )
        item.setCheckState(Qt.Checked if task.status != "skipped" else Qt.Unchecked)
        item.setData(Qt.UserRole, {"id": task.id, "description": task.description})
        item.setToolTip(task.description or task.query)
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
            query = item.text().strip()
            if not query:
                continue
            tasks.append(
                TaskSpec(
                    id=i + 1,
                    description=str(meta.get("description") or query),
                    query=query,
                    status="pending" if item.checkState() == Qt.Checked else "skipped",
                )
            )
        return tasks

    # ── Agent 调用 ────────────────────────────────────────────────────────

    def _get_agent(self):
        if self.agent is None:
            from imgsearch_agent.app import get_agent

            self.agent = get_agent()
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
            self.settings_btn,
        ):
            b.setEnabled(not busy)
        self.input.setEnabled(not busy)

    def on_plan(self) -> None:
        text = self.input.toPlainText().strip()
        if not text or self.worker is not None:
            return
        self.conversation.append(("你", text))
        self._render_memory()
        self._set_busy(True)
        self.state_label.setText("正在拆解任务…")
        self.append_log(f">>> 拆解任务：{text.replace(chr(10), ' ')[:60]}")

        self.worker = Worker(lambda: self._get_agent().plan_tasks(text))
        self.worker.done.connect(self.on_plan_done)
        self.worker.failed.connect(self.on_worker_failed)
        self.worker.finished.connect(self._finish_worker)
        self.worker.start()

    def on_plan_done(self, tasks) -> None:
        self.task_list.clear()
        for t in tasks:
            self._add_task_item(t)
        self.state_label.setText(
            f"已生成 {len(tasks)} 条任务，可编辑、勾选或调整顺序后执行。"
        )
        self.append_log(f">>> 已生成 {len(tasks)} 条任务")

    def on_run_tasks(self) -> None:
        tasks = self._collect_tasks()
        if not tasks or self.worker is not None:
            return
        pending = sum(1 for t in tasks if t.status == "pending")
        if pending == 0:
            self.state_label.setText("没有勾选任何任务。")
            return

        self._set_busy(True)
        self.state_label.setText(f"正在执行 {pending} 条任务…")
        self.append_log(f">>> 开始执行 {pending} 条任务")

        self.worker = Worker(lambda: self._get_agent().run_tasks(tasks))
        self.worker.done.connect(self.on_run_done)
        self.worker.failed.connect(self.on_worker_failed)
        self.worker.finished.connect(self._finish_worker)
        self.worker.start()

    def on_run_done(self, results) -> None:
        self.last_results = []
        replies: list[str] = []
        for r in results:
            self.conversation.append(("Agent", r.reply))
            replies.append(r.reply)
            self.last_results.extend(r.results)
            self.append_log(
                f">>> 任务完成 | 置信度 {r.confidence:.3f} | 重试 {r.retries} | trace {r.trace_id}"
            )

        self.reply_view.setHtml(
            "<br/>".join(html.escape(x) for x in replies)
            or "<span style='color:#9ca3af'>没有返回结果</span>"
        )
        self.result_grid.set_results(self.last_results)
        self.state_label.setText(
            f"完成 {len(results)} 条任务，共 {len(self.last_results)} 张图片。"
        )
        self._render_memory()
        self._refresh_results_node()

    def on_worker_failed(self, error: str) -> None:
        self.append_log(f"!!! 执行失败：{error}")
        self.state_label.setText(f"执行失败：{error}")
        self.conversation.append(("Agent", f"出错了：{error}"))
        self._render_memory()

    def _finish_worker(self) -> None:
        self.worker = None
        self._set_busy(False)
        self.input.setFocus()

    # ── 文件树 ────────────────────────────────────────────────────────────

    def _go_up(self) -> None:
        current = self.tree_root.resolve()
        parent = current.parent
        if parent == current:
            return
        self.tree_root = parent
        self._load_file_tree()

    def _load_file_tree(self) -> None:
        root = self.tree_root
        self.path_label.setText(str(root))
        self.path_label.setToolTip(str(root))
        self.tree.clear()
        self.results_node = None

        root_item = QTreeWidgetItem([root.name or str(root)])
        root_item.setData(0, Qt.UserRole, str(root))
        root_item.setData(0, LOADED_ROLE, True)
        self.tree.addTopLevelItem(root_item)

        if root.resolve() == ROOT.resolve():
            self.results_node = QTreeWidgetItem(["最近检索结果"])
            root_item.addChild(self.results_node)

        self._fill_dir(root_item, root, limit=200)
        root_item.setExpanded(True)

    def _fill_dir(self, parent: QTreeWidgetItem, path: Path, limit: int) -> None:
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))[:limit]
        except OSError:
            return
        for p in entries:
            if p.name in ("__pycache__", ".git", ".pytest_cache"):
                continue
            item = QTreeWidgetItem([p.name])
            item.setData(0, Qt.UserRole, str(p))
            item.setData(0, LOADED_ROLE, False)
            parent.addChild(item)
            if p.is_dir():
                # 占位子项：让 Qt 显示展开箭头，真正展开时再替换
                QTreeWidgetItem(item, ["…"])

    def _on_tree_expand(self, item: QTreeWidgetItem) -> None:
        if item.data(0, LOADED_ROLE):
            return
        path = item.data(0, Qt.UserRole)
        if not path:
            return
        p = Path(path)
        if not p.is_dir():
            return
        item.takeChildren()
        self._fill_dir(item, p, limit=200)
        item.setData(0, LOADED_ROLE, True)

    def _refresh_results_node(self) -> None:
        if self.results_node is None:
            return
        self.results_node.takeChildren()
        for img in self.last_results:
            path = img.get("path", "") if isinstance(img, dict) else getattr(img, "path", "")
            if not path or not Path(path).exists():
                continue
            item = QTreeWidgetItem([Path(path).name])
            item.setData(0, Qt.UserRole, path)
            item.setToolTip(0, path)
            self.results_node.addChild(item)
        self.results_node.setExpanded(True)

    def on_file_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        path = item.data(0, Qt.UserRole)
        if not path:
            return
        p = Path(path)
        if p.suffix.lower() in IMAGE_EXTS and p.exists():
            pixmap = QPixmap(str(p))
            if not pixmap.isNull():
                self.preview.setPixmap(
                    pixmap.scaled(200, 130, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                self.preview.setToolTip(str(p))
                return
        self.preview.setText(p.name)
        self.preview.setPixmap(QPixmap())


def _setup_file_logging() -> None:
    """pythonw 启动时没有控制台，日志落文件便于排查。"""
    try:
        log_dir = ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
            )
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
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
