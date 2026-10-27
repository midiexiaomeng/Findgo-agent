"""imgsearch Agent 桌面应用（PyQt5）—— 四象限布局。

┌────────────────────┬──────────────────────┐
│ ① 运行日志          │ ③ 任务链（可编辑）    │
├────────────────────┼──────────────────────┤
│ ② 记忆 / 上下文     │ ④ 文件与资源          │
└────────────────────┴──────────────────────┘

原生窗口，不占端口、不依赖浏览器。双击 start-app.bat 启动。
"""

from __future__ import annotations

import html
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from imgsearch_agent.utils.schemas import TaskSpec

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


# ── 日志桥接：把 logging 记录转发到界面 ──────────────────────────────────────


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


# ── 主窗口 ────────────────────────────────────────────────────────────────────


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("imgsearch Agent · 多模态图像检索")
        self.resize(1360, 860)

        self.worker: Worker | None = None
        self.agent = None
        self.conversation: list[tuple[str, str]] = []
        self.last_results: list[dict] = []

        self._build_ui()
        self._setup_logging()
        self._load_file_tree()

    # ── ① 日志面板 ────────────────────────────────────────────────────────

    def _build_log_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("① 系统运行日志")
        title.setStyleSheet("font-weight:600; color:#374151; font-size:13px;")
        layout.addWidget(title)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(3000)
        font = QFont("Consolas")
        font.setPointSize(9)
        self.log_view.setFont(font)
        self.log_view.setStyleSheet(
            "QPlainTextEdit{border:1px solid #e5e7eb; border-radius:8px; "
            "background:#0f172a; color:#e2e8f0; padding:6px;}"
        )
        layout.addWidget(self.log_view, 1)
        return panel

    # ── ② 记忆面板 ────────────────────────────────────────────────────────

    def _build_memory_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("② 任务记忆 / 上下文")
        title.setStyleSheet("font-weight:600; color:#374151; font-size:13px;")
        layout.addWidget(title)

        self.memory_view = QTextBrowser()
        self.memory_view.setStyleSheet(
            "QTextBrowser{border:1px solid #e5e7eb; border-radius:8px; "
            "background:#ffffff; padding:8px; font-size:12.5px;}"
        )
        layout.addWidget(self.memory_view, 1)
        return panel

    # ── ③ 任务面板 ────────────────────────────────────────────────────────

    def _build_task_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("③ 推理任务链（可编辑 · 勾选执行 · 拖拽排序）")
        title.setStyleSheet("font-weight:600; color:#374151; font-size:13px;")
        layout.addWidget(title)

        input_row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("描述检索需求，例如：找几张海边的日落和猫")
        self.input.returnPressed.connect(self.on_plan)
        self.plan_btn = QPushButton("拆解任务")
        self.plan_btn.setStyleSheet(
            "QPushButton{background:#2563eb;color:white;border:none;border-radius:6px;padding:7px 14px;}"
            "QPushButton:disabled{background:#93c5fd;}"
        )
        self.plan_btn.clicked.connect(self.on_plan)
        input_row.addWidget(self.input, 1)
        input_row.addWidget(self.plan_btn)
        layout.addLayout(input_row)

        self.task_list = QListWidget()
        self.task_list.setStyleSheet(
            "QListWidget{border:1px solid #e5e7eb; border-radius:8px; "
            "background:#ffffff; padding:4px; font-size:13px;}"
        )
        self.task_list.setDragDropMode(QListWidget.InternalMove)
        layout.addWidget(self.task_list, 1)

        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("添加")
        self.del_btn = QPushButton("删除")
        self.up_btn = QPushButton("上移")
        self.down_btn = QPushButton("下移")
        self.run_btn = QPushButton("执行任务")
        self.run_btn.setStyleSheet(
            "QPushButton{background:#16a34a;color:white;border:none;border-radius:6px;padding:7px 16px;}"
            "QPushButton:disabled{background:#86efac;}"
        )
        for b in (self.add_btn, self.del_btn, self.up_btn, self.down_btn):
            b.setStyleSheet(
                "QPushButton{border:1px solid #d1d5db;border-radius:6px;padding:6px 12px;background:#fff;}"
            )
        self.add_btn.clicked.connect(self.on_add_task)
        self.del_btn.clicked.connect(self.on_del_task)
        self.up_btn.clicked.connect(lambda: self._move_task(-1))
        self.down_btn.clicked.connect(lambda: self._move_task(1))
        self.run_btn.clicked.connect(self.on_run_tasks)

        for b in (self.add_btn, self.del_btn, self.up_btn, self.down_btn):
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        btn_row.addWidget(self.run_btn)
        layout.addLayout(btn_row)
        return panel

    # ── ④ 文件面板 ────────────────────────────────────────────────────────

    def _build_file_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("④ 项目文件与资源")
        title.setStyleSheet("font-weight:600; color:#374151; font-size:13px;")
        layout.addWidget(title)

        split = QSplitter(Qt.Vertical)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setStyleSheet(
            "QTreeWidget{border:1px solid #e5e7eb; border-radius:8px; "
            "background:#ffffff; font-size:12.5px;}"
        )
        self.tree.itemClicked.connect(self.on_file_clicked)

        self.preview = QLabel("选中图片可预览")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(160)
        self.preview.setStyleSheet(
            "QLabel{border:1px solid #e5e7eb; border-radius:8px; "
            "background:#fafafa; color:#9ca3af;}"
        )

        split.addWidget(self.tree)
        split.addWidget(self.preview)
        split.setSizes([420, 200])
        layout.addWidget(split, 1)
        return panel

    # ── 组装 ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QSplitter(Qt.Horizontal)
        left = QSplitter(Qt.Vertical)
        right = QSplitter(Qt.Vertical)

        left.addWidget(self._build_log_panel())
        left.addWidget(self._build_memory_panel())
        right.addWidget(self._build_task_panel())
        right.addWidget(self._build_file_panel())

        outer.addWidget(left)
        outer.addWidget(right)
        outer.setSizes([640, 700])
        left.setSizes([420, 380])
        right.setSizes([440, 360])

        self.setCentralWidget(outer)
        self.setStyleSheet("QMainWindow{background:#f3f4f6;}")

    # ── 日志 ──────────────────────────────────────────────────────────────

    def _setup_logging(self) -> None:
        self.bridge = LogBridge()
        self.bridge.message.connect(self.append_log)
        handler = QtLogHandler(self.bridge)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-5s | %(name)-32s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root = logging.getLogger("imgsearch_agent")
        root.addHandler(handler)
        root.setLevel(logging.INFO)

        # 第三方噪音降级
        for noisy in ("httpx", "urllib3", "chromadb", "PIL"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    def append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)
        bar = self.log_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    # ── 记忆渲染 ──────────────────────────────────────────────────────────

    def _render_memory(self) -> None:
        rows = []
        for who, text in self.conversation:
            color = "#2563eb" if who == "你" else "#111827"
            safe = html.escape(text).replace("\n", "<br/>")
            rows.append(
                f'<p style="margin:6px 0;color:{color};"><b>{who}：</b>{safe}</p>'
            )
        self.memory_view.setHtml("".join(rows) or "<p style='color:#9ca3af'>暂无交互记录</p>")
        bar = self.memory_view.verticalScrollBar()
        bar.setValue(bar.maximum())

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
        return self.agent

    def _set_busy(self, busy: bool) -> None:
        for b in (self.plan_btn, self.run_btn, self.add_btn, self.del_btn, self.up_btn, self.down_btn):
            b.setEnabled(not busy)
        self.input.setEnabled(not busy)

    def on_plan(self) -> None:
        text = self.input.text().strip()
        if not text or self.worker is not None:
            return
        self.conversation.append(("你", text))
        self._render_memory()
        self._set_busy(True)
        self.append_log(f">>> 拆解任务：{text}")

        self.worker = Worker(lambda: self._get_agent().plan_tasks(text))
        self.worker.done.connect(self.on_plan_done)
        self.worker.failed.connect(self.on_worker_failed)
        self.worker.finished.connect(lambda: self._finish_worker())
        self.worker.start()

    def on_plan_done(self, tasks) -> None:
        self.task_list.clear()
        for t in tasks:
            self._add_task_item(t)
        self.append_log(f">>> 已生成 {len(tasks)} 条任务，可编辑后执行")

    def on_run_tasks(self) -> None:
        tasks = self._collect_tasks()
        if not tasks or self.worker is not None:
            return
        self._set_busy(True)
        self.append_log(f">>> 开始执行 {sum(1 for t in tasks if t.status == 'pending')} 条任务")

        self.worker = Worker(lambda: self._get_agent().run_tasks(tasks))
        self.worker.done.connect(self.on_run_done)
        self.worker.failed.connect(self.on_worker_failed)
        self.worker.finished.connect(lambda: self._finish_worker())
        self.worker.start()

    def on_run_done(self, results) -> None:
        self.last_results = []
        for r in results:
            self.conversation.append(("Agent", r.reply))
            self.last_results.extend(r.results)
            self.append_log(
                f">>> 任务完成 | 置信度 {r.confidence:.3f} | 重试 {r.retries} | trace {r.trace_id}"
            )
        self._render_memory()
        self._refresh_results_node()

    def on_worker_failed(self, error: str) -> None:
        self.append_log(f"!!! 执行失败：{error}")
        self.conversation.append(("Agent", f"出错了：{error}"))
        self._render_memory()

    def _finish_worker(self) -> None:
        self.worker = None
        self._set_busy(False)
        self.input.setFocus()

    # ── 文件树 ────────────────────────────────────────────────────────────

    def _load_file_tree(self) -> None:
        self.tree.clear()
        root_item = QTreeWidgetItem([ROOT.name])
        root_item.setData(0, Qt.UserRole, str(ROOT))
        self.tree.addTopLevelItem(root_item)

        self.results_node = QTreeWidgetItem(["最近检索结果"])
        root_item.addChild(self.results_node)

        for name in ("models", "data", "configs", "datasets", "docs", "src", "logs"):
            path = ROOT / name
            if not path.exists():
                continue
            node = QTreeWidgetItem([name])
            node.setData(0, Qt.UserRole, str(path))
            root_item.addChild(node)
            self._fill_dir(node, path, depth=1, limit=40)

        root_item.setExpanded(True)

    def _fill_dir(self, parent: QTreeWidgetItem, path: Path, depth: int, limit: int) -> None:
        if depth <= 0:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))[:limit]
        except OSError:
            return
        for p in entries:
            if p.name in ("__pycache__", ".git", ".pytest_cache"):
                continue
            item = QTreeWidgetItem([p.name])
            item.setData(0, Qt.UserRole, str(p))
            parent.addChild(item)
            if p.is_dir():
                self._fill_dir(item, p, depth - 1, limit)

    def _refresh_results_node(self) -> None:
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
                    pixmap.scaled(420, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                self.preview.setToolTip(str(p))
                return
        self.preview.setText(f"{p.name}\n\n{p}")
        self.preview.setPixmap(QPixmap())


def main() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("imgsearch Agent")
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
