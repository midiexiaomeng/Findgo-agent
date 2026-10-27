"""imgsearch Agent 桌面应用（PyQt5）。

原生窗口，不占端口、不依赖浏览器。左侧对话，右侧图片墙。
双击 start-app.bat 启动（无控制台）。
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

THUMB_W, THUMB_H = 220, 165
COLUMNS = 3


# ── 后台线程：避免检索时界面卡死 ──────────────────────────────────────────────


class AgentWorker(QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def run(self) -> None:
        try:
            from imgsearch_agent.app import get_agent

            agent = get_agent()
            result = agent.run(self.message)
            self.done.emit(result)
        except Exception as e:  # 兜底，避免线程静默退出
            self.failed.emit(str(e))


# ── 图片网格 ──────────────────────────────────────────────────────────────────


class ImageGrid(QWidget):
    def __init__(self):
        super().__init__()
        self.grid = QGridLayout(self)
        self.grid.setSpacing(10)
        self.grid.setContentsMargins(10, 10, 10, 10)
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
            label.setFixedSize(THUMB_W + 8, THUMB_H + 8)
            label.setStyleSheet(
                "border:1px solid #e5e7eb; border-radius:8px; background:#fff;"
            )
            label.setToolTip(f"{path}\n相似度 {score:.3f}")
            self.grid.addWidget(label, index // COLUMNS, index % COLUMNS)
            index += 1

        if index == 0:
            empty = QLabel("没有可展示的图片")
            empty.setAlignment(Qt.AlignCenter)
            empty.setStyleSheet("color:#9ca3af; padding:40px;")
            self.grid.addWidget(empty, 0, 0, 1, COLUMNS)


# ── 主窗口 ────────────────────────────────────────────────────────────────────


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("imgsearch Agent · 多模态图像检索")
        self.resize(1240, 800)
        self.worker: AgentWorker | None = None
        self._build_ui()

    # ── 界面 ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)

        # 左：对话
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(10)

        title = QLabel("imgsearch Agent")
        title.setStyleSheet("font-size:20px; font-weight:700; color:#111827;")
        subtitle = QLabel("ReAct 编排 · 自主决策 · 检索 · 校验 · 总结")
        subtitle.setStyleSheet("font-size:12px; color:#6b7280;")
        left_layout.addWidget(title)
        left_layout.addWidget(subtitle)

        self.chat = QTextBrowser()
        self.chat.setOpenExternalLinks(False)
        self.chat.setStyleSheet(
            "QTextBrowser{border:1px solid #e5e7eb; border-radius:10px; "
            "background:#ffffff; padding:8px; font-size:13px;}"
        )
        left_layout.addWidget(self.chat, 1)

        input_row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("描述你想找的图片，回车发送")
        self.input.setStyleSheet(
            "QLineEdit{border:1px solid #d1d5db; border-radius:8px; padding:8px 10px; font-size:13px;}"
        )
        self.send_btn = QPushButton("发送")
        self.send_btn.setStyleSheet(
            "QPushButton{background:#2563eb; color:white; border:none; "
            "border-radius:8px; padding:8px 18px; font-size:13px;}"
            "QPushButton:disabled{background:#93c5fd;}"
        )
        input_row.addWidget(self.input, 1)
        input_row.addWidget(self.send_btn)
        left_layout.addLayout(input_row)

        self.status = QLabel("就绪。首次对话会加载 CLIP，约 10 到 20 秒。")
        self.status.setStyleSheet("font-size:12px; color:#6b7280;")
        self.status.setWordWrap(True)
        left_layout.addWidget(self.status)

        # 右：图片墙
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(8)

        gallery_title = QLabel("检索结果")
        gallery_title.setStyleSheet("font-size:14px; font-weight:600; color:#374151;")
        right_layout.addWidget(gallery_title)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet(
            "QScrollArea{border:1px solid #e5e7eb; border-radius:10px; background:#fafafa;}"
        )
        self.grid = ImageGrid()
        self.scroll.setWidget(self.grid)
        right_layout.addWidget(self.scroll, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([440, 760])

        self.setCentralWidget(splitter)
        self.setStyleSheet("QMainWindow{background:#f3f4f6;}")

        self.send_btn.clicked.connect(self.on_send)
        self.input.returnPressed.connect(self.on_send)

    # ── 交互 ──────────────────────────────────────────────────────────────

    def on_send(self) -> None:
        text = self.input.text().strip()
        if not text or self.worker is not None:
            return

        self._append("你", text, "#2563eb")
        self.input.clear()
        self.input.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.status.setText("Agent 正在思考与检索…")

        self.worker = AgentWorker(text)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self._clear_worker)
        self.worker.start()

    def on_done(self, result) -> None:
        self._append("Agent", result.reply, "#111827")
        self.grid.set_results(result.results)
        self.status.setText(
            f"工具 {result.tool or '无'} · 置信度 {result.confidence:.3f}"
            f" · 重试 {result.retries} · trace {result.trace_id}"
        )

    def on_failed(self, error: str) -> None:
        self._append("Agent", f"出错了：{error}", "#b91c1c")
        self.status.setText("执行失败，请检查日志或 API 配置。")

    def _clear_worker(self) -> None:
        self.worker = None
        self.input.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.input.setFocus()

    def _append(self, who: str, text: str, color: str) -> None:
        safe = html.escape(str(text)).replace("\n", "<br/>")
        self.chat.append(
            f'<p style="margin:6px 0; color:{color};"><b>{who}：</b>{safe}</p>'
        )
        self.chat.verticalScrollBar().setValue(
            self.chat.verticalScrollBar().maximum()
        )


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
