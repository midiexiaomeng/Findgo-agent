"""模型设置对话框：添加、测试、删除模型。

只有测试调用成功的模型才会写入 ModelStore，主界面只展示这份列表。
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from .llm.client import ChatClient
from .model_store import PROVIDERS, ModelStore


class TestWorker(QThread):
    done = pyqtSignal(bool, str)

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    def run(self) -> None:
        try:
            client = ChatClient(self.cfg)
            reply = client.chat(
                [{"role": "user", "content": "只回复两个字：正常"}], max_tokens=16
            )
            self.done.emit(True, reply.strip()[:60])
        except Exception as e:
            self.done.emit(False, str(e))


class SettingsDialog(QDialog):
    models_changed = pyqtSignal()

    def __init__(self, store: ModelStore, parent=None):
        super().__init__(parent)
        self.store = store
        self.worker: TestWorker | None = None
        self.setWindowTitle("设置 · 模型管理")
        self.resize(560, 520)
        self._build_ui()
        self._on_provider_changed()
        self._refresh_list()

    # ── 界面 ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        tip = QLabel(
            "填写服务商、模型与 API Key，点「测试连接」。"
            "只有测试成功的模型才会出现在主界面的模型选择里。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#6b7280; font-size:12px;")
        layout.addWidget(tip)

        form = QFormLayout()
        form.setSpacing(8)

        self.provider = QComboBox()
        for key, info in PROVIDERS.items():
            self.provider.addItem(info["name"], key)
        self.provider.currentIndexChanged.connect(self._on_provider_changed)
        form.addRow("服务商", self.provider)

        self.base_url = QLineEdit()
        form.addRow("Base URL", self.base_url)

        self.model = QComboBox()
        self.model.setEditable(True)
        form.addRow("模型名", self.model)

        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setPlaceholderText("sk-...")
        form.addRow("API Key", self.api_key)

        layout.addLayout(form)

        test_row = QHBoxLayout()
        self.test_btn = QPushButton("测试连接")
        self.test_btn.setStyleSheet(
            "QPushButton{background:#2563eb;color:white;border:none;"
            "border-radius:6px;padding:7px 16px;}"
            "QPushButton:disabled{background:#93c5fd;}"
        )
        self.test_btn.clicked.connect(self.on_test)
        self.test_status = QLabel("")
        self.test_status.setWordWrap(True)
        test_row.addWidget(self.test_btn)
        test_row.addWidget(self.test_status, 1)
        layout.addLayout(test_row)

        layout.addWidget(QLabel("已添加的模型（测试通过）"))
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(
            "QListWidget{border:1px solid #e5e7eb;border-radius:6px;"
            "background:#fff;padding:4px;font-size:12.5px;}"
        )
        layout.addWidget(self.list_widget, 1)

        bottom = QHBoxLayout()
        self.del_btn = QPushButton("删除选中")
        self.del_btn.clicked.connect(self.on_delete)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        bottom.addWidget(self.del_btn)
        bottom.addStretch(1)
        bottom.addWidget(close_btn)
        layout.addLayout(bottom)

    # ── 交互 ──────────────────────────────────────────────────────────────

    def _on_provider_changed(self) -> None:
        key = self.provider.currentData()
        info = PROVIDERS.get(key, {})
        self.base_url.setText(info.get("base_url", ""))
        self.model.clear()
        self.model.addItems(info.get("models", []))
        if info.get("models"):
            self.model.setCurrentIndex(0)

    def on_test(self) -> None:
        cfg = {
            "backend": "openai",
            "model": self.model.currentText().strip(),
            "base_url": self.base_url.text().strip(),
            "api_key": self.api_key.text().strip(),
        }
        if not cfg["model"] or not cfg["base_url"] or not cfg["api_key"]:
            self.test_status.setText("请填写完整")
            return

        self.test_btn.setEnabled(False)
        self.test_status.setText("测试中…")
        self.worker = TestWorker(cfg)
        self.worker.done.connect(self.on_test_done)
        self.worker.start()

    def on_test_done(self, ok: bool, message: str) -> None:
        self.test_btn.setEnabled(True)
        if not ok:
            self.test_status.setText(f"✗ 失败：{message[:120]}")
            return

        self.test_status.setText(f"✓ 成功：{message}")
        self.store.add(
            provider=str(self.provider.currentData()),
            model=self.model.currentText().strip(),
            base_url=self.base_url.text().strip(),
            api_key=self.api_key.text().strip(),
            label=f"{self.provider.currentText()} · {self.model.currentText().strip()}",
        )
        self._refresh_list()
        self.models_changed.emit()

    def on_delete(self) -> None:
        item = self.list_widget.currentItem()
        if item is None:
            return
        self.store.remove(str(item.data(Qt.UserRole)))
        self._refresh_list()
        self.models_changed.emit()

    def _refresh_list(self) -> None:
        self.list_widget.clear()
        for entry in self.store.models:
            label = entry.get("label", entry.get("model", ""))
            if entry.get("id") == self.store.active:
                label += "    （当前）"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, entry.get("id"))
            item.setToolTip(entry.get("base_url", ""))
            self.list_widget.addItem(item)
