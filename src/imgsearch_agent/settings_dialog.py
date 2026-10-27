"""设置对话框：左侧分类导航，右侧具体设置。

三个分类：
- 添加模型：填写服务商、自动带出网址与常用模型，测试成功后写入
- 检索参数：top_k、相似度阈值、重试次数
- 关于：版本与路径
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
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
                [{"role": "user", "content": "只回复两个字：正常"}], max_tokens=32
            )
            self.done.emit(True, reply.strip()[:60])
        except Exception as e:
            self.done.emit(False, str(e))


class SettingsDialog(QDialog):
    models_changed = pyqtSignal()

    PAGE_MODELS = 0
    PAGE_PARAMS = 1
    PAGE_ABOUT = 2

    def __init__(self, store: ModelStore, parent=None, page: int = PAGE_MODELS):
        super().__init__(parent)
        self.store = store
        self.worker: TestWorker | None = None

        self.setWindowTitle("设置")
        self.resize(640, 470)
        self._build_ui()
        self.nav.setCurrentRow(page)
        self._on_provider_changed()
        self._refresh_list()
        self._load_params()

    # ── 界面骨架 ──────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.nav = QListWidget()
        self.nav.setFixedWidth(132)
        self.nav.setStyleSheet(
            "QListWidget{border:none;background:#f8fafc;padding:8px 4px;font-size:13px;}"
            "QListWidget::item{padding:9px 12px;border-radius:6px;margin:2px 6px;}"
            "QListWidget::item:selected{background:#e0e7ff;color:#1e40af;}"
        )
        self.nav.addItems(["添加模型", "检索参数", "关于"])
        self.nav.currentRowChanged.connect(self._on_nav_changed)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_models_page())
        self.stack.addWidget(self._build_params_page())
        self.stack.addWidget(self._build_about_page())

        root.addWidget(self.nav)
        root.addWidget(self.stack, 1)

    def _on_nav_changed(self, index: int) -> None:
        if 0 <= index < self.stack.count():
            self.stack.setCurrentIndex(index)

    # ── 添加模型页 ────────────────────────────────────────────────────────

    def _build_models_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        tip = QLabel(
            "选择服务商后会自动填入接口地址和常用模型。"
            "填写 API Key 并「测试连接」，成功后才会计入已添加模型。"
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
        form.addRow("接口地址", self.base_url)

        self.model = QComboBox()
        self.model.setEditable(True)
        form.addRow("模型", self.model)

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

        layout.addWidget(QLabel("已添加的模型"))
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
        return page

    # ── 检索参数页 ────────────────────────────────────────────────────────

    def _build_params_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        tip = QLabel("修改后写入 configs/agent.yaml，下次启动生效。")
        tip.setStyleSheet("color:#6b7280; font-size:12px;")
        layout.addWidget(tip)

        form = QFormLayout()
        form.setSpacing(10)

        self.top_k = QSpinBox()
        self.top_k.setRange(1, 30)
        form.addRow("返回数量 top_k", self.top_k)

        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setSingleStep(0.05)
        self.threshold.setDecimals(2)
        form.addRow("相似度阈值", self.threshold)

        self.retries = QSpinBox()
        self.retries.setRange(0, 5)
        form.addRow("最大重试次数", self.retries)

        layout.addLayout(form)

        save_btn = QPushButton("保存参数")
        save_btn.setStyleSheet(
            "QPushButton{background:#16a34a;color:white;border:none;"
            "border-radius:6px;padding:7px 18px;}"
        )
        save_btn.clicked.connect(self.on_save_params)
        layout.addWidget(save_btn, 0, Qt.AlignLeft)

        self.params_status = QLabel("")
        self.params_status.setStyleSheet("color:#6b7280; font-size:12px;")
        layout.addWidget(self.params_status)
        layout.addStretch(1)
        return page

    # ── 关于页 ────────────────────────────────────────────────────────────

    def _build_about_page(self) -> QWidget:
        from . import __version__
        from .config import PROJECT_ROOT

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        lines = [
            ("版本", __version__),
            ("项目目录", str(PROJECT_ROOT)),
            ("配置文件", str(PROJECT_ROOT / "configs" / "agent.yaml")),
            ("模型配置", str(PROJECT_ROOT / "configs" / "models.json")),
            ("会话记忆", str(PROJECT_ROOT / "data" / "agent_sessions.sqlite")),
        ]
        for key, value in lines:
            row = QHBoxLayout()
            k = QLabel(key)
            k.setFixedWidth(80)
            k.setStyleSheet("color:#6b7280;")
            v = QLabel(value)
            v.setWordWrap(True)
            v.setStyleSheet("color:#111827; font-size:12px;")
            row.addWidget(k)
            row.addWidget(v, 1)
            layout.addLayout(row)

        layout.addStretch(1)
        return page

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

    # ── 检索参数 ──────────────────────────────────────────────────────────

    def _load_params(self) -> None:
        from .config import AgentConfig

        cfg = AgentConfig()
        self.top_k.setValue(cfg.top_k)
        self.threshold.setValue(cfg.similarity_threshold)
        self.retries.setValue(cfg.max_retries)

    def on_save_params(self) -> None:
        import yaml

        from .config import PROJECT_ROOT

        path = PROJECT_ROOT / "configs" / "agent.yaml"
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            data.setdefault("agent", {})
            data["agent"]["top_k"] = self.top_k.value()
            data["agent"]["similarity_threshold"] = round(self.threshold.value(), 2)
            data["agent"]["max_retries"] = self.retries.value()
            path.write_text(
                yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            self.params_status.setText("已保存，下次启动生效。")
        except Exception as e:
            self.params_status.setText(f"保存失败：{e}")
