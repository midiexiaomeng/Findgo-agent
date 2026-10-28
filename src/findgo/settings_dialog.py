"""设置对话框：左侧分类导航，右侧具体设置。

三个分类：
- 添加模型：填写服务商、自动带出网址与常用模型，测试成功后写入
- 检索参数：top_k、相似度阈值、重试次数
- 关于：版本与路径
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .llm.client import ChatClient
from .model_store import PROVIDERS, ModelStore
from .theme import DIALOG_QSS


class TestWorker(QThread):
    done = pyqtSignal(bool, str)

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    def run(self) -> None:
        try:
            client = ChatClient(self.cfg)
            reply = client.chat([{"role": "user", "content": "只回复两个字：正常"}], max_tokens=32)
            self.done.emit(True, reply.strip()[:60])
        except Exception as e:
            self.done.emit(False, str(e))


class SettingsDialog(QDialog):
    models_changed = pyqtSignal()
    config_changed = pyqtSignal()

    PAGE_MODELS = 0
    PAGE_PARAMS = 1
    PAGE_VECTOR = 2
    PAGE_PATHS = 3
    PAGE_TRASH = 4
    PAGE_CUSTOM = 5
    PAGE_DIAG = 6
    PAGE_ABOUT = 7

    def __init__(
        self,
        store: ModelStore,
        parent=None,
        page: int = PAGE_MODELS,
        agent=None,
    ):
        super().__init__(parent)
        self.store = store
        self.agent = agent
        self.worker: TestWorker | None = None

        self.setWindowTitle("设置")
        self.resize(640, 470)
        self.setStyleSheet(DIALOG_QSS)
        self._build_ui()
        self.nav.setCurrentRow(page)
        self._on_provider_changed()
        self._refresh_list()
        self._load_params()
        self._load_vector()
        self._load_paths()
        self._load_trash()
        self._load_custom_tools()

    # ── 界面骨架 ──────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.nav = QListWidget()
        self.nav.setFixedWidth(132)
        self.nav.setStyleSheet(
            "QListWidget{border:none;background:#F8F7F3;padding:8px 4px;font-size:13px;}"
            "QListWidget::item{padding:9px 12px;border-radius:6px;margin:2px 6px;}"
            "QListWidget::item:selected{background:#E3EFE7;color:#2E2E2E;}"
        )
        self.nav.addItems(
            [
                "添加模型",
                "检索参数",
                "向量库",
                "目录与安全",
                "暂存区",
                "自定义工具",
                "环境自检",
                "关于",
            ]
        )
        self.nav.currentRowChanged.connect(self._on_nav_changed)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_models_page())
        self.stack.addWidget(self._build_params_page())
        self.stack.addWidget(self._build_vector_page())
        self.stack.addWidget(self._build_paths_page())
        self.stack.addWidget(self._build_trash_page())
        self.stack.addWidget(self._build_custom_page())
        self.stack.addWidget(self._build_diag_page())
        self.stack.addWidget(self._build_about_page())

        root.addWidget(self.nav)
        root.addWidget(self.stack, 1)

    def _on_nav_changed(self, index: int) -> None:
        if 0 <= index < self.stack.count():
            self.stack.setCurrentIndex(index)
        if index == self.PAGE_DIAG and hasattr(self, "diag_list"):
            self.run_diagnostics()

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
        tip.setStyleSheet("color:#8A8A8A; font-size:12px;")
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
            "QPushButton{background:#6F9C86;color:white;border:none;"
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

        tip = QLabel(
            "返回数量由模型根据你的描述自动判断（例如“两张”就是 2 张），这里只调整置信阈值与重试次数。"
        )
        tip.setStyleSheet("color:#8A8A8A; font-size:12px;")
        layout.addWidget(tip)

        form = QFormLayout()
        form.setSpacing(10)

        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setSingleStep(0.05)
        self.threshold.setDecimals(2)
        form.addRow("相似度阈值", self.threshold)

        self.retries = QSpinBox()
        self.retries.setRange(0, 5)
        form.addRow("最大重试次数", self.retries)

        self.max_sessions = QSpinBox()
        self.max_sessions.setRange(1, 200)
        self.max_sessions.setToolTip("超过这个数量后，新建对话会删除最旧的会话")
        form.addRow("保留会话数", self.max_sessions)

        self.top_k = QSpinBox()
        self.top_k.setRange(1, 50)
        self.top_k.setToolTip("用户没明确说数量时，默认返回几张")
        form.addRow("默认返回数", self.top_k)

        self.recall_k = QSpinBox()
        self.recall_k.setRange(10, 300)
        self.recall_k.setToolTip("向量库先召回多少候选，越多越不容易漏")
        form.addRow("召回候选数", self.recall_k)

        self.rerank_enabled = QCheckBox("用大模型对候选做语义重排")
        form.addRow("LLM 重排", self.rerank_enabled)

        self.caption_enabled = QCheckBox("索引时生成图像描述，参与重排")
        form.addRow("图像描述", self.caption_enabled)

        layout.addLayout(form)

        save_btn = QPushButton("保存参数")
        save_btn.setStyleSheet(
            "QPushButton{background:#6F9C86;color:white;border:none;"
            "border-radius:6px;padding:7px 18px;}"
        )
        save_btn.clicked.connect(self.on_save_params)
        layout.addWidget(save_btn, 0, Qt.AlignLeft)

        self.params_status = QLabel("")
        self.params_status.setStyleSheet("color:#8A8A8A; font-size:12px;")
        layout.addWidget(self.params_status)
        layout.addStretch(1)
        return page

    # ── 向量库页 ──────────────────────────────────────────────────────────

    def _build_vector_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        tip = QLabel("切换向量库只更换连接，两个后端的数据不共享，切换后需要重新建索引。")
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#8A8A8A; font-size:12px;")
        layout.addWidget(tip)

        form = QFormLayout()
        form.setSpacing(10)

        self.backend_box = QComboBox()
        self.backend_box.addItems(["qdrant", "chroma"])
        form.addRow("向量库", self.backend_box)

        self.collection_edit = QLineEdit()
        form.addRow("集合名", self.collection_edit)

        self.vector_info = QLabel("")
        self.vector_info.setStyleSheet("color:#4A4A4A; font-size:12px;")
        self.vector_info.setWordWrap(True)
        form.addRow("当前状态", self.vector_info)

        layout.addLayout(form)

        apply_btn = QPushButton("应用后端")
        apply_btn.setStyleSheet(
            "QPushButton{background:#6F9C86;color:white;border:none;"
            "border-radius:6px;padding:7px 18px;}"
        )
        apply_btn.clicked.connect(self.on_apply_vector)
        layout.addWidget(apply_btn, 0, Qt.AlignLeft)

        self.vector_status = QLabel("")
        self.vector_status.setStyleSheet("color:#8A8A8A; font-size:12px;")
        self.vector_status.setWordWrap(True)
        layout.addWidget(self.vector_status)
        layout.addStretch(1)
        return page

    # ── 目录与安全页 ──────────────────────────────────────────────────────

    def _build_paths_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        tip = QLabel(
            "这里列出的是只读素材目录：Agent 能读图，但不能修改或删除。"
            "写操作（缓存、向量库、导出）只会发生在项目的 data 目录。"
            "系统目录（Windows、Program Files）始终禁止。留空表示使用默认范围。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#8A8A8A; font-size:12px;")
        layout.addWidget(tip)

        self.roots_list = QListWidget()
        self.roots_list.setStyleSheet(
            "QListWidget{border:1px solid #e5e7eb;border-radius:6px;"
            "background:#fff;padding:4px;font-size:12.5px;}"
        )
        layout.addWidget(self.roots_list, 1)

        buttons = QHBoxLayout()
        add_btn = QPushButton("添加目录")
        add_btn.clicked.connect(self.on_add_root)
        del_btn = QPushButton("移除选中")
        del_btn.clicked.connect(self.on_del_root)
        save_btn = QPushButton("保存")
        save_btn.setStyleSheet(
            "QPushButton{background:#6F9C86;color:white;border:none;"
            "border-radius:6px;padding:7px 18px;}"
        )
        save_btn.clicked.connect(self.on_save_paths)
        buttons.addWidget(add_btn)
        buttons.addWidget(del_btn)
        buttons.addStretch(1)
        buttons.addWidget(save_btn)
        layout.addLayout(buttons)

        self.paths_status = QLabel("")
        self.paths_status.setStyleSheet("color:#8A8A8A; font-size:12px;")
        layout.addWidget(self.paths_status)
        return page

    # ── 暂存区页 ──────────────────────────────────────────────────────────

    def _build_trash_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        tip = QLabel(
            "待删除的图片先进入暂存区，用硬链接保留，不占额外空间。"
            "超过留存天数或容量上限后，才会被永久删除。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#8A8A8A; font-size:12px;")
        layout.addWidget(tip)

        form = QFormLayout()
        form.setSpacing(10)

        self.trash_path_edit = QLineEdit()
        self.trash_path_edit.setReadOnly(True)
        self.trash_path_edit.setPlaceholderText("默认 ./data/trash_staging")
        browse = QPushButton("选择目录")
        browse.clicked.connect(self.on_pick_trash_path)

        path_row = QWidget()
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.setSpacing(6)
        path_layout.addWidget(self.trash_path_edit, 1)
        path_layout.addWidget(browse)
        form.addRow("暂存位置", path_row)

        self.trash_days = QSpinBox()
        self.trash_days.setRange(1, 365)
        self.trash_days.setSuffix(" 天")
        form.addRow("留存时间", self.trash_days)

        self.trash_gb = QDoubleSpinBox()
        self.trash_gb.setRange(0.5, 1000.0)
        self.trash_gb.setSingleStep(0.5)
        self.trash_gb.setDecimals(1)
        self.trash_gb.setSuffix(" GB")
        form.addRow("容量上限", self.trash_gb)

        self.trash_auto = QCheckBox("超过限制后自动清理最旧的文件")
        form.addRow("自动清理", self.trash_auto)

        layout.addLayout(form)

        dl_title = QLabel("联网下载缓存")
        dl_title.setStyleSheet("font-weight:600; color:#2E2E2E; font-size:12px;")
        layout.addWidget(dl_title)

        self.dl_path_edit = QLineEdit()
        self.dl_path_edit.setReadOnly(True)
        self.dl_path_edit.setPlaceholderText("默认 ./data/downloads")
        dl_browse = QPushButton("选择目录")
        dl_browse.clicked.connect(self.on_pick_download_path)

        dl_row = QWidget()
        dl_layout = QHBoxLayout(dl_row)
        dl_layout.setContentsMargins(0, 0, 0, 0)
        dl_layout.setSpacing(6)
        dl_layout.addWidget(self.dl_path_edit, 1)
        dl_layout.addWidget(dl_browse)
        form.addRow("下载位置", dl_row)

        self.dl_gb = QDoubleSpinBox()
        self.dl_gb.setRange(0.5, 500.0)
        self.dl_gb.setSingleStep(0.5)
        self.dl_gb.setDecimals(1)
        self.dl_gb.setSuffix(" GB")
        form.addRow("下载容量上限", self.dl_gb)

        self.trash_info = QLabel("")
        self.trash_info.setStyleSheet("color:#4A4A4A; font-size:12px;")
        layout.addWidget(self.trash_info)

        action_row = QHBoxLayout()
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self._refresh_trash_info)
        self.purge_btn = QPushButton("永久删除暂存")
        self.purge_btn.setStyleSheet(
            "QPushButton{background:#C97B6E;color:white;border:none;"
            "border-radius:6px;padding:7px 16px;}"
        )
        self.purge_btn.clicked.connect(self.on_purge_trash)
        action_row.addWidget(refresh_btn)
        action_row.addStretch(1)
        action_row.addWidget(self.purge_btn)
        layout.addLayout(action_row)

        save_btn = QPushButton("保存暂存区设置")
        save_btn.setStyleSheet(
            "QPushButton{background:#6F9C86;color:white;border:none;"
            "border-radius:6px;padding:7px 18px;}"
        )
        save_btn.clicked.connect(self.on_save_trash)
        layout.addWidget(save_btn, 0, Qt.AlignLeft)

        self.trash_status = QLabel("")
        self.trash_status.setStyleSheet("color:#8A8A8A; font-size:12px;")
        layout.addWidget(self.trash_status)
        layout.addStretch(1)
        return page

    def _refresh_trash_info(self) -> None:
        try:
            from .config import PROJECT_ROOT
            from .trash import TrashStaging

            staging = TrashStaging(PROJECT_ROOT / "data" / "trash_staging")
            stats = staging.stats()
            mb = stats["total_bytes"] / 1024**2
            self.trash_info.setText(f"暂存区现有 {stats['count']} 张，占用 {mb:.1f} MB")
        except Exception as e:
            self.trash_info.setText(f"读取暂存区失败：{e}")

    def on_purge_trash(self) -> None:
        """永久删除必须人工确认，Agent 无法直接触发。"""
        if self.agent is None:
            self.trash_status.setText("Agent 未初始化，无法操作暂存区")
            return

        tool = self.agent.tools.get("trash_manage")
        if tool is None:
            self.trash_status.setText("暂存区工具未注册")
            return

        items = tool.staging.list_staged()
        if not items:
            self.trash_status.setText("暂存区是空的")
            return

        reply = QMessageBox.question(
            self,
            "确认永久删除",
            f"将永久删除暂存区里的 {len(items)} 张图片，原图会一起删除且无法恢复。确定吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        total = 0
        for task_id in {item["task_id"] for item in items}:
            result = tool.run("purge", task_id=task_id, allow_danger=True)
            total += int(result.get("purged", 0))

        self.trash_status.setText(f"已永久删除 {total} 张")
        self._refresh_trash_info()

    def on_pick_download_path(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择下载缓存目录")
        if folder:
            self.dl_path_edit.setText(folder)

    def on_pick_trash_path(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择暂存区目录")
        if folder:
            self.trash_path_edit.setText(folder)

    def _load_trash(self) -> None:
        try:
            from .config import AgentConfig

            cfg = AgentConfig()
            self.dl_path_edit.setText(str(cfg.download_path))
            self.dl_gb.setValue(float(cfg.download_max_gb))
            self.trash_path_edit.setText(str(cfg.trash_path))
            self.trash_days.setValue(int(cfg.trash_max_days))
            self.trash_gb.setValue(float(cfg.trash_max_gb))
            self.trash_auto.setChecked(bool(cfg.trash_auto_clean))
        except Exception:
            self.trash_days.setValue(14)
            self.trash_gb.setValue(10.0)
            self.trash_auto.setChecked(True)

        self._refresh_trash_info()

    def on_save_trash(self) -> None:
        import yaml

        from .config import PROJECT_ROOT

        path = PROJECT_ROOT / "configs" / "agent.yaml"
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            data.setdefault("download_cache", {})
            data["download_cache"]["path"] = self.dl_path_edit.text().strip()
            data["download_cache"]["max_total_size_gb"] = round(self.dl_gb.value(), 1)

            data.setdefault("trash_staging", {})
            data["trash_staging"]["path"] = self.trash_path_edit.text().strip()
            data["trash_staging"]["max_keep_days"] = self.trash_days.value()
            data["trash_staging"]["max_total_size_gb"] = round(self.trash_gb.value(), 1)
            data["trash_staging"]["auto_clean"] = self.trash_auto.isChecked()
            path.write_text(
                yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            self.trash_status.setText("已保存，下次清理时生效。")
            self.config_changed.emit()
        except Exception as e:
            self.trash_status.setText(f"保存失败：{e}")

    # ── 自定义工具页 ──────────────────────────────────────────────────────

    def _build_custom_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        tip = QLabel(
            "从动态工具审核时勾选保存下来的全局工具，存在 configs/custom_tools.yaml，"
            "和内置工具完全隔离，可以单独备份或清空。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#8A8A8A; font-size:12px;")
        layout.addWidget(tip)

        self.custom_list = QListWidget()
        self.custom_list.setStyleSheet(
            "QListWidget{border:1px solid #EDEBE6;border-radius:8px;"
            "background:#FFFFFF;padding:4px;font-size:12.5px;}"
        )
        layout.addWidget(self.custom_list, 1)

        row = QHBoxLayout()
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self._load_custom_tools)
        delete = QPushButton("删除选中")
        delete.setStyleSheet(
            "QPushButton{background:#C97B6E;color:white;border:none;"
            "border-radius:6px;padding:6px 14px;}"
        )
        delete.clicked.connect(self.on_delete_custom)
        row.addWidget(refresh)
        row.addStretch(1)
        row.addWidget(delete)
        layout.addLayout(row)

        self.custom_status = QLabel("")
        self.custom_status.setStyleSheet("color:#8A8A8A; font-size:12px;")
        layout.addWidget(self.custom_status)
        return page

    def _load_custom_tools(self) -> None:
        try:
            from .agent.custom_tools import list_custom_tools

            items = list_custom_tools()
        except Exception as e:
            self.custom_status.setText(f"读取失败：{e}")
            return

        self.custom_list.clear()
        for item in items:
            label = f"{item.get('name', '')} — {str(item.get('description', ''))[:36]}"
            li = QListWidgetItem(label)
            li.setData(Qt.UserRole, item.get("name"))
            li.setToolTip(item.get("safety_note", ""))
            self.custom_list.addItem(li)

        self.custom_status.setText(f"共 {len(items)} 个自定义工具")

    def on_delete_custom(self) -> None:
        item = self.custom_list.currentItem()
        if item is None:
            self.custom_status.setText("先选中一个工具")
            return

        name = str(item.data(Qt.UserRole))
        reply = QMessageBox.question(
            self,
            "删除自定义工具",
            f"确定删除「{name}」吗？删除后需要重新生成和审核。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            from .agent.custom_tools import delete_custom_tool

            delete_custom_tool(name)
            self.custom_status.setText(f"已删除 {name}")
            self._load_custom_tools()
        except Exception as e:
            self.custom_status.setText(f"删除失败：{e}")

    # ── 环境自检页 ────────────────────────────────────────────────────────

    def _build_diag_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        tip = QLabel("只读探测：不加载模型、不连接向量库，检查依赖、显卡、模型文件与配置。")
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#8A8A8A; font-size:12px;")
        layout.addWidget(tip)

        self.diag_list = QListWidget()
        self.diag_list.setStyleSheet(
            "QListWidget{border:1px solid #e5e7eb;border-radius:6px;"
            "background:#fff;padding:6px;font-size:12.5px;}"
        )
        layout.addWidget(self.diag_list, 1)

        row = QHBoxLayout()
        self.diag_btn = QPushButton("重新检测")
        self.diag_btn.clicked.connect(self.run_diagnostics)
        self.diag_status = QLabel("")
        self.diag_status.setStyleSheet("color:#8A8A8A; font-size:12px;")
        row.addWidget(self.diag_btn)
        row.addWidget(self.diag_status, 1)
        layout.addLayout(row)
        return page

    def run_diagnostics(self) -> None:
        from .diagnostics import run_checks, summarize

        self.diag_btn.setEnabled(False)
        self.diag_status.setText("检测中…")
        QApplication.processEvents()

        try:
            checks = run_checks()
            self.diag_list.clear()
            for check in checks:
                mark = "✓" if check["ok"] else "✗"
                item = QListWidgetItem(f"{mark}  {check['name']}：{check['detail']}")
                item.setForeground(QColor("#16a34a" if check["ok"] else "#dc2626"))
                self.diag_list.addItem(item)
            self.diag_status.setText(summarize(checks))
        except Exception as e:
            self.diag_status.setText(f"检测失败：{e}")
        finally:
            self.diag_btn.setEnabled(True)

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
            v.setStyleSheet("color:#2E2E2E; font-size:12px;")
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
        self.threshold.setValue(cfg.similarity_threshold)
        self.max_sessions.setValue(cfg.memory_max_sessions)
        self.retries.setValue(cfg.max_retries)

        try:
            from .retriever import Config

            rcfg = Config.from_yaml()
            self.top_k.setValue(int(rcfg.search.top_k))
            self.recall_k.setValue(int(rcfg.search.recall_k))
            self.rerank_enabled.setChecked(bool(rcfg.rerank.enabled))
            self.caption_enabled.setChecked(bool(rcfg.caption.enabled))
        except Exception:
            self.top_k.setValue(12)
            self.recall_k.setValue(50)
            self.rerank_enabled.setChecked(True)
            self.caption_enabled.setChecked(True)

    def on_save_params(self) -> None:
        import yaml

        from .config import PROJECT_ROOT

        path = PROJECT_ROOT / "configs" / "agent.yaml"
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            data.setdefault("agent", {})
            data["agent"]["similarity_threshold"] = round(self.threshold.value(), 2)
            data["agent"]["max_retries"] = self.retries.value()
            data.setdefault("memory", {})
            data["memory"]["max_sessions"] = self.max_sessions.value()
            path.write_text(
                yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )

            cpath = PROJECT_ROOT / "config.yaml"
            cdata = yaml.safe_load(cpath.read_text(encoding="utf-8")) or {}
            cdata.setdefault("search", {})
            cdata["search"]["top_k"] = self.top_k.value()
            cdata["search"]["recall_k"] = self.recall_k.value()
            cdata.setdefault("rerank", {})
            cdata["rerank"]["enabled"] = self.rerank_enabled.isChecked()
            cdata.setdefault("caption", {})
            cdata["caption"]["enabled"] = self.caption_enabled.isChecked()
            cpath.write_text(
                yaml.safe_dump(cdata, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )

            self.params_status.setText("已保存并即时生效。")
            self.config_changed.emit()
        except Exception as e:
            self.params_status.setText(f"保存失败：{e}")

    # ── 向量库 ────────────────────────────────────────────────────────────

    def _load_vector(self) -> None:
        try:
            from .retriever import Config

            cfg = Config.from_yaml()
            self.collection_edit.setText(str(cfg.vector_store.collection))
            backend = str(cfg.vector_store.backend).lower()
            if backend in ("chroma", "chromadb"):
                self.backend_box.setCurrentText("chroma")
            else:
                self.backend_box.setCurrentText("qdrant")
            self.vector_info.setText(f"配置文件：{cfg.vector_store.backend}")
        except Exception as e:
            self.vector_info.setText(f"读取失败：{e}")

    def on_apply_vector(self) -> None:
        import yaml

        from .config import PROJECT_ROOT

        backend = self.backend_box.currentText()
        collection = self.collection_edit.text().strip() or "images"
        try:
            path = PROJECT_ROOT / "config.yaml"
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            data.setdefault("vector_store", {})
            data["vector_store"]["backend"] = backend
            data["vector_store"]["collection"] = collection
            path.write_text(
                yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
        except Exception as e:
            self.vector_status.setText(f"写入配置失败：{e}")
            return

        if self.agent is None:
            self.vector_status.setText("已保存，重启应用后生效。")
            return

        tool = self.agent.tools.get("switch_vector_backend")
        if tool is None:
            self.vector_status.setText("已保存，重启应用后生效。")
            return

        try:
            result = tool.run(backend)
            self.vector_status.setText(str(result.get("summary", "已切换")))
        except Exception as e:
            self.vector_status.setText(f"切换失败：{e}")

    # ── 目录与安全 ────────────────────────────────────────────────────────

    def _load_paths(self) -> None:
        try:
            from .config import AgentConfig

            cfg = AgentConfig()
            for root in cfg.image_roots:
                self.roots_list.addItem(str(root))
        except Exception:
            pass

    def on_add_root(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择允许访问的目录")
        if folder:
            self.roots_list.addItem(folder)

    def on_del_root(self) -> None:
        row = self.roots_list.currentRow()
        if row >= 0:
            self.roots_list.takeItem(row)

    def on_save_paths(self) -> None:
        import yaml

        from .config import PROJECT_ROOT

        roots = [
            self.roots_list.item(i).text().strip()
            for i in range(self.roots_list.count())
            if self.roots_list.item(i).text().strip()
        ]
        try:
            path = PROJECT_ROOT / "configs" / "agent.yaml"
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            data.setdefault("sandbox", {})
            data["sandbox"]["image_roots"] = roots
            path.write_text(
                yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            self.paths_status.setText("已保存并即时生效。")
            self.config_changed.emit()
        except Exception as e:
            self.paths_status.setText(f"保存失败：{e}")
