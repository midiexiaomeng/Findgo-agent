"""设置对话框：左侧搜索与分类导航，右侧具体设置。

无边框窗口，标题栏可拖拽，四边四角可拉伸。
左侧顶部的搜索框可以按关键词定位到具体页面。
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
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
from .ui.input_panel import ResizeHandle
from .ui.title_bar import TitleBar

# 搜索关键词：页面名 + 该页能调的主要设置项，用起来比只看标题快
PAGE_SEARCH_TERMS = {
    0: ["模型", "服务商", "api", "key", "deepseek", "openai", "ollama", "接口", "测试连接"],
    1: [
        "检索参数",
        "阈值",
        "相似度",
        "重试",
        "返回数",
        "召回",
        "重排",
        "候选",
        "温度",
        "描述",
        "会话",
    ],
    2: ["向量库", "qdrant", "chroma", "后端", "集合", "索引"],
    3: ["运行环境", "设备", "cuda", "cpu", "gpu", "日志", "离线", "预热", "显存"],
    4: ["目录", "安全", "沙箱", "素材", "授权", "导出", "路径"],
    5: ["暂存区", "删除", "回收", "预览", "暂存", "留存", "容量"],
    6: [
        "高级",
        "维度",
        "embedding",
        "clip",
        "blip",
        "远端",
        "视觉",
        "向量维度",
    ],
    7: ["自定义工具", "动态工具", "工具", "脚本", "保存"],
    8: ["自检", "诊断", "环境", "检测", "依赖", "显卡"],
    9: ["关于", "版本", "路径", "重置", "默认"],
}


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
    PAGE_ENV = 3
    PAGE_PATHS = 4
    PAGE_TRASH = 5
    PAGE_ADVANCED = 6
    PAGE_CUSTOM = 7
    PAGE_DIAG = 8
    PAGE_ABOUT = 9

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
        self._resizing = False
        self._resize_handles: dict[str, ResizeHandle] = {}

        self.setWindowTitle("设置")
        self.resize(680, 520)
        self.setMinimumSize(560, 420)
        self.setObjectName("SettingsDialog")
        # 全局 QSS 给 QDialog 设了不透明背景，这里单独改成透明，让容器圆角露出来
        self.setStyleSheet(DIALOG_QSS + "\n#SettingsDialog{background:transparent;}")

        # 无边框：透明背景 + 容器圆角，四边四角挂可拖拽手柄
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self._build_ui()
        self._install_resize_handles()
        self.nav.setCurrentRow(page)
        self._on_provider_changed()
        self._refresh_list()
        self._load_params()
        self._load_vector()
        self._load_env()
        self._load_paths()
        self._load_trash()
        self._load_advanced()
        self._load_custom_tools()

    # ── 界面骨架 ──────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.container = QWidget()
        self.container.setObjectName("settingsFrame")
        self.container.setStyleSheet("#settingsFrame{background:#FBFAF7;border-radius:12px;}")
        outer.addWidget(self.container)

        inner = QVBoxLayout(self.container)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(0)

        self.title_bar = TitleBar(self, show_panels=False, title_text="设置")
        inner.addWidget(self.title_bar)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # 左列：搜索框 + 导航
        left = QWidget()
        left.setFixedWidth(168)
        left.setStyleSheet("background:#F8F7F3;border-bottom-left-radius:12px;")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索设置…")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setStyleSheet(
            "QLineEdit{border:1px solid #E5E3DD;border-radius:8px;background:#FFFFFF;"
            "padding:6px 10px;font-size:12.5px;color:#2E2E2E;}"
            "QLineEdit:focus{border:1px solid #A8C9B8;}"
        )
        self.search_edit.textChanged.connect(self._on_search)
        self.search_edit.returnPressed.connect(self._on_search_enter)
        left_layout.addWidget(self.search_edit)

        self.nav = QListWidget()
        self.nav.setStyleSheet(
            "QListWidget{border:none;background:transparent;padding:0;font-size:13px;}"
            "QListWidget::item{padding:9px 12px;border-radius:6px;margin:2px 0;}"
            "QListWidget::item:selected{background:#E3EFE7;color:#2E2E2E;}"
        )
        self.nav.addItems(
            [
                "添加模型",
                "检索参数",
                "向量库",
                "运行环境",
                "目录与安全",
                "暂存区",
                "高级",
                "自定义工具",
                "环境自检",
                "关于",
            ]
        )
        self.nav.currentRowChanged.connect(self._on_nav_changed)
        left_layout.addWidget(self.nav, 1)
        body.addWidget(left)

        self.stack = QStackedWidget()
        self.stack.setStyleSheet("QStackedWidget{background:#FBFAF7;}")
        self.stack.addWidget(self._build_models_page())
        self.stack.addWidget(self._build_params_page())
        self.stack.addWidget(self._build_vector_page())
        self.stack.addWidget(self._build_env_page())
        self.stack.addWidget(self._build_paths_page())
        self.stack.addWidget(self._build_trash_page())
        self.stack.addWidget(self._build_advanced_page())
        self.stack.addWidget(self._build_custom_page())
        self.stack.addWidget(self._build_diag_page())
        self.stack.addWidget(self._build_about_page())
        body.addWidget(self.stack, 1)

        inner.addLayout(body, 1)

    # ── 无边框拖拽 ────────────────────────────────────────────────────────

    def _install_resize_handles(self) -> None:
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

    def _layout_handles(self) -> None:
        if not self._resize_handles or self._resizing:
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

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._layout_handles()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # 拖动过程中不重排，否则鼠标会脱离手柄
        if not hasattr(self, "_handle_timer"):
            self._handle_timer = QTimer(self)
            self._handle_timer.setSingleShot(True)
            self._handle_timer.timeout.connect(self._layout_handles)
        self._handle_timer.start(30)

    def toggle_maximize(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    # ── 设置搜索 ──────────────────────────────────────────────────────────

    def _on_search(self, text: str) -> None:
        query = text.strip().lower()
        first_visible = -1

        for row in range(self.nav.count()):
            if not query:
                self.nav.setRowHidden(row, False)
                if first_visible < 0:
                    first_visible = row
                continue

            terms = PAGE_SEARCH_TERMS.get(row, [])
            name = self.nav.item(row).text()
            hit = query in name.lower() or any(query in str(t).lower() for t in terms)

            self.nav.setRowHidden(row, not hit)
            if hit and first_visible < 0:
                first_visible = row

        if query and first_visible >= 0:
            self.nav.setCurrentRow(first_visible)

    def _on_search_enter(self) -> None:
        """回车跳到第一个匹配项；无匹配时给个提示。"""
        query = self.search_edit.text().strip()
        if not query:
            return

        for row in range(self.nav.count()):
            if not self.nav.isRowHidden(row):
                self.nav.setCurrentRow(row)
                return

        self.search_edit.setPlaceholderText("没有匹配的设置项")

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

        self.rerank_candidates = QSpinBox()
        self.rerank_candidates.setRange(4, 64)
        self.rerank_candidates.setToolTip(
            "送进大模型做语义重排的候选数量。显存和上下文够就调高，重排质量更好但更慢"
        )
        form.addRow("重排候选数", self.rerank_candidates)

        self.rerank_temperature = QDoubleSpinBox()
        self.rerank_temperature.setRange(0.0, 1.5)
        self.rerank_temperature.setSingleStep(0.1)
        self.rerank_temperature.setDecimals(2)
        self.rerank_temperature.setToolTip("重排要稳定可复现，一般保持 0")
        form.addRow("重排温度", self.rerank_temperature)

        self.rerank_max_tokens = QSpinBox()
        self.rerank_max_tokens.setRange(64, 4096)
        self.rerank_max_tokens.setSingleStep(64)
        self.rerank_max_tokens.setToolTip("单次重排请求的最大输出长度")
        form.addRow("重排最大输出", self.rerank_max_tokens)

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

    # ── 运行环境页 ────────────────────────────────────────────────────────

    def _build_env_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        tip = QLabel(
            "设备指模型跑在哪个硬件上。显存紧张时给 CLIP 选 CPU，"
            "查询编码只慢几十毫秒，把显存留给重排模型。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#8A8A8A; font-size:12px;")
        layout.addWidget(tip)

        form = QFormLayout()
        form.setSpacing(10)

        self.device_box = QComboBox()
        self.device_box.addItem("自动选择", "auto")
        self.device_box.addItem("GPU (CUDA)", "cuda")
        self.device_box.addItem("CPU", "cpu")
        self.device_box.setToolTip("没有可用显卡时自动回落 CPU")
        form.addRow("推理设备", self.device_box)

        self.log_level_box = QComboBox()
        self.log_level_box.addItems(["INFO", "DEBUG", "WARNING", "ERROR"])
        self.log_level_box.setToolTip("DEBUG 会把每次请求与耗时写进日志，排查时用")
        form.addRow("日志等级", self.log_level_box)

        self.offline_box = QCheckBox("禁止一切外网调用")
        self.offline_box.setToolTip(
            "开启后：不调云端 LLM、不联网搜图、不下载模型。本地模型与检索照常工作"
        )
        form.addRow("离线模式", self.offline_box)

        self.warmup_box = QCheckBox("启动时后台加载 CLIP")
        self.warmup_box.setToolTip("预热后首次检索更快；不常用检索时可关掉省显存")
        form.addRow("模型预热", self.warmup_box)

        layout.addLayout(form)

        save_btn = QPushButton("保存设置")
        save_btn.setStyleSheet(
            "QPushButton{background:#6F9C86;color:white;border:none;"
            "border-radius:6px;padding:7px 18px;}"
        )
        save_btn.clicked.connect(self.on_save_env)
        layout.addWidget(save_btn, 0, Qt.AlignLeft)

        self.env_status = QLabel("")
        self.env_status.setStyleSheet("color:#8A8A8A; font-size:12px;")
        self.env_status.setWordWrap(True)
        layout.addWidget(self.env_status)
        layout.addStretch(1)
        return page

    # ── 高级页 ────────────────────────────────────────────────────────────

    def _build_advanced_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        # ── 嵌入维度 ──
        dim_title = QLabel("向量维度")
        dim_title.setStyleSheet("font-weight:600;color:#4A4A4A;font-size:13px;")
        layout.addWidget(dim_title)

        dim_tip = QLabel("改维度会让现有索引全部作废，必须重建。改之前先确认自己能接受重建耗时。")
        dim_tip.setWordWrap(True)
        dim_tip.setStyleSheet("color:#C0504D; font-size:12px;")
        layout.addWidget(dim_tip)

        dim_row = QHBoxLayout()
        self.dim_spin = QSpinBox()
        self.dim_spin.setRange(64, 4096)
        self.dim_spin.setSingleStep(64)
        self.dim_spin.setFixedWidth(110)
        self.dim_spin.setToolTip("Chinese-CLIP base 固定输出 512 维，改了就对不上")
        dim_row.addWidget(self.dim_spin)
        self.dim_btn = QPushButton("修改维度")
        self.dim_btn.clicked.connect(self.on_change_dim)
        dim_row.addWidget(self.dim_btn)
        dim_row.addStretch(1)
        layout.addLayout(dim_row)

        sep = QLabel("")
        sep.setFixedHeight(1)
        sep.setStyleSheet("background:#E8E6E0;")
        layout.addWidget(sep)

        # ── CLIP 后端 ──
        clip_title = QLabel("图像编码（CLIP）")
        clip_title.setStyleSheet("font-weight:600;color:#4A4A4A;font-size:13px;")
        layout.addWidget(clip_title)

        clip_tip = QLabel(
            "默认用本地 Chinese-CLIP，不联网。也可以接兼容 OpenAI /v1/embeddings 的远端服务，"
            "省掉 1.6GB 模型下载。远端服务只要不是同一个模型，向量就对不上，需要重建索引。"
        )
        clip_tip.setWordWrap(True)
        clip_tip.setStyleSheet("color:#8A8A8A; font-size:12px;")
        layout.addWidget(clip_tip)

        clip_form = QFormLayout()
        clip_form.setSpacing(8)

        self.clip_backend_box = QComboBox()
        self.clip_backend_box.addItem("本地模型", "local")
        self.clip_backend_box.addItem("远端 API", "api")
        self.clip_backend_box.currentIndexChanged.connect(self._on_clip_backend_changed)
        clip_form.addRow("后端", self.clip_backend_box)

        self.clip_api_url = QLineEdit()
        self.clip_api_url.setPlaceholderText("https://api.example.com/v1")
        clip_form.addRow("接口地址", self.clip_api_url)

        self.clip_api_key = QLineEdit()
        self.clip_api_key.setEchoMode(QLineEdit.Password)
        self.clip_api_key.setPlaceholderText("sk-...")
        clip_form.addRow("API Key", self.clip_api_key)

        self.clip_api_model = QLineEdit()
        self.clip_api_model.setPlaceholderText("模型名，例如 clip-vit-large")
        clip_form.addRow("模型名", self.clip_api_model)

        layout.addLayout(clip_form)

        sep2 = QLabel("")
        sep2.setFixedHeight(1)
        sep2.setStyleSheet("background:#E8E6E0;")
        layout.addWidget(sep2)

        # ── BLIP 后端 ──
        blip_title = QLabel("图像描述（BLIP）")
        blip_title.setStyleSheet("font-weight:600;color:#4A4A4A;font-size:13px;")
        layout.addWidget(blip_title)

        blip_tip = QLabel(
            "默认用本地 BLIP，输出英文描述。接视觉模型 API 可以直接得到中文描述，"
            "语义重排会更准，但每张图都要联网。"
        )
        blip_tip.setWordWrap(True)
        blip_tip.setStyleSheet("color:#8A8A8A; font-size:12px;")
        layout.addWidget(blip_tip)

        blip_form = QFormLayout()
        blip_form.setSpacing(8)

        self.blip_backend_box = QComboBox()
        self.blip_backend_box.addItem("本地模型", "local")
        self.blip_backend_box.addItem("视觉 API", "api")
        self.blip_backend_box.currentIndexChanged.connect(self._on_blip_backend_changed)
        blip_form.addRow("后端", self.blip_backend_box)

        self.blip_api_url = QLineEdit()
        self.blip_api_url.setPlaceholderText("https://api.example.com/v1")
        blip_form.addRow("接口地址", self.blip_api_url)

        self.blip_api_key = QLineEdit()
        self.blip_api_key.setEchoMode(QLineEdit.Password)
        self.blip_api_key.setPlaceholderText("sk-...")
        blip_form.addRow("API Key", self.blip_api_key)

        self.blip_api_model = QLineEdit()
        self.blip_api_model.setPlaceholderText("支持视觉的模型名，例如 gpt-4o-mini")
        blip_form.addRow("模型名", self.blip_api_model)

        layout.addLayout(blip_form)

        save_btn = QPushButton("保存高级设置")
        save_btn.setStyleSheet(
            "QPushButton{background:#6F9C86;color:white;border:none;"
            "border-radius:6px;padding:7px 18px;}"
        )
        save_btn.clicked.connect(self.on_save_advanced)
        layout.addWidget(save_btn, 0, Qt.AlignLeft)

        self.advanced_status = QLabel("")
        self.advanced_status.setStyleSheet("color:#8A8A8A; font-size:12px;")
        self.advanced_status.setWordWrap(True)
        layout.addWidget(self.advanced_status)
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

        export_form = QFormLayout()
        export_form.setSpacing(8)
        export_row = QHBoxLayout()
        self.export_edit = QLineEdit()
        self.export_edit.setPlaceholderText("./data/exports")
        browse_btn = QPushButton("浏览…")
        browse_btn.clicked.connect(self.on_browse_export)
        export_row.addWidget(self.export_edit, 1)
        export_row.addWidget(browse_btn)
        export_form.addRow("导出目录", export_row)
        layout.addLayout(export_form)

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

        reset_row = QHBoxLayout()
        reset_btn = QPushButton("重置为默认设置")
        reset_btn.setStyleSheet(
            "QPushButton{background:#fff;color:#C0504D;border:1px solid #E5C4C2;"
            "border-radius:6px;padding:7px 16px;}"
        )
        reset_btn.clicked.connect(self.on_reset_defaults)
        reset_row.addWidget(reset_btn)
        reset_row.addStretch(1)
        layout.addLayout(reset_row)

        self.about_status = QLabel("")
        self.about_status.setStyleSheet("color:#8A8A8A; font-size:12px;")
        self.about_status.setWordWrap(True)
        layout.addWidget(self.about_status)

        return page

    def on_reset_defaults(self) -> None:
        """只重置设置页里能改的字段，不动模型列表与目录白名单。"""
        reply = QMessageBox.question(
            self,
            "重置为默认设置",
            "会把检索参数、运行环境、导出目录与模型后端恢复为默认值。\n"
            "已添加的模型、素材目录、索引数据都不会动。\n\n"
            "确定继续吗？",
            QMessageBox.Cancel | QMessageBox.Yes,
            QMessageBox.Cancel,
        )
        if reply != QMessageBox.Yes:
            return

        import yaml

        from .config import PROJECT_ROOT

        try:
            cpath = PROJECT_ROOT / "config.yaml"
            if cpath.exists():
                backup = cpath.with_suffix(".yaml.bak")
                backup.write_text(cpath.read_text(encoding="utf-8"), encoding="utf-8")

                data = yaml.safe_load(cpath.read_text(encoding="utf-8")) or {}
                data["device"] = "auto"
                data["offline"] = False
                data["warmup_on_start"] = True

                data.setdefault("app", {})
                data["app"]["log_level"] = "INFO"

                data.setdefault("search", {})
                data["search"]["top_k"] = 12
                data["search"]["recall_k"] = 50

                data.setdefault("rerank", {})
                data["rerank"]["enabled"] = True
                data["rerank"]["max_candidates"] = 16
                data["rerank"]["temperature"] = 0.0
                data["rerank"]["max_tokens"] = 512

                data.setdefault("caption", {})
                data["caption"]["enabled"] = True
                data["caption"]["backend"] = "local"

                data.setdefault("embedding", {})
                if int(data["embedding"].get("embedding_dim", 512)) != 512:
                    data["embedding"]["embedding_dim"] = 512
                data["embedding"]["backend"] = "local"

                data.setdefault("paths", {})
                data["paths"]["export_dir"] = "./data/exports"

                cpath.write_text(
                    yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                    encoding="utf-8",
                )

            apath = PROJECT_ROOT / "configs" / "agent.yaml"
            if apath.exists():
                adata = yaml.safe_load(apath.read_text(encoding="utf-8")) or {}
                adata.setdefault("agent", {})
                adata["agent"]["similarity_threshold"] = 0.25
                adata["agent"]["max_retries"] = 2
                adata.setdefault("memory", {})
                adata["memory"]["max_sessions"] = 20
                apath.write_text(
                    yaml.safe_dump(adata, allow_unicode=True, sort_keys=False),
                    encoding="utf-8",
                )

            self._load_params()
            self._load_env()
            self._load_paths()
            self._load_advanced()
            self.about_status.setText("已重置。原 config.yaml 已备份为 config.yaml.bak。")
            self.config_changed.emit()
        except Exception as e:
            self.about_status.setText(f"重置失败：{e}")

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

    # ── 运行环境 ──────────────────────────────────────────────────────────

    def _load_env(self) -> None:
        from .retriever import Config

        try:
            cfg = Config.from_yaml()
            device = str(cfg.device or "auto").lower()
            if device.startswith("cuda"):
                self.device_box.setCurrentIndex(1)
            elif device == "cpu":
                self.device_box.setCurrentIndex(2)
            else:
                self.device_box.setCurrentIndex(0)
            self.log_level_box.setCurrentText(str(cfg.app.log_level or "INFO").upper())
            self.offline_box.setChecked(bool(cfg.offline))
            self.warmup_box.setChecked(bool(cfg.warmup_on_start))
        except Exception as e:
            self.env_status.setText(f"读取失败：{e}")

    def on_save_env(self) -> None:
        import yaml

        from .config import PROJECT_ROOT

        path = PROJECT_ROOT / "config.yaml"
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            data["device"] = self.device_box.currentData()
            data["offline"] = self.offline_box.isChecked()
            data["warmup_on_start"] = self.warmup_box.isChecked()
            data.setdefault("app", {})
            data["app"]["log_level"] = self.log_level_box.currentText()
            path.write_text(
                yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )

            hint = "已保存，设备与日志等级下次启动生效。"
            if data["offline"]:
                hint += "离线模式已开启，云端重排与联网搜图会被跳过。"
            self.env_status.setText(hint)
            self.config_changed.emit()
        except Exception as e:
            self.env_status.setText(f"保存失败：{e}")

    # ── 高级页 ────────────────────────────────────────────────────────────

    def _load_advanced(self) -> None:
        from .retriever import Config

        try:
            cfg = Config.from_yaml()
            self.dim_spin.setValue(int(cfg.embedding.embedding_dim))

            backend = str(cfg.embedding.backend or "local").lower()
            self.clip_backend_box.setCurrentIndex(1 if backend == "api" else 0)
            self.clip_api_url.setText(str(cfg.embedding.api_base_url or ""))
            self.clip_api_key.setText(str(cfg.embedding.api_key or ""))
            self.clip_api_model.setText(str(cfg.embedding.api_model or ""))

            cbackend = str(cfg.caption.backend or "local").lower()
            self.blip_backend_box.setCurrentIndex(1 if cbackend == "api" else 0)
            self.blip_api_url.setText(str(cfg.caption.api_base_url or ""))
            self.blip_api_key.setText(str(cfg.caption.api_key or ""))
            self.blip_api_model.setText(str(cfg.caption.api_model or ""))

            self._on_clip_backend_changed(self.clip_backend_box.currentIndex())
            self._on_blip_backend_changed(self.blip_backend_box.currentIndex())
        except Exception as e:
            self.advanced_status.setText(f"读取失败：{e}")

    def _on_clip_backend_changed(self, index: int) -> None:
        self.clip_api_url.setEnabled(index == 1)
        self.clip_api_key.setEnabled(index == 1)
        self.clip_api_model.setEnabled(index == 1)

    def _on_blip_backend_changed(self, index: int) -> None:
        self.blip_api_url.setEnabled(index == 1)
        self.blip_api_key.setEnabled(index == 1)
        self.blip_api_model.setEnabled(index == 1)

    def on_change_dim(self) -> None:
        from .retriever import Config

        try:
            current = int(Config.from_yaml().embedding.embedding_dim)
        except Exception:
            current = 0
        new_dim = self.dim_spin.value()
        if new_dim == current:
            self.advanced_status.setText("维度没有变化。")
            return

        box = QMessageBox(self)
        box.setWindowTitle("确认修改向量维度")
        box.setIcon(QMessageBox.Warning)
        box.setText(
            f"将把向量维度从 {current} 改为 {new_dim}。\n\n"
            "现有索引会全部失效，必须重建才能检索；"
            "重建期间无法搜索，耗时取决于图片数量。"
        )
        box.setInformativeText("确实要继续吗？")
        box.setStandardButtons(QMessageBox.Cancel | QMessageBox.Yes)
        box.button(QMessageBox.Yes).setText("我已了解，继续修改")
        box.button(QMessageBox.Cancel).setText("取消")
        box.setDefaultButton(QMessageBox.Cancel)

        if box.exec_() != QMessageBox.Yes:
            self.dim_spin.setValue(current)
            self.advanced_status.setText("已取消，维度未修改。")
            return

        import yaml

        from .config import PROJECT_ROOT

        path = PROJECT_ROOT / "config.yaml"
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            data.setdefault("embedding", {})
            data["embedding"]["embedding_dim"] = new_dim
            path.write_text(
                yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            self.advanced_status.setText(
                f"维度已改为 {new_dim}。请重新建索引，否则检索不到旧数据。"
            )
            self.config_changed.emit()
        except Exception as e:
            self.advanced_status.setText(f"写入失败：{e}")

    def on_save_advanced(self) -> None:
        import yaml

        from .config import PROJECT_ROOT

        clip_api = self.clip_backend_box.currentData() == "api"
        blip_api = self.blip_backend_box.currentData() == "api"

        if clip_api and not self.clip_api_url.text().strip():
            self.advanced_status.setText("CLIP 选了远端 API，但没填接口地址。")
            return
        if blip_api and not self.blip_api_url.text().strip():
            self.advanced_status.setText("BLIP 选了视觉 API，但没填接口地址。")
            return

        path = PROJECT_ROOT / "config.yaml"
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

            data.setdefault("embedding", {})
            data["embedding"]["backend"] = "api" if clip_api else "local"
            data["embedding"]["api_base_url"] = self.clip_api_url.text().strip()
            data["embedding"]["api_key"] = self.clip_api_key.text().strip()
            data["embedding"]["api_model"] = self.clip_api_model.text().strip()

            data.setdefault("caption", {})
            data["caption"]["backend"] = "api" if blip_api else "local"
            data["caption"]["api_base_url"] = self.blip_api_url.text().strip()
            data["caption"]["api_key"] = self.blip_api_key.text().strip()
            data["caption"]["api_model"] = self.blip_api_model.text().strip()

            path.write_text(
                yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            self.advanced_status.setText("已保存。后端切换后建议重建索引。")
            self.config_changed.emit()
        except Exception as e:
            self.advanced_status.setText(f"保存失败：{e}")

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
            self.rerank_candidates.setValue(int(rcfg.rerank.max_candidates))
            self.rerank_temperature.setValue(float(rcfg.rerank.temperature))
            self.rerank_max_tokens.setValue(int(rcfg.rerank.max_tokens))
            self.caption_enabled.setChecked(bool(rcfg.caption.enabled))
        except Exception:
            self.top_k.setValue(12)
            self.recall_k.setValue(50)
            self.rerank_enabled.setChecked(True)
            self.rerank_candidates.setValue(16)
            self.rerank_temperature.setValue(0.0)
            self.rerank_max_tokens.setValue(512)
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
            cdata["rerank"]["max_candidates"] = self.rerank_candidates.value()
            cdata["rerank"]["temperature"] = round(self.rerank_temperature.value(), 2)
            cdata["rerank"]["max_tokens"] = self.rerank_max_tokens.value()
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

        try:
            from .retriever import Config

            self.export_edit.setText(str(Config.from_yaml().paths.export_dir))
        except Exception:
            self.export_edit.setText("./data/exports")

    def on_browse_export(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if folder:
            self.export_edit.setText(folder)

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

            cpath = PROJECT_ROOT / "config.yaml"
            cdata = yaml.safe_load(cpath.read_text(encoding="utf-8")) or {}
            cdata.setdefault("paths", {})
            export_dir = self.export_edit.text().strip() or "./data/exports"
            cdata["paths"]["export_dir"] = export_dir
            cpath.write_text(
                yaml.safe_dump(cdata, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )

            self.paths_status.setText("已保存并即时生效。")
            self.config_changed.emit()
        except Exception as e:
            self.paths_status.setText(f"保存失败：{e}")
