"""输入区左下角的命令菜单、斜杠候选与权限切换按钮。"""

from __future__ import annotations

from PyQt5.QtCore import QPoint, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QMenu,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# 命令定义：图标 + 命令名 + 中文说明 + 选中后填入输入框的内容
COMMANDS = [
    {
        "icon": "📖",
        "name": "/diary",
        "desc": "写今日日记",
        "prompt": "帮我写今天的日记，把今天的检索记录和对话要点整理进去。",
    },
    {
        "icon": "💎",
        "name": "/xing",
        "desc": "从对话中提炼技能",
        "prompt": "从我们刚才的对话里提炼一个可复用的技能，写成 SKILL 描述。",
    },
    {
        "icon": "↗",
        "name": "/compact",
        "desc": "压缩上下文",
        "prompt": "把当前会话的上下文压缩成摘要，保留关键约束与结论。",
    },
    {
        "icon": "🔁",
        "name": "/loop",
        "desc": "在当前会话循环执行任务",
        "prompt": "在当前会话里循环执行这个任务，直到我让你停：",
    },
    {
        "icon": "✧",
        "name": "/hana-plugin-creator",
        "desc": "Create Hana plugin scaffol…",
        "prompt": "帮我创建一个 Hana 插件脚手架。",
    },
    {
        "icon": "✧",
        "name": "/office-documents",
        "desc": "Use when the user asks to o…",
        "prompt": "帮我处理这个 Office 文档。",
    },
]

# 权限四态：符号、名称、颜色、说明
PERMISSIONS = [
    {
        "key": "full",
        "symbol": ">_",
        "name": "完整权限",
        "color": "#C0504D",
        "tip": "Agent 可以执行任何工具，包括写文件与删除",
    },
    {
        "key": "auto",
        "symbol": "//",
        "name": "自动审批",
        "color": "#4A90C2",
        "tip": "写操作自动通过，危险操作仍需确认",
    },
    {
        "key": "ask",
        "symbol": "？",
        "name": "操作前询问",
        "color": "#6B6B6B",
        "tip": "每次调用写类工具前都弹窗确认",
    },
    {
        "key": "readonly",
        "symbol": "[|]",
        "name": "只读模式",
        "color": "#C99BA6",
        "tip": "只允许检索与读取，禁止一切写入",
    },
]


class CommandMenu(QMenu):
    """点击命令按钮弹出的菜单，每项是图标 + 命令 + 说明。"""

    command_picked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet(
            "QMenu{background:#FFFFFF;border:1px solid #E5E3DD;border-radius:10px;padding:6px;}"
            "QMenu::item{padding:7px 10px;border-radius:7px;}"
            "QMenu::item:selected{background:#E3EFE7;}"
        )
        self._build()

    def _build(self) -> None:
        for cmd in COMMANDS:
            action = self.addAction(self._item_text(cmd))
            action.setData(cmd["name"])
            action.triggered.connect(lambda _=False, n=cmd["name"]: self.command_picked.emit(n))

    @staticmethod
    def _item_text(cmd: dict) -> str:
        # QMenu 不支持富文本，用对齐空格模拟左右两列
        return f"{cmd['icon']}  {cmd['name']:<24}{cmd['desc']}"


class SlashPopup(QListWidget):
    """输入 `/` 时浮在输入框上方的候选列表。"""

    command_picked = pyqtSignal(str)

    ROW_HEIGHT = 34

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setStyleSheet(
            "QListWidget{background:#FFFFFF;border:1px solid #E5E3DD;border-radius:10px;"
            "padding:6px;outline:none;}"
            "QListWidget::item{padding:0px;border-radius:7px;}"
            "QListWidget::item:selected{background:#E3EFE7;}"
            "QListWidget::item:hover{background:#F2F0E9;}"
        )
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.itemClicked.connect(self._on_click)
        self._rows: list[dict] = []

    # ── 填充 ──

    def fill(self, items: list[dict]) -> None:
        self.clear()
        self._rows = list(items)
        for cmd in items:
            item = QListWidgetItem()
            item.setSizeHint(QSize(0, self.ROW_HEIGHT))
            self.addItem(item)
            self.setItemWidget(item, CommandRow(cmd))
        if self.count():
            self.setCurrentRow(0)

    def is_empty(self) -> bool:
        return self.count() == 0

    def current_command(self) -> str:
        row = self.currentRow()
        if 0 <= row < len(self._rows):
            return self._rows[row]["name"]
        return ""

    def move_selection(self, step: int) -> None:
        if not self.count():
            return
        row = (self.currentRow() + step) % self.count()
        self.setCurrentRow(row)

    def _on_click(self, item: QListWidgetItem) -> None:
        row = self.row(item)
        if 0 <= row < len(self._rows):
            self.command_picked.emit(self._rows[row]["name"])


class CommandRow(QWidget):
    """一行：左侧图标、中间命令名、右侧中文说明。"""

    def __init__(self, cmd: dict, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        from PyQt5.QtWidgets import QHBoxLayout, QLabel

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 0, 12, 0)
        row.setSpacing(10)

        icon = QLabel(cmd["icon"])
        icon.setFixedWidth(20)
        icon.setStyleSheet("font-size:14px;")

        name = QLabel(cmd["name"])
        name_font = QFont()
        name_font.setPointSize(9)
        name_font.setBold(True)
        name.setFont(name_font)
        name.setStyleSheet("color:#2E2E2E;")

        desc = QLabel(cmd["desc"])
        desc.setStyleSheet("color:#8A8A8A; font-size:12px;")
        desc.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        row.addWidget(icon)
        row.addWidget(name)
        row.addStretch(1)
        row.addWidget(desc)

    def mousePressEvent(self, event) -> None:
        # 点击行本身交给 QListWidget 处理
        event.ignore()


class PermissionMenu(QMenu):
    """权限选择菜单：四项带符号、名称与说明，当前项打勾。"""

    picked = pyqtSignal(str)

    def __init__(self, current: str, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet(
            "QMenu{background:#FFFFFF;border:1px solid #E5E3DD;border-radius:10px;padding:6px;}"
            "QMenu::item{padding:8px 14px;border-radius:7px;}"
            "QMenu::item:selected{background:#E3EFE7;}"
            "QMenu::separator{height:1px;background:#EFEDE7;margin:4px 8px;}"
        )
        self._current = current
        self._build()

    def _build(self) -> None:
        for p in PERMISSIONS:
            mark = "● " if p["key"] == self._current else "   "
            action = self.addAction(f"{mark}{p['symbol']}   {p['name']}")
            action.setData(p["key"])
            action.setToolTip(p["tip"])
            action.triggered.connect(lambda _=False, k=p["key"]: self.picked.emit(k))


class PermissionButton(QToolButton):
    """点击弹权限菜单，按钮文字就是当前等级符号。"""

    changed = pyqtSignal(str)

    def __init__(self, current: str = "ask", parent=None):
        super().__init__(parent)
        self.setFixedSize(34, 28)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.setAutoRaise(True)

        self._index = self._index_of(current)
        self._refresh()

    @staticmethod
    def _index_of(key: str) -> int:
        for i, p in enumerate(PERMISSIONS):
            if p["key"] == key:
                return i
        return 2

    @property
    def current(self) -> dict:
        return PERMISSIONS[self._index]

    def key(self) -> str:
        return self.current["key"]

    def _refresh(self) -> None:
        p = self.current
        self.setText(p["symbol"])
        color = p["color"]
        self.setStyleSheet(
            f"QToolButton{{border:1px solid #EDEBE6;border-radius:8px;background:#FFFFFF;"
            f"color:{color};font-size:11px;font-weight:600;padding:2px 4px;}}"
            f"QToolButton:hover{{background:#F2F0EA;}}"
        )
        self.setToolTip(f"{p['name']}：{p['tip']}\n点击切换")

    def open_menu(self) -> None:
        menu = PermissionMenu(self.key(), self)
        menu.picked.connect(self.set_key)
        menu.exec_(self.mapToGlobal(QPoint(0, self.height() + 4)))

    def set_key(self, key: str) -> None:
        index = self._index_of(key)
        if index == self._index:
            return
        self._index = index
        self._refresh()
        self.changed.emit(self.key())

    def next(self) -> None:
        """保留循环接口，供快捷键或其他入口使用。"""
        self.set_key(PERMISSIONS[(self._index + 1) % len(PERMISSIONS)]["key"])

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.open_menu()
            return
        super().mousePressEvent(event)


def match_commands(text: str) -> list[dict]:
    """按 `/` 后面的内容过滤命令，text 为空时返回全部。"""
    query = text.strip().lstrip("/").lower()
    if not query:
        return list(COMMANDS)
    return [
        c for c in COMMANDS if query in c["name"].lstrip("/").lower() or query in c["desc"].lower()
    ]


def command_by_name(name: str) -> dict | None:
    for c in COMMANDS:
        if c["name"] == name:
            return c
    return None


def popup_position(widget, cursor_rect) -> QPoint:
    """把候选列表放在光标上方，空间不够时放下方。"""
    global_pos = widget.mapToGlobal(cursor_rect.bottomLeft())
    height = 6 * 34 + 14
    screen = widget.screen().availableGeometry() if hasattr(widget, "screen") else None
    if screen is not None and global_pos.y() + height > screen.bottom():
        global_pos.setY(widget.mapToGlobal(cursor_rect.topLeft()).y() - height)
    return QPoint(global_pos.x(), global_pos.y() + 4)


__all__ = [
    "COMMANDS",
    "PERMISSIONS",
    "CommandMenu",
    "CommandRow",
    "PermissionButton",
    "PermissionMenu",
    "SlashPopup",
    "command_by_name",
    "match_commands",
    "popup_position",
]

_ = QColor  # 保留导入，供后续自定义绘制使用
