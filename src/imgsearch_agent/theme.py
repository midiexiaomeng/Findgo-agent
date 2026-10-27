"""统一主题：米白底、低饱和绿、圆角控件。

放在独立模块，桌面主窗口、设置对话框、图库窗口共用同一份。
"""

from __future__ import annotations

GLOBAL_QSS = """
QDialog { background: #FBFAF7; }
QWidget { color: #2E2E2E; }

QPushButton {
    background: #FFFFFF;
    border: 1px solid #EDEBE6;
    border-radius: 8px;
    padding: 6px 14px;
    color: #2E2E2E;
}
QPushButton:hover { background: #F2F0EA; }
QPushButton:pressed { background: #E9E6DF; }
QPushButton:disabled { color: #B5B2AA; }

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {
    background: #FFFFFF;
    border: 1px solid #EDEBE6;
    border-radius: 8px;
    padding: 6px 10px;
    color: #2E2E2E;
}
QComboBox::drop-down { border: none; width: 20px; }

QListWidget, QTreeWidget {
    background: #FFFFFF;
    border: 1px solid #EDEBE6;
    border-radius: 10px;
}
QListWidget::item { padding: 6px 8px; border-radius: 6px; }
QListWidget::item:selected { background: #E3EFE7; color: #2E2E2E; }

QCheckBox { color: #2E2E2E; spacing: 8px; }

QMenu {
    background: #FFFFFF;
    border: 1px solid #EDEBE6;
    border-radius: 10px;
    padding: 4px;
}
QMenu::item { padding: 6px 18px; border-radius: 6px; }
QMenu::item:selected { background: #E3EFE7; }

QScrollBar:vertical { background: transparent; width: 8px; }
QScrollBar::handle:vertical {
    background: #DEDBD3; border-radius: 4px; min-height: 24px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QToolTip {
    background: #FFFFFF;
    color: #2E2E2E;
    border: 1px solid #EDEBE6;
    border-radius: 6px;
    padding: 4px 8px;
}
"""

#: 对话框与主窗口目前共用一套，保留名字便于以后单独微调
DIALOG_QSS = GLOBAL_QSS
