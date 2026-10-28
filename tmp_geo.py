from pathlib import Path

p = Path("desktop_app.py")
text = p.read_text(encoding="utf-8")

# ── 1. 右栏默认宽度 300 -> 225 ──
text = text.replace("main.setSizes([240, 900, 300])", "main.setSizes([240, 900, 225])")
text = text.replace("self._right_width = 300", "self._right_width = 225")
text = text.replace("sizes[2] = 300", "sizes[2] = 225")
text = text.replace('self._right_width or 300', 'self._right_width or 225')
text = text.replace("right.setSizes([400, 380])", "right.setSizes([400, 330])")

# ── 2. QSettings 导入 ──
if "QSettings" not in text.split("class ")[0]:
    text = text.replace(
        "from PyQt5.QtCore import QEvent, QObject, QPoint, QRect, Qt, QThread, pyqtSignal",
        "from PyQt5.QtCore import (\n    QEvent,\n    QObject,\n    QPoint,\n    QRect,\n    QSettings,\n    Qt,\n    QThread,\n    pyqtSignal,\n)",
    )

# ── 3. 启动恢复窗口尺寸 ──
old = '''        screen = QApplication.primaryScreen()'''
new = '''        self._restore_geometry()

        screen = QApplication.primaryScreen()'''
assert old in text, "__init__ 未匹配"
text = text.replace(old, new)

# ── 4. 保存 / 恢复方法 ──
old = '''    def toggle_maximize(self) -> None:'''
new = '''    @staticmethod
    def _settings() -> QSettings:
        return QSettings("imgsearch", "agent")

    def _restore_geometry(self) -> None:
        try:
            saved = self._settings().value("window/geometry")
            if saved:
                self.restoreGeometry(saved)
                return
        except Exception:
            pass

    def _save_geometry(self) -> None:
        try:
            self._settings().setValue("window/geometry", self.saveGeometry())
        except Exception:
            pass

    def toggle_maximize(self) -> None:'''
assert old in text
text = text.replace(old, new, 1)

# ── 5. closeEvent 保存 ──
old = '''    def closeEvent(self, event) -> None:
        """关闭窗口时释放向量库文件锁，便于下次启动。"""
        try:'''
new = '''    def closeEvent(self, event) -> None:
        """关闭窗口时保存尺寸并释放向量库文件锁。"""
        self._save_geometry()
        try:'''
assert old in text
text = text.replace(old, new)

p.write_text(text, encoding="utf-8")
print("右栏宽度与窗口尺寸记忆已修改")
