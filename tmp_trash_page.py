from pathlib import Path

# ── agent.yaml 补 path ──
for name in ("configs/agent.yaml", "configs/agent.example.yaml"):
    p = Path(name)
    text = p.read_text(encoding="utf-8")
    if "trash_staging:" not in text:
        continue
    block_start = text.find("trash_staging:")
    block = text[block_start:block_start + 260]
    if "path:" in block:
        continue
    text = text.replace(
        "trash_staging:\n",
        "trash_staging:\n  path: ./data/trash_staging   # 暂存区位置，可改到其他磁盘\n",
        1,
    )
    p.write_text(text, encoding="utf-8")
    print("已补 path:", name)

# ── 设置页加位置选择 ──
p = Path("src/imgsearch_agent/settings_dialog.py")
text = p.read_text(encoding="utf-8")

old = '''        self.trash_days = QSpinBox()'''
new = '''        self.trash_path_edit = QLineEdit()
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

        self.trash_days = QSpinBox()'''
assert old in text
text = text.replace(old, new)

old = '''    def _load_trash(self) -> None:
        try:
            from .config import AgentConfig

            cfg = AgentConfig()
            self.trash_days.setValue(int(cfg.trash_max_days))'''
new = '''    def on_pick_trash_path(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择暂存区目录")
        if folder:
            self.trash_path_edit.setText(folder)

    def _load_trash(self) -> None:
        try:
            from .config import AgentConfig

            cfg = AgentConfig()
            self.trash_path_edit.setText(str(cfg.trash_path))
            self.trash_days.setValue(int(cfg.trash_max_days))'''
assert old in text
text = text.replace(old, new)

old = '''            data.setdefault("trash_staging", {})
            data["trash_staging"]["max_keep_days"] = self.trash_days.value()'''
new = '''            data.setdefault("trash_staging", {})
            data["trash_staging"]["path"] = self.trash_path_edit.text().strip()
            data["trash_staging"]["max_keep_days"] = self.trash_days.value()'''
assert old in text
text = text.replace(old, new)

p.write_text(text, encoding="utf-8")
print("设置页已支持选择暂存位置")
