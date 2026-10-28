from pathlib import Path

# ── 1. AgentConfig 读暂存路径 ──
p = Path("src/imgsearch_agent/config.py")
text = p.read_text(encoding="utf-8")
old = '''        trash = raw.get("trash_staging", {}) or {}
        self.trash_max_days = int(trash.get("max_keep_days", 14))'''
new = '''        trash = raw.get("trash_staging", {}) or {}
        self.trash_path = str(trash.get("path") or "./data/trash_staging")
        self.trash_max_days = int(trash.get("max_keep_days", 14))'''
assert old in text
text = text.replace(old, new)
p.write_text(text, encoding="utf-8")
print("AgentConfig 已支持暂存路径")

# ── 2. 工具按配置解析暂存目录 ──
def patch_tool(path: Path, prop_body: str):
    text = path.read_text(encoding="utf-8")
    old = '''    @property
    def staging(self):
        if self._staging is None:
            from pathlib import Path

            from ..retriever.runtime import PROJECT_ROOT
            from ..trash import TrashStaging

            self._staging = TrashStaging(
                Path(PROJECT_ROOT) / "data" / "trash_staging"
            )
        return self._staging'''
    new = prop_body
    assert old in text, path
    path.write_text(text.replace(old, new, 1), encoding="utf-8")

resolver = '''    @property
    def staging(self):
        if self._staging is None:
            from pathlib import Path

            from ..config import AgentConfig
            from ..retriever.runtime import PROJECT_ROOT
            from ..trash import TrashStaging

            raw = AgentConfig().trash_path
            root = Path(raw).expanduser()
            if not root.is_absolute():
                root = Path(PROJECT_ROOT) / root
            self._staging = TrashStaging(root)
        return self._staging'''

patch_tool(Path("src/imgsearch_agent/tools/preview_staged.py"), resolver)

# trash_manage 的 property 结构相同
patch_tool(Path("src/imgsearch_agent/tools/trash_manage.py"), resolver)

print("两个暂存工具已按配置解析路径")

# ── 3. agent.yaml 加 path ──
for name in ("configs/agent.yaml", "configs/agent.example.yaml"):
    p = Path(name)
    text = p.read_text(encoding="utf-8")
    if "trash_staging:" in text and "path:" not in text.split("trash_staging:")[1][:200]:
        text = text.replace(
            "trash_staging:\n",
            "trash_staging:\n  path: ./data/trash_staging   # 暂存区位置，可改成其他磁盘\n",
            1,
        )
        p.write_text(text, encoding="utf-8")
        print("已更新", name)
