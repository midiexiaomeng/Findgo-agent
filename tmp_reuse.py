from pathlib import Path

# ── 1. TaskSpec 加 reuse_last ──
p = Path("src/imgsearch_agent/utils/schemas.py")
text = p.read_text(encoding="utf-8")
old = '''    is_dynamic: bool = Field(False, description="是否由动态工具执行")'''
new = '''    reuse_last: bool = Field(False, description="直接复用上一轮的图片结果")
    is_dynamic: bool = Field(False, description="是否由动态工具执行")'''
assert old in text
text = text.replace(old, new)
p.write_text(text, encoding="utf-8")

# ── 2. Planner 识别重复发送意图 ──
p = Path("src/imgsearch_agent/agent/planner.py")
text = p.read_text(encoding="utf-8")

old = '''RISK_KEYWORDS = ("删除", "删掉", "清理", "移除", "清空", "purge", "trash")'''
new = '''RISK_KEYWORDS = ("删除", "删掉", "清理", "移除", "清空", "purge", "trash")

REUSE_KEYWORDS = (
    "刚才",
    "上次",
    "上一次",
    "再发",
    "重新发",
    "重复",
    "还是那",
    "同样那",
    "刚找的",
)


def detect_reuse(query: str) -> bool:
    """用户是否在要求复用上一轮的图片，而不是重新检索。"""
    text = (query or "").lower()
    return any(word in text for word in REUSE_KEYWORDS)'''
assert old in text
text = text.replace(old, new)

old = '''        risk = assess_risk(query)
        for task in result:
            task.risk = "high" if task.is_dynamic else risk'''
new = '''        risk = assess_risk(query)
        reuse = detect_reuse(query)
        for task in result:
            task.risk = "high" if task.is_dynamic else risk
            task.reuse_last = reuse'''
assert old in text
text = text.replace(old, new)
p.write_text(text, encoding="utf-8")
print("Planner 已识别复用意图")

# ── 3. Orchestrator 缓存上一轮图片并复用 ──
p = Path("src/imgsearch_agent/agent/orchestrator.py")
text = p.read_text(encoding="utf-8")

old = '''        # 上一轮返回过的图片，下一轮默认排除，避免重复推荐
        self._last_paths: set[str] = set()'''
new = '''        # 上一轮返回过的图片，下一轮默认排除，避免重复推荐
        self._last_paths: set[str] = set()
        self._last_images: list[dict] = []'''
assert old in text
text = text.replace(old, new)

old = '''        results: list[AgentResult] = []
        previous_paths = set(self._last_paths)'''
new = '''        results: list[AgentResult] = []
        previous_paths = set(self._last_paths)
        previous_images = list(self._last_images)'''
assert old in text
text = text.replace(old, new)

old = '''            if task.is_dynamic:'''
new = '''            if task.reuse_last:
                if previous_images:
                    results.append(
                        AgentResult(
                            reply=f"这是上一次找的 {len(previous_images)} 张。",
                            results=list(previous_images),
                            tool="image_retrieval",
                            confidence=1.0,
                            trace_id=uuid.uuid4().hex[:8],
                        )
                    )
                    logger.info("复用上一轮结果 %d 张", len(previous_images))
                    continue
                # 没有可复用的结果，退回正常检索
                logger.info("用户要求重发但上一轮没有结果，改为重新检索")

            if task.is_dynamic:'''
assert old in text, "run_tasks 分支未匹配"
text = text.replace(old, new)

old = '''        self._last_paths = this_round_paths
        return results'''
new = '''        self._last_paths = this_round_paths
        if this_round_paths:
            self._last_images = [
                item
                for result in results
                for item in result.results
                if isinstance(item, dict)
            ]
        return results'''
assert old in text, "_last_paths 未匹配"
text = text.replace(old, new)

p.write_text(text, encoding="utf-8")
print("orchestrator 已支持复用上一轮图片")
