"""路径提取：正则优先，LLM 辅助确认。

为什么不让 LLM 直接抽路径：
- 反斜杠在 JSON / 对话里容易被转义或吞掉
- 中文路径、带空格路径、相对路径，模型经常只抽出半截
- 模型还可能把 "两张" 之类的数字误当成路径片段

所以先用正则抓形状明确的部分，再用「文件是否存在」做硬校验，
最后才把不确定的候选交给 LLM 判断。
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

logger = logging.getLogger("findgo.path_extractor")


class PathExtractor:
    #: 目录存在性缓存，30 秒内不重复查磁盘
    _exists_cache: dict[str, tuple[float, bool]] = {}
    _cache_ttl: float = 30.0

    def __init__(self, sandbox=None):
        # 沙箱只做前置拦截：不在白名单里的路径直接丢弃，不送 LLM 判断
        self.sandbox = sandbox

    def _allowed(self, path: str) -> bool:
        """路径是否在沙箱允许范围内。没配沙箱时放行。"""
        if self.sandbox is None or not path:
            return True
        try:
            self.sandbox.check(path, "read")
            return True
        except Exception as e:
            logger.info("路径不在允许范围，已丢弃: %s（%s）", path, e)
            return False

    def _exists_dir(self, path: Path) -> bool:
        key = str(path).lower()
        now = time.time()
        hit = self._exists_cache.get(key)
        if hit is not None and now - hit[0] < self._cache_ttl:
            return hit[1]
        try:
            value = path.exists() and path.is_dir()
        except OSError:
            value = False
        self._exists_cache[key] = (now, value)
        return value

    # 盘符路径：C:\xxx、D:/xxx
    WINDOWS = re.compile(r"[A-Za-z]:[\\/][^\s\"'<>|?*，。；：、]+")
    # UNC 路径：\\server\share
    UNC = re.compile(r"\\\\[^\s\"'<>|?*，。；：、]+")
    # 相对路径：./xxx、../xxx
    RELATIVE = re.compile(r"(?<![\w.])\.{1,2}[\\/][^\s\"'<>|?*，。；：、]+")
    # 引号包裹：可能是带空格的路径
    QUOTED = re.compile(r"[\"']([^\"']{2,})[\"']")
    # 通用片段：只要同时含路径分隔符，就当作候选（后面用存在性校验兜底）
    SEGMENT = re.compile(r"[^\s\"'<>|?*，。；：、]+[/\\][^\s\"'<>|?*，。；：、]+")

    def candidates(self, text: str) -> list[str]:
        """按优先级抓候选路径：引号内 > 盘符 > UNC > 相对。"""
        found: list[str] = []

        for match in self.QUOTED.finditer(text or ""):
            value = match.group(1).strip()
            if self._looks_like_path(value):
                found.append(value)

        for pattern in (self.WINDOWS, self.UNC, self.RELATIVE, self.SEGMENT):
            for match in pattern.finditer(text or ""):
                value = match.group(0).rstrip(".,;:!?，。；：！？")
                if pattern is self.SEGMENT and not self._looks_like_relative(value):
                    continue
                found.append(value)

        # 去重并保持顺序，顺便去掉末尾标点
        seen: set[str] = set()
        out: list[str] = []
        for item in found:
            cleaned = item.rstrip(".,;:!?，。；：！？")
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(cleaned)
        return out

    @staticmethod
    def _looks_like_relative(value: str) -> bool:
        """过滤掉 1/2、http 链接这类伪路径。"""
        if len(value) < 3:
            return False
        if re.fullmatch(r"[\d./\\ -]+", value):
            return False
        if value.startswith(("http://", "https://", "www.")):
            return False
        return True

    @staticmethod
    def _looks_like_path(value: str) -> bool:
        return bool(
            re.match(r"^[A-Za-z]:[\\/]", value)
            or value.startswith("\\\\")
            or value.startswith(("./", "../", ".\\", "..\\"))
        )

    def resolve(self, candidates: list[str]) -> str:
        """硬校验：存在的目录优先。"""
        for item in candidates:
            try:
                path = Path(item).expanduser()
                if self._exists_dir(path):
                    return str(path.resolve())
            except OSError:
                continue
        return ""

    def resolve_with_llm(self, text: str, candidates: list[str], client) -> str:
        """正则 + 存在性校验都没定论时，让 LLM 在候选里挑一个。"""
        if not candidates:
            return ""

        # 0. 沙箱前置：不在白名单里的候选直接踢掉，不浪费一次 LLM 调用
        allowed = [c for c in candidates if self._allowed(c)]
        if not allowed:
            logger.info("所有路径候选都被沙箱拒绝: %s", candidates)
            return ""
        candidates = allowed

        # 1. 存在性校验（最可靠）
        resolved = self.resolve(candidates)
        if resolved:
            return resolved

        # 2. 只有一个候选且形如盘符路径，直接采用
        drive_like = [c for c in candidates if re.match(r"^[A-Za-z]:[\\/]", c)]
        if len(drive_like) == 1 and len(candidates) == 1:
            return drive_like[0]

        # 3. 交给 LLM 判断
        if client is None:
            return drive_like[0] if drive_like else ""

        prompt = (
            f"用户输入：{text}\n"
            f"正则提取的候选：{candidates}\n\n"
            "判断哪个候选是用户想检索的文件夹路径。"
            "如果都不是路径，path 填空字符串。"
            '只输出 JSON：{"path": "确认的路径或空字符串"}'
        )
        try:
            raw = client.chat(
                [
                    {"role": "system", "content": "你只输出 JSON，不做解释。"},
                    {"role": "user", "content": prompt},
                ],
                json_mode=True,
                max_tokens=128,
            )
            match = re.search(r"\{.*\}", raw or "", re.S)
            if match:
                data = json.loads(match.group(0))
                path = str(data.get("path") or "").strip()
                if path and path in candidates:
                    return path
        except Exception as e:
            logger.warning("LLM 确认路径失败: %s", e)

        return drive_like[0] if drive_like else ""

    def strip_paths(self, text: str, paths: list[str]) -> str:
        """把路径从原句里剔除，留下检索描述。"""
        cleaned = text or ""
        for item in paths:
            cleaned = cleaned.replace(item, " ")
        # 去掉常见的路径引导词残留
        cleaned = re.sub(r"(在|从|到|打开|扫描|遍历|查找|搜索|检索)\s*", " ", cleaned)
        cleaned = re.sub(r"(里|中|内|下|文件夹|目录)\s*", " ", cleaned)
        cleaned = re.sub(r"[，。；：、,.;:]+", " ", cleaned)
        cleaned = cleaned.replace('"', " ").replace("'", " ")
        cleaned = re.sub(r"^(的|在|从|把|将)\s*", "", cleaned.strip())
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned or (text or "")
