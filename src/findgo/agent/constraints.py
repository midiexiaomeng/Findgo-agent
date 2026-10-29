"""结构化约束：把用户的话拆成 can/cannot/should 三类。

为什么需要这层：
用户的「白色猫，不要黑的，最好躺着」在向量检索里只能变成一个字符串，
否定词、偏好词全部丢失。约束必须独立建模，才能在召回路过滤、
在重排路打分。

中文约束要落到 BLIP 的英文描述上，所以颜色、动作、场景这些高频维度
维护了一张对照表；表里没有的交给 LLM 判断。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 中文约束词 → 英文描述里可能出现的写法
# 只覆盖高频维度，不求全；未命中的词交给 LLM 在重排时判断
TERM_MAP: dict[str, tuple[str, ...]] = {
    # 颜色
    "白色": ("white", "snow"),
    "白": ("white", "snow"),
    "黑色": ("black", "dark"),
    "黑": ("black", "dark"),
    "灰色": ("gray", "grey"),
    "灰": ("gray", "grey"),
    "红色": ("red", "crimson"),
    "蓝色": ("blue", "navy"),
    "绿色": ("green",),
    "黄色": ("yellow", "gold", "golden"),
    "橙色": ("orange",),
    "紫色": ("purple", "violet"),
    "粉色": ("pink", "rose"),
    "棕色": ("brown", "tan"),
    "褐色": ("brown", "tan"),
    # 状态 / 动作
    "躺着": ("lying", "lay", "resting", "lying down"),
    "坐着": ("sitting", "sit", "seated"),
    "站着": ("standing", "stand"),
    "跑": ("running", "run"),
    "睡觉": ("sleeping", "sleep", "asleep"),
    "吃": ("eating", "eat"),
    "看镜头": ("looking at the camera", "facing the camera"),
    "特写": ("close-up", "close up", "closeup"),
    # 题材
    "猫": ("cat", "kitten", "kitty"),
    "狗": ("dog", "puppy"),
    "人": ("person", "people", "man", "woman", "human"),
    "风景": ("landscape", "scenery"),
    "食物": ("food", "meal", "dish"),
    "建筑": ("building", "architecture"),
    "车": ("car", "vehicle", "automobile"),
    "花": ("flower", "blossom"),
    "海": ("sea", "ocean", "beach"),
    "山": ("mountain", "hill"),
    "天空": ("sky",),
    "室内": ("indoor", "indoors", "room", "inside"),
    "室外": ("outdoor", "outdoors", "outside"),
    # 画质 / 风格
    "卡通": ("cartoon", "anime", "illustration", "drawing"),
    "插画": ("illustration", "drawing", "artwork"),
    "照片": ("photo", "photograph"),
    "黑白": ("black and white", "monochrome"),
    "模糊": ("blurry", "blurred"),
    "夜景": ("night", "nighttime"),
}

# 否定词：出现在约束词前面时，把它翻到 must_not
NEGATIVE_PREFIX = ("不要", "不想要", "排除", "非", "别", "去掉", "没有", "不是")

# 偏好词：出现在约束词前面时，归到 should
PREFERENCE_PREFIX = ("最好", "尽量", "优先", "偏向", "希望", "喜欢", "要是能")


@dataclass
class Constraints:
    """一条检索任务的结构化约束。"""

    core_query: str = ""
    must: list[str] = field(default_factory=list)
    must_not: list[str] = field(default_factory=list)
    should: list[str] = field(default_factory=list)
    diversity: bool = True
    #: 用户没明确数量时为 0，由上层用默认值
    retrieve_count: int = 0

    def is_empty(self) -> bool:
        return not (self.must or self.must_not or self.should)

    def to_dict(self) -> dict:
        return {
            "core_query": self.core_query,
            "must": list(self.must),
            "must_not": list(self.must_not),
            "should": list(self.should),
            "diversity": self.diversity,
            "retrieve_count": self.retrieve_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Constraints":
        if not isinstance(data, dict):
            return cls()

        def _clean_list(value) -> list[str]:
            if not isinstance(value, list):
                return []
            return [str(v).strip() for v in value if str(v).strip()]

        try:
            count = int(data.get("retrieve_count", 0) or 0)
        except (TypeError, ValueError):
            count = 0

        return cls(
            core_query=str(data.get("core_query", "") or "").strip(),
            must=_clean_list(data.get("must")),
            must_not=_clean_list(data.get("must_not")),
            should=_clean_list(data.get("should")),
            diversity=bool(data.get("diversity", True)),
            retrieve_count=max(0, count),
        )


# ── 英文匹配 ──────────────────────────────────────────────────────────────


def english_terms(term: str) -> tuple[str, ...]:
    """把中文约束翻成英文关键词；表里没有就原样返回，交给字符串匹配。"""
    key = term.strip()
    if not key:
        return ()

    mapped = TERM_MAP.get(key)
    if mapped:
        return mapped

    # 「白色猫」这类复合词，拆成能认出来的部分
    parts: list[str] = []
    for zh, en in TERM_MAP.items():
        if zh in key:
            parts.extend(en)
    return tuple(parts) if parts else (key.lower(),)


def matches_term(caption: str, term: str) -> bool:
    """描述里是否出现这个约束词（中英都试）。"""
    text = (caption or "").lower()
    if not text:
        return False

    for en in english_terms(term):
        if en and en in text:
            return True

    # 描述本身是中文时直接比
    return term.lower() in text


def filter_by_constraints(
    items: list[dict],
    must_not: list[str],
    must: list[str],
) -> tuple[list[dict], list[dict]]:
    """按硬约束过滤，返回 (保留, 被剔除)。

    只有 must_not 是真正的硬淘汰。must 不做淘汰，原因：
    BLIP 的描述经常只写「a cat on a chair」，并不提颜色、动作；
    拿中文约束去匹配不完整的描述，会把大量合格的图误杀。
    must 命中作为加分标记，未命中交给后面的语义重排判断。

    没有描述的图片不参与硬过滤：无法判断就不淘汰。
    """
    kept: list[dict] = []
    dropped: list[dict] = []

    for item in items:
        caption = str(item.get("caption", "") or "")
        if not caption.strip():
            kept.append(item)
            continue

        blocked = next((t for t in must_not if matches_term(caption, t)), None)
        if blocked:
            dropped.append({**item, "_drop_reason": f"命中排除条件：{blocked}"})
            continue

        # must 只做标记，不淘汰
        hit_must = [t for t in must if matches_term(caption, t)]
        if hit_must:
            kept.append({**item, "_must_hit": hit_must})
        else:
            kept.append(item)

    return kept, dropped


# ── 从自然语言里兜底抽取 ──────────────────────────────────────────────────


def parse_from_text(text: str) -> Constraints:
    """LLM 没给出约束时的兜底：用否定词与偏好词切分。

    只在 LLM 输出缺字段时使用，覆盖最常见的「不要 X」「最好 X」句式。
    """
    raw = (text or "").strip()
    if not raw:
        return Constraints()

    must_not: list[str] = []
    should: list[str] = []
    consumed: set[int] = set()

    for prefix in NEGATIVE_PREFIX:
        for m in re.finditer(re.escape(prefix) + r"\s*([^\s，。；、,.;]{1,8})", raw):
            term = m.group(1).strip()
            if term and term not in must_not:
                must_not.append(term)
            consumed.update(range(m.start(), m.end()))

    for prefix in PREFERENCE_PREFIX:
        for m in re.finditer(re.escape(prefix) + r"\s*([^\s，。；、,.;]{1,8})", raw):
            term = m.group(1).strip()
            if term and term not in should:
                should.append(term)
            consumed.update(range(m.start(), m.end()))

    core = "".join(ch for i, ch in enumerate(raw) if i not in consumed).strip(" ，。；、,.;")
    return Constraints(core_query=core, must_not=must_not, should=should)


# ── core_query 清洗 ─────────────────────────────────────────────────────

#: 数量表述：两张、三张、十个、5 张……
COUNT_PATTERN = re.compile(
    r"[\d一二两三四五六七八九十百千]+\s*(?:张|幅|个|条|只|种|张图|张照片|张图片)"
)

#: 否定前缀，命中后把它后面的词提到 must_not
NEG_PATTERN = re.compile(
    r"(?:不要|不想要|排除|别要|别|去掉|不是|非)\s*([^\s，。；、,.;]{1,8}?)(?=[，。；、,.;\s]|$)"
)

#: 偏好前缀，命中后把它后面的词提到 should
PREF_PATTERN = re.compile(
    r"(?:最好|尽量|优先|偏向|希望|喜欢|要是能)\s*([^\s，。；、,.;]{1,8}?)(?=[，。；、,.;\s]|$)"
)

#: 检索动词与尾缀，不构成画面内容
VERB_PATTERN = re.compile(
    r"^(?:帮我|给我|麻烦)?\s*(?:找|搜|查找|搜索|看看|来|要|列|弄)(?:一下|几张|一些)?\s*"
)
TAIL_PATTERN = re.compile(r"\s*(?:的)?(?:图片|照片|图|图像)\s*$")


def clean_core_query(text: str) -> tuple[str, list[str], list[str]]:
    """把 core_query 里的数量词、否定词、偏好词剥出去。

    即使 LLM 把它们写进 core_query，这一步也会拉到正确字段。
    返回 (干净的主体词, 剥离出的否定词, 剥离出的偏好词)。
    """
    raw = (text or "").strip()
    if not raw:
        return "", [], []

    must_not: list[str] = []
    should: list[str] = []

    for m in NEG_PATTERN.finditer(raw):
        term = m.group(1).strip(" 的")
        if term and term not in must_not:
            must_not.append(term)

    for m in PREF_PATTERN.finditer(raw):
        term = m.group(1).strip(" 的")
        if term and term not in should:
            should.append(term)

    cleaned = NEG_PATTERN.sub(" ", raw)
    cleaned = PREF_PATTERN.sub(" ", cleaned)
    cleaned = COUNT_PATTERN.sub(" ", cleaned)

    # 去掉残留的连接词与标点
    cleaned = re.sub(r"[，。；、,.;]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" 的")

    # 检索动词与「的图片」这类尾缀不属于画面内容
    cleaned = VERB_PATTERN.sub("", cleaned)
    cleaned = TAIL_PATTERN.sub("", cleaned)
    cleaned = cleaned.strip(" 的")

    # 只剩虚词时当作没洗出东西，交给上层兜底
    if not cleaned or cleaned in {"图片", "照片", "图", "找到", "找"}:
        cleaned = ""

    return cleaned, must_not, should


__all__ = [
    "Constraints",
    "TERM_MAP",
    "clean_core_query",
    "english_terms",
    "filter_by_constraints",
    "matches_term",
    "parse_from_text",
]
