from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping, Tuple
from urllib.parse import urlparse

PAPER_TOPIC_KEYWORDS = {
    "Physical AI": [
        "physical ai",
        "embodied ai",
        "embodied agent",
        "sim2real",
        "real-robot",
        "vision-language-action",
        "vla",
        "robot foundation model",
        "policy learning",
    ],
    "Robotics": [
        "robot",
        "robotics",
        "manipulation",
        "grasp",
        "locomotion",
        "navigation",
        "bimanual",
        "humanoid",
        "imitation learning",
    ],
    "World Model": [
        "world model",
        "world models",
        "predictive model",
        "predictive control",
        "video prediction",
        "latent dynamics",
        "dynamics model",
        "planning model",
        "dreamer",
    ],
}

TOPIC_CN_MAP = {
    "Physical AI": "具身智能",
    "World Model": "世界模型",
    "Robotics": "机器人",
    "Other": "其他",
}

CATEGORY_CN_MAP = {
    "Physical AI": "具身智能",
    "World Model": "世界模型",
    "Robotics": "机器人",
    "模型/研究": "模型/研究",
    "产品发布": "产品发布",
    "基础设施": "基础设施",
    "开源生态": "开源生态",
    "行业动态": "行业动态",
    "企业合作": "企业合作",
    "社交讨论": "社交讨论",
    "视频观点": "视频解读",
    "视频解读": "视频解读",
    "应用落地": "应用落地",
    "Other": "其他",
    "其他": "其他",
}

ADVANCED_PAPER_HINTS = [
    "foundation model",
    "generalist",
    "benchmark",
    "scaling",
    "sota",
    "state of the art",
    "large-scale",
    "end-to-end",
    "diffusion policy",
    "transformer",
    "reasoning",
    "multimodal",
]

WEB_AI_KEYWORDS = [
    "ai",
    "artificial intelligence",
    "machine learning",
    "llm",
    "agent",
    "robot",
    "robotics",
    "world model",
    "foundation model",
    "multimodal",
    "diffusion",
    "openai",
    "anthropic",
    "deepmind",
    "gemini",
    "gpt",
    "claude",
]

LOW_SIGNAL_TITLE_PATTERNS = [
    r"\bbest\b",
    r"\btop\s*\d+\b",
    r"\bvs\b",
    r"\breview\b",
    r"\bguide\b",
    r"\bhow to\b",
    r"\bactually works\b",
    r"\badventures?\b",
    r"\bupgrade\b",
    r"\bnews roundup\b",
    r"\bstock picks\b",
]

LOW_SIGNAL_HOST_HINTS = (
    "youtube.com",
    "youtu.be",
    "msn.com",
    "blockchain-council.org",
)

HIGH_SIGNAL_HOST_HINTS = (
    "openai.com",
    "anthropic.com",
    "deepmind.google",
    "google.com",
    "huggingface.co",
    "nvidia.com",
    "aws.amazon.com",
    "microsoft.com",
    "techcrunch.com",
    "venturebeat.com",
    "marktechpost.com",
    "unite.ai",
    "artificialintelligence-news.com",
)

HIGH_SIGNAL_WEB_KEYWORDS = [
    "release",
    "released",
    "launch",
    "launched",
    "raises",
    "raised",
    "funding",
    "partnership",
    "deploy",
    "deployment",
    "benchmark",
    "paper",
    "world model",
    "robotics",
    "humanoid",
    "inference",
    "datacenter",
    "chip",
    "open source",
]

HIGH_SIGNAL_ACTION_PATTERNS = [
    "launch",
    "launched",
    "release",
    "released",
    "raise",
    "raised",
    "funding",
    "partnership",
    "deploy",
    "deployment",
    "open source",
    "benchmark",
    "paper",
    "acquire",
    "acquisition",
    "invest",
    "investment",
    "roadmap",
]

LOW_INFORMATION_PATTERNS = [
    r"\bopinion\b",
    r"\bwatch\b",
    r"\brecap\b",
    r"\bpodcast\b",
    r"\broundup\b",
    r"\breaction\b",
    r"\bexplained\b",
    r"\beverything you need to know\b",
    r"\bmy thoughts\b",
]

HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_text(*parts: str) -> str:
    combined = " ".join(part or "" for part in parts)
    combined = HTML_TAG_RE.sub(" ", combined)
    return MULTI_SPACE_RE.sub(" ", combined).strip().lower()


def count_keyword_hits(text: str, keywords: Iterable[str]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def classify_paper_topic(title: str, content: str) -> Tuple[str, int]:
    text = normalize_text(title, content)
    best_topic = "Other"
    best_hits = 0
    for topic, keywords in PAPER_TOPIC_KEYWORDS.items():
        hits = count_keyword_hits(text, keywords)
        if hits > best_hits:
            best_topic = topic
            best_hits = hits
    return best_topic, best_hits


def score_paper_relevance(title: str, content: str) -> Tuple[str, int]:
    text = normalize_text(title, content)
    topic, topic_hits = classify_paper_topic(title, content)
    advanced_hits = count_keyword_hits(text, ADVANCED_PAPER_HINTS)
    relevance = topic_hits * 3 + advanced_hits
    return topic, relevance


def is_relevant_paper(title: str, content: str) -> bool:
    _, relevance = score_paper_relevance(title, content)
    return relevance >= 3


def is_ai_web_content(title: str, content: str) -> bool:
    text = normalize_text(title, content)
    return count_keyword_hits(text, WEB_AI_KEYWORDS) > 0


def infer_platform(url: str, source_detail: str = "") -> str:
    host = urlparse(url).netloc.lower()
    detail = (source_detail or "").lower()
    if "x.com" in host or "twitter.com" in host or detail == "x":
        return "X"
    if "youtube.com" in host or "youtu.be" in host:
        return "YouTube"
    if "arxiv.org" in host or detail == "arxiv":
        return "ArXiv"
    if "huggingface.co" in host:
        return "Hugging Face"
    if any(token in host for token in ["substack.com", "medium.com", "blog", "openai.com", "anthropic.com", "deepmind.google"]):
        return "Blog"
    return "Web"


def clean_snippet(text: str, limit: int = 500) -> str:
    text = HTML_TAG_RE.sub(" ", text or "")
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def topic_to_cn(topic: str) -> str:
    return TOPIC_CN_MAP.get((topic or "").strip(), "其他")


def category_to_cn(category: str) -> str:
    return CATEGORY_CN_MAP.get((category or "").strip(), category or "其他")


def content_kind_to_cn(content_type: str) -> str:
    return "论文" if (content_type or "").strip() == "paper" else "全网动态"


def _matching_weight(mapping: Mapping[str, Any] | None, key: str) -> float:
    if not mapping:
        return 0.0
    lowered_key = (key or "").lower()
    best = 0.0
    for raw_name, raw_value in mapping.items():
        name = str(raw_name).lower()
        if name and name in lowered_key:
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if abs(value) > abs(best):
                best = value
    return best


def score_preference_boost(
    item: Dict[str, Any],
    preference_config: Mapping[str, Any] | None = None,
) -> float:
    if not preference_config:
        return 0.0

    score = 0.0
    content_type = str(item.get("content_type", "")).strip()
    category = str(item.get("category") or item.get("topic") or "").strip()
    text = normalize_text(
        item.get("title", ""),
        item.get("title_cn", ""),
        item.get("summary", ""),
        item.get("content", ""),
        category,
    )

    content_type_weights = preference_config.get("content_type_weights") or {}
    if content_type in content_type_weights:
        try:
            score += float(content_type_weights[content_type])
        except (TypeError, ValueError):
            pass

    category_weights = preference_config.get("category_weights") or {}
    if category in category_weights:
        try:
            score += float(category_weights[category])
        except (TypeError, ValueError):
            pass

    keyword_weights = preference_config.get("keyword_weights") or {}
    for keyword, raw_weight in keyword_weights.items():
        if str(keyword).lower() in text:
            try:
                score += float(raw_weight)
            except (TypeError, ValueError):
                continue

    return round(score, 2)


def score_update_quality(
    title: str,
    content: str,
    url: str = "",
    platform: str = "",
    source_detail: str = "",
    category: str = "",
    source_preferences: Mapping[str, Any] | None = None,
) -> float:
    text = normalize_text(title, content, category)
    host = urlparse(url or "").netloc.lower()
    platform = (platform or "").lower()
    source_detail = (source_detail or "").lower()
    score = 0.0

    if any(hint in host for hint in HIGH_SIGNAL_HOST_HINTS):
        score += 2.0
    if platform in {"blog", "news", "website"}:
        score += 1.0
    if source_detail in {"openai blog", "google research", "deepmind blog", "techcrunch ai", "marktechpost"}:
        score += 1.0

    score += min(3, count_keyword_hits(text, HIGH_SIGNAL_WEB_KEYWORDS)) * 0.7

    if any(hint in host for hint in LOW_SIGNAL_HOST_HINTS):
        score -= 1.4
    if platform == "youtube":
        score -= 0.8
    if len(re.findall(r"[A-Z]{4,}", title or "")) >= 2:
        score -= 0.5
    if len((title or "").strip()) < 28:
        score -= 0.6

    lowered_title = (title or "").lower()
    for pattern in LOW_SIGNAL_TITLE_PATTERNS:
        if re.search(pattern, lowered_title):
            score -= 1.6

    if any(token in lowered_title for token in ["sponsored", "coupon", "deals"]):
        score -= 2.0

    summary_length = len(clean_snippet(content, 500))
    if summary_length >= 220:
        score += 0.8
    elif summary_length <= 90:
        score -= 0.8

    action_hits = count_keyword_hits(text, HIGH_SIGNAL_ACTION_PATTERNS)
    score += min(3, action_hits) * 0.45

    unique_tokens = {token for token in re.split(r"[^a-z0-9\u4e00-\u9fff]+", text) if len(token) >= 3}
    if len(unique_tokens) >= 24:
        score += 0.6
    elif len(unique_tokens) <= 9:
        score -= 0.7

    if re.search(r"\$\d", title or "") or re.search(r"\b\d+(\.\d+)?\s*(million|billion)\b", lowered_title):
        score += 0.5
    if re.search(r"\b\d+%\b", lowered_title) or re.search(r"\b\d+x\b", lowered_title):
        score += 0.4

    for pattern in LOW_INFORMATION_PATTERNS:
        if re.search(pattern, lowered_title):
            score -= 1.2

    if platform == "youtube":
        if action_hits == 0 or summary_length < 150:
            score -= 1.2
        if any(token in lowered_title for token in ["review", "roundup", "podcast", "reaction", "watch"]):
            score -= 1.0

    if source_preferences:
        whitelist_hosts = [str(item).lower() for item in (source_preferences.get("whitelist_hosts") or [])]
        blacklist_hosts = [str(item).lower() for item in (source_preferences.get("blacklist_hosts") or [])]
        if any(token in host for token in whitelist_hosts):
            score += 0.8
        if any(token in host for token in blacklist_hosts):
            score -= 1.2

        score += _matching_weight(source_preferences.get("source_weights") or {}, host)
        score += _matching_weight(source_preferences.get("platform_weights") or {}, platform)
        score += _matching_weight(source_preferences.get("platform_weights") or {}, source_detail)
        score += _matching_weight(source_preferences.get("category_weights") or {}, category)

    return round(score, 2)


def is_low_signal_update(
    title: str,
    content: str,
    url: str = "",
    platform: str = "",
    source_detail: str = "",
    category: str = "",
    source_preferences: Mapping[str, Any] | None = None,
) -> bool:
    quality_score = score_update_quality(
        title,
        content,
        url,
        platform,
        source_detail,
        category,
        source_preferences=source_preferences,
    )
    host = urlparse(url or "").netloc.lower()
    lowered_title = (title or "").lower()

    if quality_score <= -1.0:
        return True
    if len(clean_snippet(content, 300)) < 70 and quality_score < 1.4:
        return True
    if "youtube.com" in host or "youtu.be" in host:
        return quality_score < 1.0 and any(
            token in lowered_title
            for token in ["best", "top", "vs", "roundup", "review", "guide", "actually works", "adventures"]
        )
    return False


def infer_impact_tag(
    title: str,
    summary: str = "",
    why_now: str = "",
    expected_effect: str = "",
    future_impact: str = "",
    category: str = "",
) -> str:
    text = normalize_text(title, summary, why_now, expected_effect, future_impact, category)
    mapping = [
        ("降成本", ["cost", "成本", "memory wall", "显存", "waterless cooling", "效率"]),
        ("提速度", ["real-time", "latency", "加速", "推理", "实时", "吞吐"]),
        ("扩生态", ["open source", "开源", "ecosystem", "生态", "developer"]),
        ("抢算力", ["chip", "gpu", "datacenter", "算力", "基础设施", "infra"]),
        ("强落地", ["deploy", "adoption", "enterprise", "客户", "落地", "应用"]),
        ("提能力", ["benchmark", "world model", "robot", "humanoid", "capability", "safety"]),
    ]
    for tag, keywords in mapping:
        if any(keyword.lower() in text for keyword in keywords):
            return tag
    return "值得看"
