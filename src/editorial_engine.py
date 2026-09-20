from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


DOMAIN_ORDER = (
    "world_model",
    "physical_ai",
    "agent_models",
    "infra_open_source",
    "products_business",
)

DOMAIN_LABELS = {
    "world_model": "World Model",
    "physical_ai": "Physical AI / Robotics",
    "agent_models": "Agent / Models",
    "infra_open_source": "Infra / Open Source",
    "products_business": "Products / Business",
}

BAD_PUBLIC_PHRASES = (
    "学习重点是",
    "技术上，它主要围绕",
    "需要回看原文确认",
    "当前摘要还缺少",
    "是否真正",
    "值得持续关注",
    "未来可能带来影响",
    "AI领域新进展",
    "Product Release",
    "Industry Update",
    "据行业来源称",
    "从标题和摘要看",
    "当前材料不足以展开",
    "材料给出",
    "要解决的是",
    "核心动作是",
    "行业资源配置",
    "开始出现新变化",
    "正在重排资源",
)

FIELD_LABEL_LEAKS = (
    "Product Release",
    "Industry Update",
    "AI领域新进展",
    "相关进展",
    "学习重点是",
    "技术方向：",
)

ATTRIBUTION_OPENERS = (
    "团队表示",
    "公司称",
    "发布方披露",
    "官方公告显示",
    "团队介绍",
    "官方称",
)

MOJIBAKE_MARKERS = (
    "�",
    "Ã",
    "Â",
    "鍏",
    "鐨",
    "璁",
    "鎶",
    "绋",
    "涓",
    "浠",
    "杩",
    "瀛",
)


def attribution_opener_pattern(value: Any) -> str:
    opening = re.split(r"[。！？!?]", str(value or ""), maxsplit=1)[0].strip()
    match = re.match(
        rf"^[^，,。！？!?]{{0,48}}({'|'.join(map(re.escape, ATTRIBUTION_OPENERS))})[，,]",
        opening,
    )
    return match.group(1) if match else ""

METHOD_PATTERN = re.compile(
    r"方法|机制|框架|训练|学习|预测|规划|控制|判断|决策|生成|对齐|检索|微调|蒸馏|评估|分析|证明|构造|比较|指出|研究|采用|使用|利用|结合|设计|引入|构建|通过|编码|分配|约束|拆分|交给|benchmark|dataset|baseline|model",
    re.IGNORECASE,
)
RESULT_PATTERN = re.compile(
    r"实验|指标|结果|报告|成功率|准确率|提升|降低|超过|优于|对比|对照|参照|基准|数据集|验证|消融|迁移|观察|差异|结论|测得|达到|保持|benchmark|result|outperform|success|%",
    re.IGNORECASE,
)
PAPER_OUTCOME_PATTERN = re.compile(
    r"成功率|准确率|召回率|精度|延迟|吞吐|显存|成本|百分点|厘米|毫秒|R²|F1|BLEU|mAP|AUC|FPS|TOPS|"
    r"提升|降低|超过|优于|胜过|达到|改善|"
    r"缩短|更强|更鲁棒|鲁棒性|更准确|有效|可行|一致|相当|稳定|保持|保留|区分|远未解决|缺陷|"
    r"outperform|improv|reduc|increase|decrease|compar|evaluat|report|"
    r"faster|lower|higher|robust|competitive|%",
    re.IGNORECASE,
)

GENERIC_FACT_VALUES = {
    "ai",
    "llm",
    "vla",
    "人工智能",
    "模型",
    "方法",
    "框架",
    "论文",
    "论文中的方法与实验",
    "模型/研究",
    "世界模型",
    "具身智能",
    "机器人",
    "基础模型",
}

NON_RESULT_UNIT_PATTERN = re.compile(
    r"^\s*[\$￥]?\d+(?:[.,]\d+)?\s*(?:k|m|b|million|billion|参数|parameters?|tokens?|小时|hours?|分钟|minutes?|米|cm|mm)\s*$",
    re.IGNORECASE,
)


def _clean(value: Any, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"^[：:;；,\s-]+", "", text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip("，,。.;；:： ") + "…"


def _clean_complete(value: Any, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"^[：:;；,\s-]+", "", text)
    text = re.split(r"…|\.\.\.", text, maxsplit=1)[0].rstrip("，,。.;；:： ")
    if len(text) <= limit:
        return text
    clipped = text[:limit]
    boundary = max(clipped.rfind(mark) for mark in ("。", "；", ";", "，", ","))
    if boundary >= max(24, int(limit * 0.45)):
        return clipped[:boundary].rstrip("，,。.;；:： ")
    if re.search(r"[A-Za-z0-9]$", clipped) and len(text) > limit and re.match(r"[A-Za-z0-9]", text[limit]):
        token_trimmed = re.sub(r"\s+\S+$", "", clipped).rstrip()
        if token_trimmed and token_trimmed != clipped:
            clipped = token_trimmed
        else:
            clipped = re.sub(r"[A-Za-z0-9._+\-]+$", "", clipped).rstrip("，,、和与及 ")
    return clipped.rstrip("，,。.;；:： ")


def _facts(item: Dict[str, Any]) -> Dict[str, Any]:
    facts_cn = item.get("facts_cn")
    if isinstance(facts_cn, dict) and facts_cn:
        raw_facts = item.get("facts")
        merged = dict(facts_cn)
        if isinstance(raw_facts, dict):
            for key in (
                "who",
                "action",
                "target",
                "claim_type",
                "source_excerpt",
                "evidence_locator",
                "primary_section",
                "technical_category",
                "paper_domain_key",
            ):
                if merged.get(key) in (None, "", [], {}) and raw_facts.get(key) not in (
                    None,
                    "",
                    [],
                    {},
                ):
                    merged[key] = raw_facts[key]
        return merged
    facts = item.get("facts")
    return facts if isinstance(facts, dict) else {}


def _raw_facts(item: Dict[str, Any]) -> Dict[str, Any]:
    facts = item.get("facts")
    return facts if isinstance(facts, dict) else {}


def has_untranslated_prose(text: Any) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    if re.search(
        r"funding round|product update|industry update|strategic acquisition|letter of intent|customer workflow",
        value,
        re.IGNORECASE,
    ):
        return True
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9'./_-]*\b", value)
    ascii_letters = len(re.findall(r"[A-Za-z]", value))
    cjk = len(re.findall(r"[\u4e00-\u9fff]", value))
    proper_name_connectors = {"and", "for", "of", "to", "with", "by", "on"}
    if cjk >= 8 and words and all(
        word.lower() in proper_name_connectors or word[0].isupper() or word.isupper()
        for word in words
    ):
        return False
    return (len(words) >= 5 and cjk < 6) or (len(words) >= 6 and ascii_letters >= max(32, cjk * 2))


def _translate_fact_text(value: Any, *, kind: str = "", fallback: str = "") -> str:
    text = _clean_complete(value, 150)
    if not text:
        return fallback
    if len(re.findall(r"[\u4e00-\u9fff]", text)) >= 4 and not has_untranslated_prose(text):
        return text

    lowered = text.lower()
    phrase_map = (
        ("world models", "世界模型"),
        ("world model", "世界模型"),
        ("video prediction", "视频预测"),
        ("latent dynamics", "潜空间动态建模"),
        ("robot manipulation", "机器人操作任务"),
        ("vision-language-action", "视觉-语言-动作模型"),
        ("reinforcement learning", "强化学习"),
        ("multi-agent", "多智能体"),
        ("open-source", "开源"),
        ("open source", "开源"),
        ("customer workflow", "客户工作流"),
        ("enterprise workflow", "企业工作流"),
        ("real-world", "真实环境"),
        ("success rate", "成功率"),
        ("funding round", "新一轮融资"),
        ("product update", "产品更新"),
        ("strategic acquisition", "战略收购"),
        ("letter of intent", "意向书"),
    )
    translated = text
    for source, target in phrase_map:
        translated = re.sub(re.escape(source), target, translated, flags=re.IGNORECASE)
    if len(re.findall(r"[\u4e00-\u9fff]", translated)) >= 4 and not has_untranslated_prose(translated):
        return translated

    numbers = re.findall(r"(?:\$\s*)?\d+(?:\.\d+)?(?:\s?%|\s?[BMKbm]|\s?billion|\s?million)?", text)
    number = numbers[0].strip() if numbers else ""
    if kind == "metric" and number and re.search(r"[\u4e00-\u9fff]", text):
        return text
    if kind == "metric" and number:
        if "success" in lowered:
            return f"成功率达到 {number}"
        if "valuation" in lowered:
            return f"估值达到 {number}"
        if any(token in lowered for token in ("raised", "funding", "financing")):
            return f"融资金额为 {number}"
        if any(token in lowered for token in ("parameter", "model size")):
            return f"模型规模为 {number}"
        if any(token in lowered for token in ("latency", "throughput", "fps")):
            return f"性能指标达到 {number}"
    if kind == "method":
        if "world model" in lowered or "predict" in lowered:
            return "先预测环境后续状态，再把预测结果交给规划或控制模块"
        if "vision-language-action" in lowered or re.search(r"\bvla\b", lowered):
            return "把视觉、语言指令和动作策略接入同一模型链路"
        if "reinforcement learning" in lowered:
            return "通过强化学习在持续交互中优化策略"
        if "benchmark" in lowered or "evaluation" in lowered:
            return "统一任务设置、数据划分和指标后进行对照评测"
        if "multi-agent" in lowered:
            return "通过任务拆分、工具路由和状态协同组织多智能体流程"
    if kind in {"who", "benchmark", "project"}:
        word_count = len(re.findall(r"[A-Za-z][A-Za-z0-9.+#&/_-]*", text))
        if word_count <= 5 and len(text) <= 64:
            return text
    return fallback


def normalize_facts_cn(item: Dict[str, Any]) -> Dict[str, Any]:
    facts = _raw_facts(item)
    if not facts:
        return {}
    evidence = facts.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    target_fallback = _target_cn(item.get("display_topic") or item.get("category"), "")
    normalized_evidence = [
        _translate_fact_text(point, kind="metric")
        for point in evidence[:3]
    ]
    normalized_evidence = [point for point in normalized_evidence if point]
    result = {
        "who": _translate_fact_text(facts.get("who"), kind="who", fallback=""),
        "action": _action_cn(facts.get("action"), str(item.get("content_type") or "")),
        "target": _target_cn(facts.get("target"), target_fallback),
        "evidence": normalized_evidence,
        "audience": _translate_fact_text(facts.get("audience"), fallback=""),
        "research_problem": _translate_fact_text(facts.get("research_problem"), fallback=""),
        "core_method": _translate_fact_text(facts.get("core_method"), kind="method", fallback=""),
        "architecture": _translate_fact_text(facts.get("architecture"), kind="method", fallback=""),
        "training_objective": _translate_fact_text(facts.get("training_objective"), kind="method", fallback=""),
        "input_output": _translate_fact_text(facts.get("input_output"), kind="method", fallback=""),
        "method": _translate_fact_text(facts.get("method"), kind="method", fallback=""),
        "dataset_or_benchmark": _translate_fact_text(facts.get("dataset_or_benchmark"), kind="benchmark", fallback=""),
        "metric_result": _translate_fact_text(facts.get("metric_result"), kind="metric", fallback=""),
        "baseline": _translate_fact_text(facts.get("baseline"), kind="benchmark", fallback=""),
        "limitation": _translate_fact_text(facts.get("limitation"), fallback=""),
        "code_or_project": _translate_fact_text(facts.get("code_or_project"), kind="project", fallback=""),
        "deployment_context": _translate_fact_text(facts.get("deployment_context"), fallback=""),
        "confidence": facts.get("confidence", 0),
    }
    if not result["who"]:
        result["who"] = _translate_fact_text(item.get("source_detail") or item.get("platform"), kind="who", fallback="")
    return result


def _evidence_list(facts: Dict[str, Any]) -> List[str]:
    evidence = facts.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    return [_clean_complete(point, 120) for point in evidence if str(point or "").strip()]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def contains_mojibake(text: Any) -> bool:
    value = str(text or "")
    if not value:
        return False
    hits = sum(1 for marker in MOJIBAKE_MARKERS if marker in value)
    return hits >= 2 or "�" in value


def has_field_label_leak(text: Any) -> bool:
    value = str(text or "")
    return any(token in value for token in FIELD_LABEL_LEAKS)


def has_bad_public_phrase(text: Any) -> bool:
    value = str(text or "")
    if any(token in value for token in BAD_PUBLIC_PHRASES):
        return True
    if re.search(
        r"(发布|推出|上线|开源|更新)(?:产品|模型|系统|项目|功能)?\1",
        value,
        re.IGNORECASE,
    ):
        return True
    return bool(
        re.search(
            r"\b([A-Za-z][A-Za-z0-9_.+-]{1,30})\s*"
            r"(?:发布|推出|上线|开源|更新|release[sd]?|launch(?:es|ed)?)\s*"
            r"\1\b",
            value,
            re.IGNORECASE,
        )
    )


def _normalized_fact_value(value: Any) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").lower())


def _generic_fact_value(value: Any) -> bool:
    normalized = _normalized_fact_value(value)
    if not normalized:
        return True
    generic = {_normalized_fact_value(item) for item in GENERIC_FACT_VALUES}
    return normalized in generic or normalized.startswith("论文中的")


def _claim_numbers(value: Any) -> List[str]:
    return [
        re.sub(r"[,\s]", "", token).lower()
        for token in re.findall(
            r"[\$￥]?\d+(?:[.,]\d+)?\s*(?:%|k|m|b|million|billion|万|亿|参数|parameters?|tokens?|小时|hours?|分钟|minutes?|米|cm|mm)?",
            str(value or ""),
            re.IGNORECASE,
        )
        if token.strip()
    ]


def _numbers_supported_by_source(value: Any, source_text: str) -> bool:
    claim_numbers = _claim_numbers(value)
    if not claim_numbers:
        return True
    normalized_source = re.sub(r"[,\s]", "", source_text.lower())
    return all(token in normalized_source for token in claim_numbers)


def assess_fact_publishability(item: Dict[str, Any], facts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Decide whether structured facts are safe enough to expand into public prose."""
    facts = facts if isinstance(facts, dict) else _facts(item)
    content_type = str(item.get("content_type") or "news").lower()
    model_used = str(item.get("model_used") or "")
    raw_facts = _raw_facts(item)
    has_source_body = bool(str(item.get("content") or "").strip())
    source_text = re.sub(
        r"\s+",
        " ",
        " ".join(str(item.get(key) or "") for key in ("title", "content", "source_detail")),
    ).strip()
    reasons: List[str] = []

    if not facts or not all(str(facts.get(key) or "").strip() for key in ("who", "action", "target")):
        reasons.append("missing_key_facts")
    if model_used == "template_fallback":
        reasons.append("template_fallback")

    evidence = _evidence_list(facts)
    if not evidence and not has_source_body and model_used != "template_fallback":
        evidence = _evidence_list(raw_facts)
    supported_evidence = [
        point for point in evidence
        if len(point) >= 4
        and (
            _numbers_supported_by_source(point, source_text)
            or (not has_source_body and model_used != "template_fallback")
        )
    ]
    if not supported_evidence:
        reasons.append("unsupported_evidence")

    unsupported_numeric = has_source_body and any(
        str(facts.get(key) or "").strip()
        and not _numbers_supported_by_source(facts.get(key), source_text)
        for key in ("metric_result", "dataset_or_benchmark", "evidence")
    )
    if unsupported_numeric:
        reasons.append("unsupported_numeric_claim")

    strict_paper_validation = bool(
        item.get("_strict_evidence_editor")
        or model_used == "template_fallback"
        or has_source_body
    )
    if content_type == "paper" and strict_paper_validation:
        method = str(
            facts.get("core_method")
            or facts.get("method")
            or raw_facts.get("core_method")
            or raw_facts.get("method")
            or ""
        ).strip()
        result = str(facts.get("metric_result") or raw_facts.get("metric_result") or "").strip()
        if len(method) < 12 or _generic_fact_value(method) or not METHOD_PATTERN.search(method):
            reasons.append("generic_method")
        if (
            not result
            or _generic_fact_value(result)
            or NON_RESULT_UNIT_PATTERN.fullmatch(result)
            or not PAPER_OUTCOME_PATTERN.search(result)
        ):
            reasons.append("invalid_result")
        elif has_source_body and not _numbers_supported_by_source(result, source_text):
            reasons.append("unsupported_numeric_claim")

    reasons = sorted(set(reasons))
    if not reasons:
        tier = "editorial_ready"
    elif content_type == "paper":
        tier = "index_only"
    else:
        tier = "brief_only"
    return {"tier": tier, "reasons": reasons}


def mixed_language_title(title: Any) -> bool:
    value = str(title or "").strip()
    if not value:
        return True
    if contains_mojibake(value) or has_field_label_leak(value):
        return True
    cjk = len(re.findall(r"[\u4e00-\u9fff]", value))
    english_words = re.findall(r"[A-Za-z][A-Za-z0-9.+#_-]{2,}", value)
    proper_entity_prefix = re.match(
        r"^[A-Za-z0-9$^+@&._'()\-/ ]{2,64}(?=发布|推出|部署|合作|融资|提出|展示|表示|报道|披露|开源|更新)",
        value,
    )
    if proper_entity_prefix:
        suffix = value[proper_entity_prefix.end() :]
        suffix_words = re.findall(r"[A-Za-z][A-Za-z0-9.+#_-]{2,}", suffix)
        if len(re.findall(r"[\u4e00-\u9fff]", suffix)) >= 2 and len(suffix_words) <= 2:
            return False
        if len(suffix_words) > 2:
            return True
    action_stems = re.compile(
        r"\b(demonstrat|propos|publish|launch|release|deploy|announce|introduc|unveil|weav|move|partner|raise|build)",
        re.IGNORECASE,
    )
    if cjk and action_stems.search(value):
        return True
    if not cjk and action_stems.search(value) and len(english_words) >= 4:
        return True
    if cjk >= 2 and len(english_words) >= 5:
        return True
    if not cjk and len(english_words) >= 12:
        return True
    return False


def _domain_key(item: Dict[str, Any]) -> str:
    existing = str(item.get("domain_key") or "")
    paper_domain_key = str(item.get("paper_domain_key") or "")
    if str(item.get("content_type") or "") == "paper" and paper_domain_key == "other":
        return "products_business"
    if str(item.get("content_type") or "") == "paper" and paper_domain_key in DOMAIN_ORDER:
        return paper_domain_key
    facts = _facts(item)
    evidence = " ".join(_evidence_list(facts))
    text = " ".join(
        str(item.get(key, "") or "")
        for key in ("title_cn", "title", "summary", "summary_preview", "category", "topic_cn", "display_topic")
    )
    text = f"{text} {facts.get('who', '')} {facts.get('action', '')} {facts.get('target', '')} {evidence}".lower()
    if any(token in text for token in ("world model", "world models", "video prediction", "latent dynamics", "planning", "世界模型")):
        return "world_model"
    if any(token in text for token in ("physical ai", "embodied", "robot", "robotics", "humanoid", "vla", "manipulation", "具身", "机器人")):
        return "physical_ai"
    if any(token in text for token in ("gpu", "chip", "inference", "datacenter", "open source", "github", "license", "infra", "算力", "推理", "开源", "cloudflare", "crawler", "ci/cd", "sdk", "typescript api", "数据中心")):
        return "infra_open_source"
    if any(token in text for token in ("funding", "valuation", "acquisition", "customer", "subscription", "hospital", "healthcare", "medical", "融资", "估值", "收购", "客户", "用户", "医院", "医疗", "订阅")):
        return "products_business"
    if any(token in text for token in ("agent", "workflow", "reasoning", "llm", "gpt", "claude", "gemini", "智能体", "模型")):
        return "agent_models"
    return existing if existing in DOMAIN_ORDER else "products_business"


def _action_cn(action: Any, content_type: str = "") -> str:
    raw = str(action or "").strip()
    lowered = raw.lower()
    mapping = (
        (("sign", "ink"), "签署"),
        (("launch", "release", "publish", "announce", "unveil"), "发布"),
        (("introduce", "roll out", "ship"), "推出"),
        (("deploy", "integrate", "weav"), "部署"),
        (("partner", "collaborat"), "合作推进"),
        (("raise", "funding", "invest"), "融资"),
        (("propos", "present"), "提出"),
        (("demonstrat", "show"), "展示"),
        (("open source", "github"), "开源"),
    )
    for stems, label in mapping:
        if any(stem in lowered for stem in stems):
            return label
    chinese_actions = (
        ("签署", "签署"),
        ("发布", "发布"),
        ("推出", "推出"),
        ("部署", "部署"),
        ("合作", "合作推进"),
        ("融资", "融资"),
        ("提出", "提出"),
        ("展示", "展示"),
        ("开源", "开源"),
    )
    for token, label in chinese_actions:
        if token in raw:
            return label
    if re.search(r"表示|认为|预测|主张|称", raw):
        return "表示"
    if re.search(r"报道|披露", raw):
        return "报道"
    if content_type == "paper":
        return "提出"
    if re.search(r"[\u4e00-\u9fff]", raw):
        return _clean(raw, 12)
    return "披露"


def _target_cn(value: Any, fallback: str = "这项进展") -> str:
    text = _clean(value, 80)
    if not text:
        return fallback
    lowered = text.lower()
    phrase_map = {
        "web data extraction": "网页数据抓取能力",
        "dense reasoning model": "密集推理模型",
        "telecom services": "电信服务",
        "world model and jepa": "世界模型和 JEPA 路线",
        "world model training": "世界模型训练",
        "latent dynamics": "潜空间动态建模",
        "video prediction": "视频预测",
        "scanpath prediction": "扫描路径预测",
        "liquid neural networks": "液态神经网络",
        "robot manipulation": "机器人操作任务",
        "real-world benchmark": "真实任务基准",
        "agent workflow": "智能体工作流",
        "funding round": "新一轮融资",
        "product update": "产品更新",
        "strategic acquisition": "战略收购",
        "letter of intent": "意向书",
        "physical ai": "具身智能评测",
    }
    for source, target in phrase_map.items():
        if source in lowered:
            return target
    replacements = {
        "Product Release": "产品更新",
        "AI领域新进展": fallback,
        "Industry Update": "行业动态",
        "For You": "个性化推荐",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"融资\s*(?:funding round)", "新一轮融资", text, flags=re.IGNORECASE)
    if contains_mojibake(text):
        return fallback
    if len(re.findall(r"[A-Za-z][A-Za-z0-9.+#_-]{2,}", text)) >= 3 and not re.search(r"[\u4e00-\u9fff]", text):
        return fallback
    return text


def _paper_target_cn(value: Any, fallback: str = "方法机制") -> str:
    text = _target_cn(value, fallback)
    text = re.sub(
        r"[A-Za-z][A-Za-z\- ]{4,}\s*\(([A-Z][A-Z0-9-]{1,10})\)",
        lambda match: match.group(1),
        text,
    )
    text = text.replace("在通用", "").replace("中的", "").replace("方面的", "")
    return _clean(text, 24)


def _dedupe_subject_target(subject: str, target: str) -> str:
    cleaned = str(target or "").strip()
    if not subject or not cleaned:
        return cleaned
    escaped = re.escape(subject.strip())
    cleaned = re.sub(rf"^{escaped}(?:平台|产品|模型)?\s*[，,:：-]?\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned or target


def _compose_title(subject: str, action: str, target: str, limit: int = 56) -> str:
    concise_target = re.split(r"[，；;。]", str(target or ""), maxsplit=1)[0].strip()
    concise_target = concise_target or str(target or "").strip()
    if action and action in concise_target:
        title = f"{subject}{concise_target}"
    else:
        title = f"{subject}{action}{concise_target}"
    if len(title) <= limit:
        return title
    budget = max(8, limit - len(subject) - len(action))
    return f"{subject}{action}{concise_target[:budget]}".rstrip("，,；;：: -")


def _subject_cn(item: Dict[str, Any]) -> str:
    facts = _facts(item)
    content_type = str(item.get("content_type") or "").strip().lower()
    if content_type in {"project", "open_source", "opensource"}:
        project = _clean_complete(facts.get("target") or facts.get("code_or_project"), 28)
        if project and not contains_mojibake(project) and not has_untranslated_prose(project):
            return project
    raw_who = re.sub(r"\s+", " ", str(facts.get("who") or "")).strip()
    if len(raw_who) > 36 and "；" in raw_who:
        raw_who = raw_who.split("；", 1)[0].strip()
    who = _clean(raw_who, 36)
    if who and who.lower() not in {"unknown", "this paper", "researchers"} and not contains_mojibake(who):
        return who
    title = _clean(item.get("title_cn") or item.get("title"), 42)
    if title and not contains_mojibake(title):
        return title
    source = _clean(item.get("source_detail") or item.get("platform"), 24)
    return source or "该来源"


def _paper_name(item: Dict[str, Any]) -> str:
    facts = _facts(item)
    who = _clean(facts.get("who"), 34)
    if who and who.lower() not in {
        "unknown",
        "this paper",
        "researchers",
        "physical ai",
        "world model",
        "robotics",
        "arxiv",
    } and not contains_mojibake(who):
        acronym = re.match(r"^\s*([A-Za-z][A-Za-z0-9$^+._-]{1,20})\s*\(", who)
        if acronym:
            return acronym.group(1)
        return re.sub(r"[（(](?:论文)?(?:方法|模型|系统)[）)]$", "", who).strip()
    title = _clean(item.get("title_cn") or item.get("title"), 54)
    if "：" in title:
        title = title.split("：", 1)[0]
    if ":" in title:
        title = title.split(":", 1)[0]
    return title or "这篇论文"


def _paper_mechanism(item: Dict[str, Any]) -> str:
    facts = _facts(item)
    text = " ".join(
        str(part or "")
        for part in (
            item.get("title_cn"),
            item.get("title"),
            item.get("summary"),
            item.get("summary_preview"),
            facts.get("method"),
            facts.get("target"),
            " ".join(_evidence_list(facts)),
        )
    ).lower()
    method = _clean_complete(facts.get("method"), 120)
    if method and not contains_mojibake(method) and not has_bad_public_phrase(method):
        return method
    if any(token in text for token in ("gazelnn", "scanpath", "liquid neural")):
        return "用轻量级液态神经网络预测视觉扫描路径，再用扫描路径指标评估模型是否接近人类注意力变化"
    if any(token in text for token in ("alignment", "preference", "instruction", "指令", "对齐")):
        return "把细粒度指令对齐转成更明确的操作约束，再让机器人策略按这些约束执行动作"
    if any(token in text for token in ("predict-then-act", "motion-aware", "dynamic manipulation", "frozen vla", "openvla")):
        return "先预测动态物体的下一步位置，再把预测结果交给冻结 VLA 做动作决策"
    if any(token in text for token in ("world model", "planning", "rollout", "prediction", "世界模型")):
        return "先预测后续状态或环境变化，再把预测结果交给规划、控制或策略模块做决策"
    if any(token in text for token in ("vla", "vision-language-action", "robot", "manipulation", "grasp")):
        return "把视觉、语言指令和动作策略接在同一条链路里，用任务描述直接约束机器人动作"
    if any(token in text for token in ("benchmark", "dataset", "leaderboard", "bench")):
        return "把任务、数据划分和指标统一起来，让不同模型能在同一条件下比较"
    if any(token in text for token in ("agent", "tool", "workflow")):
        return "把模型推理接入工具调用、状态记录和多步任务流程，而不是只生成一次性回答"
    target = _target_cn(facts.get("target") or item.get("summary_preview") or item.get("title_cn"), "研究问题")
    return f"把{target}拆成可测试的方法假设，再用实验结果判断是否成立"


def _paper_result(item: Dict[str, Any]) -> str:
    facts = _facts(item)
    value = _clean_complete(facts.get("metric_result"), 120)
    if value and not contains_mojibake(value) and not has_bad_public_phrase(value):
        return value
    evidence = _evidence_list(facts)
    for point in evidence:
        if re.search(r"\d", point):
            return point
    for point in evidence:
        if RESULT_PATTERN.search(point):
            return point
    if evidence:
        return evidence[0]
    return ""


def paper_technical_intro_passes(text: Any) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    if contains_mojibake(value) or has_bad_public_phrase(value) or has_field_label_leak(value) or has_untranslated_prose(value):
        return False
    if "围绕" in value and "设计" in value and "关键做法" not in value:
        return False
    if any(token in value for token in ("只停留在概念描述", "只能作为技术线索", "需要重点看数据集", "把研究问题拆成")):
        return False
    return bool(METHOD_PATTERN.search(value) and RESULT_PATTERN.search(value))


def paper_plain_summary_passes(text: Any) -> bool:
    value = str(text or "").strip()
    if not 90 <= len(value) <= 200:
        return False
    if contains_mojibake(value) or has_bad_public_phrase(value) or has_field_label_leak(value) or has_untranslated_prose(value):
        return False
    if value.count("。") < 2:
        return False
    return bool(METHOD_PATTERN.search(value))


def trusted_codex_research_item(item: Dict[str, Any], facts: Optional[Dict[str, Any]] = None) -> bool:
    """Recognize inbox rows that already passed the strict research collector checks."""
    if str(item.get("model_used") or "") != "codex-automation":
        return False
    if str(item.get("analysis_version") or "") not in {"codex-research-v2", "codex-research-v3"}:
        return False
    facts = facts if isinstance(facts, dict) else _facts(item)
    raw_facts = _raw_facts(item)
    if not all(str(raw_facts.get(key) or facts.get(key) or "").strip() for key in ("who", "action", "target")):
        return False
    if not (_evidence_list(raw_facts) or _evidence_list(facts)):
        return False
    if not str(raw_facts.get("source_excerpt") or facts.get("source_excerpt") or "").strip():
        return False
    if not str(raw_facts.get("evidence_locator") or facts.get("evidence_locator") or "").strip():
        return False
    if _safe_float(item.get("evidence_quality")) < 0.45:
        return False
    if _safe_float(item.get("information_density"), _safe_float(item.get("evidence_quality"))) < 0.45:
        return False
    if str(item.get("content_type") or "") == "paper":
        plain = raw_facts.get("paper_plain_summary") or item.get("paper_plain_summary")
        technical = raw_facts.get("paper_technical_intro") or item.get("paper_technical_intro")
        return paper_plain_summary_passes(plain) and paper_technical_intro_passes(technical)
    return True


class EditorialEngine:
    def __init__(self, *, physical_ai_min_items: int = 4, paper_technical_intro_min_count: int = 12) -> None:
        self.physical_ai_min_items = int(physical_ai_min_items or 4)
        self.paper_technical_intro_min_count = int(paper_technical_intro_min_count or 12)

    def decorate_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(item)
        content_type = str(result.get("content_type") or "").strip() or "news"
        result["facts_cn"] = normalize_facts_cn(result)
        facts = _facts(result)
        publishability = assess_fact_publishability(result, facts)
        trusted_codex_item = trusted_codex_research_item(result, facts)
        if trusted_codex_item:
            publishability = {"tier": "editorial_ready", "reasons": []}
            result["_codex_research_validated"] = True
        result["summary_quality_tier"] = publishability["tier"]
        result["summary_quality_reasons"] = publishability["reasons"]
        evidence = _evidence_list(facts)
        domain_key = _domain_key(result)
        result["domain_key"] = domain_key
        result["domain_label"] = DOMAIN_LABELS.get(domain_key, "Products / Business")
        raw_facts = _raw_facts(result)
        curated_plain_summary = str(
            raw_facts.get("paper_plain_summary") or result.get("paper_plain_summary") or ""
        ).strip()
        curated_technical_intro = str(
            raw_facts.get("paper_technical_intro") or result.get("paper_technical_intro") or ""
        ).strip()
        curated_codex_paper = bool(
            content_type == "paper"
            and str(result.get("model_used") or "") == "codex-automation"
            and paper_plain_summary_passes(curated_plain_summary)
            and paper_technical_intro_passes(curated_technical_intro)
        )
        if content_type == "paper" and str(result.get("model_used") or "") == "codex-automation":
            result["paper_plain_summary"] = curated_plain_summary
            result["paper_technical_intro"] = curated_technical_intro

        title = self.editorial_title(result)
        result["editorial_title"] = title
        result["title_cn"] = title

        evidence_line = self.evidence_line(result, evidence)
        result["evidence_line"] = evidence_line

        if content_type == "paper" and (publishability["tier"] == "editorial_ready" or curated_codex_paper):
            plain_summary = self.paper_plain_summary(result)
            intro = self.paper_technical_intro(result)
            compact_summary = self.paper_compact_summary(result)
            result["paper_plain_summary"] = plain_summary
            result["paper_technical_intro"] = intro
            result["paper_compact_summary"] = compact_summary
            result["editorial_lead"] = _clean(plain_summary.split("。", 1)[0], 120)
            result["analysis_body"] = plain_summary
            result["reader_next_step"] = self.paper_next_step(result)
        elif content_type == "paper":
            result["paper_plain_summary"] = ""
            result["paper_technical_intro"] = ""
            result["paper_compact_summary"] = ""
            result["paper_index_note"] = "仅保留原文入口，暂不生成未经来源支撑的技术总结。"
            result["editorial_lead"] = ""
            result["analysis_body"] = ""
            result["reader_next_step"] = "直接阅读原文摘要、方法与实验章节。"
        else:
            result["paper_plain_summary"] = ""
            result["paper_technical_intro"] = ""
            result["paper_compact_summary"] = ""
            result["editorial_lead"] = self.editorial_lead(result)
            result["analysis_body"] = self.analysis_body(result)
            result["reader_next_step"] = self.reader_next_step(result)

        result["quality_tier"] = self.quality_tier(result)
        flags = self.editorial_flags(result)
        result["editorial_flags"] = flags
        result["brief_line"] = self.brief_line(result)
        result["editorial_brief"] = result["analysis_body"] if result["quality_tier"] != "brief" else result["brief_line"]
        result["learning_takeaway"] = result["editorial_lead"]
        result["technical_context"] = result["paper_technical_intro"] or result["analysis_body"]
        result["background_context"] = result["evidence_line"]
        result["deep_dive_prompt"] = result["reader_next_step"]
        return result

    def editorial_title(self, item: Dict[str, Any]) -> str:
        original = _clean(item.get("title_cn") or item.get("title"), 72)
        facts = _facts(item)
        content_type = str(item.get("content_type") or "")
        force_fact_title = bool(item.get("_v8_force_fact_title"))
        trusted_research_title = str(item.get("model_used") or "") == "codex-automation"
        if (
            original
            and (not force_fact_title or trusted_research_title)
            and not mixed_language_title(original)
            and not has_bad_public_phrase(original)
            and "..." not in original
            and "…" not in original
            and not has_untranslated_prose(original)
            and not self._title_uses_truncated_source_prefix(original, item)
            and len(original) <= 56
        ):
            return original
        if content_type == "paper":
            name = _paper_name(item)
            target = _paper_target_cn(
                facts.get("target") or item.get("summary_preview") or item.get("title"),
                "方法机制",
            )
            if name in {"这篇论文", "Physical AI", "World Model", "Robotics", "arXiv"} or has_bad_public_phrase(name):
                return _compose_title("", "", f"{target}的新方法与实验")
            return _compose_title(name, "：", target)
        subject = _subject_cn(item)
        action = _action_cn(facts.get("action"), content_type)
        target = _target_cn(facts.get("target") or item.get("display_topic") or item.get("category"), "产品或技术变化")
        target = _dedupe_subject_target(subject, target)
        return _compose_title(subject, action, target)

    @staticmethod
    def _title_uses_truncated_source_prefix(title: str, item: Dict[str, Any]) -> bool:
        source_title = re.sub(r"\s+", " ", str(item.get("title") or "")).strip()
        prefix_match = re.match(r"^([^\u4e00-\u9fff]+)(?=[\u4e00-\u9fff])", str(title or ""))
        if not source_title or not prefix_match:
            return False
        prefix = prefix_match.group(1).rstrip()
        if not prefix or not source_title.lower().startswith(prefix.lower()) or len(prefix) >= len(source_title):
            return False
        return bool(re.match(r"[A-Za-z0-9]", source_title[len(prefix) :]))

    def editorial_lead(self, item: Dict[str, Any]) -> str:
        if (
            str(item.get("model_used") or "") == "codex-automation"
            and str(item.get("content_type") or "").strip().lower() in {"project", "open_source", "opensource"}
        ):
            curated = _clean_complete(item.get("editorial_title") or item.get("title_cn"), 110)
            if curated:
                return curated
        facts = _facts(item)
        subject = _subject_cn(item)
        action = _action_cn(facts.get("action"), str(item.get("content_type") or ""))
        target = _target_cn(facts.get("target") or item.get("display_topic") or item.get("category"), "这项变化")
        target = _dedupe_subject_target(subject, target)
        if action and target.startswith(action):
            target = target[len(action) :].lstrip("，,:：- ")
        return _clean(f"{subject}{action}{target}", 110)

    def evidence_line(self, item: Dict[str, Any], evidence: Optional[List[str]] = None) -> str:
        evidence = evidence if evidence is not None else _evidence_list(_facts(item))
        if not evidence:
            return "目前缺少实验、客户、代码或指标证据。"
        clean_points = [
            point
            for point in evidence
            if not contains_mojibake(point)
            and not has_bad_public_phrase(point)
            and not has_untranslated_prose(point)
        ]
        if not clean_points:
            return "材料提供了线索，但证据表述质量不足，需要降级阅读。"
        return _clean_complete("；".join(clean_points[:2]), 150)

    def analysis_body(self, item: Dict[str, Any]) -> str:
        raw_facts = _raw_facts(item)
        primary_section = str(
            item.get("primary_section") or raw_facts.get("primary_section") or ""
        ).strip().lower()
        if str(item.get("model_used") or "") == "codex-automation":
            content_type = str(item.get("content_type") or "").strip().lower()
            body_limit = 380 if primary_section == "technical" else 300
            if content_type in {"interview", "podcast", "video"}:
                body_limit = 500
            original_curated = str(item.get("summary") or "").strip()
            if (
                120 <= len(original_curated) <= body_limit
                and not contains_mojibake(original_curated)
                and not has_bad_public_phrase(original_curated)
                and not has_field_label_leak(original_curated)
            ):
                return original_curated
            curated = _clean_complete(original_curated, body_limit)
            if (
                len(curated) >= 120
                and not contains_mojibake(curated)
                and not has_bad_public_phrase(curated)
                and not has_field_label_leak(curated)
            ):
                return curated
        facts = _facts(item)
        evidence_line = str(item.get("evidence_line") or self.evidence_line(item))
        domain_key = str(item.get("domain_key") or _domain_key(item))
        subject = _subject_cn(item)
        action = _action_cn(facts.get("action"), str(item.get("content_type") or ""))
        target = _target_cn(facts.get("target") or item.get("display_topic") or item.get("category"), "这项变化")
        target = _dedupe_subject_target(subject, target)
        if action and target.startswith(action):
            target = target[len(action) :].lstrip("，,:：- ")
        if (
            str(item.get("model_used") or "") == "codex-automation"
            and str(item.get("content_type") or "").strip().lower() in {"project", "open_source", "opensource"}
        ):
            lead = _clean_complete(item.get("editorial_title") or item.get("title_cn"), 110)
        else:
            lead = f"{subject}{action}{target}"
        method = _clean_complete(facts.get("core_method") or facts.get("method"), 92)
        deployment = _clean_complete(facts.get("deployment_context"), 64)
        if "（" not in deployment:
            deployment = deployment.replace("）", "")
        if "(" not in deployment:
            deployment = deployment.replace(")", "")
        baseline = _clean_complete(facts.get("baseline"), 54)
        metric = _clean_complete(facts.get("metric_result"), 64)
        code = _clean_complete(facts.get("code_or_project"), 56)
        evidence = _evidence_list(facts)
        content_text = " ".join(str(item.get(key, "") or "") for key in ("title", "summary", "summary_preview", "category")).lower()
        if self.quality_tier(item) == "brief":
            return self.brief_line(item)
        if "interview" in content_text or "访谈" in content_text or "观点" in content_text:
            return _clean_complete(f"{subject}在访谈中主张{target}。这是受访者的判断，支撑它的公开材料包括{evidence_line}", 180)

        details: List[str] = []
        if method:
            details.append(f"实现上，{method}")
        metric_has_number = bool(re.search(r"\d|%|倍|亿|万|million|billion", metric, re.IGNORECASE))
        metric_has_outcome = bool(PAPER_OUTCOME_PATTERN.search(metric))
        if metric and metric not in method and (metric_has_number or metric_has_outcome):
            details.append(f"{'关键数字为' if metric_has_number else '公开结果为'}{metric}")
        elif evidence:
            numeric_evidence = next(
                (point for point in evidence if re.search(r"\d|%|倍|亿|万|million|billion", point, re.IGNORECASE)),
                "",
            )
            if numeric_evidence:
                details.append(f"公开信息显示，{_clean_complete(numeric_evidence, 72)}")
        if deployment and deployment not in target:
            if not re.search(r"挣扎|失败|困难|缺少|不足", deployment):
                if deployment.startswith("面向"):
                    details.append(f"它{deployment}")
                elif deployment.startswith("部署在"):
                    details.append(f"系统{deployment}")
                else:
                    details.append(f"应用场景是{deployment}")
        if baseline and baseline not in method:
            details.append(f"它对照的是{baseline}")
        if code and code.lower() not in {"open-source", "open source", "github", "开源"}:
            details.append(f"代码或项目入口指向{code}")
        if not details and evidence:
            details.append(f"公开信息显示，{evidence[0]}")

        detail_text = "。".join(details[:2])
        if detail_text:
            detail_text += "。"
        return _clean_complete(f"{lead}。{detail_text}", 210 if domain_key != "products_business" else 220)

    def paper_plain_summary(self, item: Dict[str, Any]) -> str:
        raw_existing = str(item.get("paper_plain_summary") or "").strip()
        if paper_plain_summary_passes(raw_existing):
            return raw_existing
        existing = _clean_complete(raw_existing, 180)
        if paper_plain_summary_passes(existing):
            return existing
        facts = _facts(item)
        raw_problem = facts.get("research_problem") or facts.get("target") or item.get("summary_preview")
        raw_method = facts.get("core_method") or facts.get("method") or _paper_mechanism(item)
        raw_result = _paper_result(item)
        if not raw_problem or not raw_method or not raw_result:
            return ""

        name = _paper_name(item)
        opening_index = sum(ord(char) for char in name) % 6
        opening_templates = (
            "这篇论文想解决的问题是：{}",
            "{}，是这篇论文要处理的难点",
            "论文先把问题落在{}",
            "研究团队关注的是{}",
            "这项工作从{}切入",
            "这篇研究讨论{}",
        )

        def build_candidate(
            problem_limit: int,
            method_limit: int,
            benchmark_limit: int,
            baseline_limit: int,
            result_limit: int,
        ) -> str:
            def compact_complete(source: Any, limit: int, fallback: Any = "") -> str:
                text = re.sub(r"\s+", " ", str(source or "")).strip()
                if len(text) <= limit:
                    return text.rstrip("且和与或并、 ")
                parts = [part.strip() for part in re.split(r"[，；;]", text) if part.strip()]
                selected: List[str] = []
                for part in parts:
                    joined = "，".join(selected + [part])
                    if len(joined) > limit:
                        break
                    selected.append(part)
                if selected:
                    return "，".join(selected).rstrip("且和与或并、 ")
                fallback_text = re.sub(r"\s+", " ", str(fallback or "")).strip()
                if 6 <= len(fallback_text) <= limit:
                    return fallback_text.rstrip("且和与或并、 ")
                return _clean_complete(text, limit).rstrip("且和与或并、 ")

            problem_source = str(raw_problem or "")
            problem_head = re.split(r"[，；;]", problem_source, maxsplit=1)[0].strip()
            if len(problem_head) >= 12 and not re.search(r"[且和与或并]$", problem_head):
                problem_source = problem_head
            target_fallback = facts.get("target") or ""
            problem = compact_complete(problem_source, problem_limit, target_fallback)
            method = compact_complete(raw_method, method_limit)
            result_source = str(raw_result or "").strip()
            result_clauses = [
                clause.strip()
                for clause in re.split(r"[；;]", result_source)
                if clause.strip()
            ]
            outcome_clause = next(
                (clause for clause in result_clauses if PAPER_OUTCOME_PATTERN.search(clause)),
                "",
            )
            if outcome_clause:
                result_source = outcome_clause
            result_parts = [
                part.strip()
                for part in re.split(r"[，,]", result_source)
                if part.strip()
            ]
            focused_result = next(
                (part for part in result_parts if PAPER_OUTCOME_PATTERN.search(part)),
                "",
            )
            if focused_result:
                result_source = focused_result
            result = compact_complete(result_source, result_limit)
            benchmark_source = str(facts.get("dataset_or_benchmark") or "").strip()
            benchmark_source = re.sub(r"[（(].*$", "", benchmark_source).strip()
            benchmark_head = re.split(r"[：:，；;]", benchmark_source, maxsplit=1)[0].strip()
            if len(benchmark_head) >= 4:
                benchmark_source = benchmark_head
            if "、" in benchmark_source:
                benchmark_parts = [part.strip() for part in benchmark_source.split("、") if part.strip()]
                benchmark_source = "与".join(benchmark_parts[:2])
            baseline_source = re.sub(r"[（(].*$", "", str(facts.get("baseline") or "")).strip()
            if baseline_source in {"未明确提及", "未说明", "无", "不适用"}:
                baseline_source = ""
            benchmark = _clean_complete(benchmark_source, benchmark_limit)
            baseline = _clean_complete(baseline_source, baseline_limit)
            opening = opening_templates[opening_index].format(problem)
            method_templates = (
                "方法上，论文采用{}",
                "作者采用{}",
                "它的关键步骤是{}",
                "具体实现是{}",
                "为处理这个问题，论文{}",
                "研究中的做法是{}",
            )
            method_sentence = method_templates[opening_index].format(method)
            benchmark_location = benchmark if benchmark.endswith(("上", "中", "内")) else f"{benchmark}上"
            if benchmark and baseline:
                result_sentence = f"在{benchmark_location}与{baseline}比较，论文报告{result}"
            elif benchmark:
                result_sentence = f"在{benchmark_location}验证后，论文报告{result}"
            elif baseline:
                result_sentence = f"与{baseline}相比，论文报告{result}"
            else:
                result_sentence = f"实验中，论文报告{result}"
            return "。".join((opening, method_sentence, result_sentence)).rstrip("。") + "。"

        candidate = ""
        for limits in (
            (46, 68, 32, 30, 58),
            (40, 58, 27, 25, 52),
            (34, 50, 22, 20, 46),
            (30, 44, 20, 18, 42),
            (26, 38, 18, 16, 38),
        ):
            candidate = build_candidate(*limits)
            if len(candidate) <= 170:
                break

        limitation = _clean_complete(facts.get("limitation"), 44)
        if len(candidate) < 100 and limitation:
            candidate += f"目前结论的边界是{limitation}。"
        if len(candidate) < 90:
            evidence = _evidence_list(facts)
            extra = next((point for point in evidence if point and point not in str(raw_result)), "")
            if extra:
                candidate += f"原文还报告了{_clean_complete(extra, 48)}。"
        if len(candidate) < 60:
            return ""
        return _clean_complete(candidate, 170).rstrip("。") + "。"

    def paper_compact_summary(self, item: Dict[str, Any]) -> str:
        facts = _facts(item)
        name = _paper_name(item)
        target = _clean_complete(
            facts.get("research_problem") or facts.get("target") or item.get("summary_preview"),
            48,
        )
        method = _clean_complete(facts.get("core_method") or facts.get("method"), 58)
        result = _clean_complete(_paper_result(item), 52)
        if not target:
            return ""
        if method and result:
            text = f"{name}研究{target}，方法是{method}；原文报告{result}。"
        elif method:
            text = f"{name}研究{target}，核心方法是{method}。"
        elif result:
            text = f"{name}研究{target}；原文目前能确认的结果是{result}。"
        else:
            evidence = _evidence_list(facts)
            if not evidence:
                return ""
            text = f"{name}研究{target}；原文给出的事实是{_clean_complete(evidence[0], 52)}。"
        text = _clean_complete(text, 96).rstrip("。") + "。"
        return text if len(text) >= 36 else ""

    def paper_technical_intro(self, item: Dict[str, Any]) -> str:
        raw_existing = str(item.get("paper_technical_intro") or "").strip()
        if str(item.get("model_used") or "") == "codex-automation" and paper_technical_intro_passes(raw_existing):
            return raw_existing
        existing = _clean_complete(raw_existing, 300)
        facts = _facts(item)
        name = _paper_name(item)
        fact_mechanism = _clean_complete(facts.get("core_method") or facts.get("method"), 88)
        fact_result = _paper_result(item)
        if (
            paper_technical_intro_passes(existing)
            and (
                str(item.get("model_used") or "") == "codex-automation"
                or not (fact_mechanism and fact_result)
            )
        ):
            return existing
        mechanism = fact_mechanism or _paper_mechanism(item)
        result = fact_result
        architecture = _clean_complete(facts.get("architecture"), 72)
        objective = _clean_complete(facts.get("training_objective"), 72)
        input_output = _clean_complete(facts.get("input_output"), 72)
        baseline = _clean_complete(facts.get("baseline"), 48)
        benchmark = _clean_complete(facts.get("dataset_or_benchmark"), 52)
        limitation = _clean_complete(facts.get("limitation"), 54)
        if re.search(r"未明确|未提及|未说明|没有提供|not mentioned|not provided|unspecified", baseline, re.IGNORECASE):
            baseline = ""
        if re.search(r"未提及|未说明|没有提供|not mentioned|not provided", limitation, re.IGNORECASE):
            limitation = ""
        if not mechanism or not result:
            return ""
        mechanism = _clean_complete(mechanism, 88)
        result = _clean_complete(result, 82)
        first_parts = [mechanism]
        if architecture and architecture not in mechanism:
            first_parts.append(f"整体由{architecture}构成")
        if objective and objective not in mechanism and objective not in architecture:
            first_parts.append(f"优化目标或训练方式是{objective}")
        sentences = [f"{name}采用{'；'.join(first_parts)}"]
        if input_output and input_output not in " ".join(first_parts):
            sentences.append(f"输入输出关系是：{input_output}")
        setting = benchmark if benchmark and not contains_mojibake(benchmark) else ""
        comparison = baseline if baseline and not contains_mojibake(baseline) else ""
        descriptive_comparison = bool(re.search(r"通常|可能|缺乏|不足|难以|无法", comparison))
        if setting and comparison:
            if descriptive_comparison:
                sentences.append(f"实验在 {setting} 上验证；现有做法{comparison}，论文结果是{result}")
            else:
                sentences.append(f"实验在 {setting} 上以 {comparison} 为对照，结果是{result}")
        elif setting:
            sentences.append(f"实验在 {setting} 上验证，结果是{result}")
        elif comparison:
            sentences.append(f"与{comparison}相比，实验结果是{result}")
        else:
            sentences.append(f"实验结果是{result}")
        base_sentences = list(sentences)
        if limitation and not contains_mojibake(limitation):
            sentences.append(f"论文明确的局限是{limitation}")
        for candidate_sentences in (sentences, base_sentences):
            candidate = "。".join(candidate_sentences) + "。"
            if len(candidate) <= 300:
                return candidate
        compact = f"{name}的关键做法是{mechanism}。实验结果是{result}。"
        return _clean_complete(compact, 300).rstrip("。") + "。"

    def paper_next_step(self, item: Dict[str, Any]) -> str:
        facts = _facts(item)
        if facts.get("code_or_project"):
            return "先看代码、数据和第三方复现是否支持论文结论。"
        return "先看方法图、实验表格、对照基线和失败案例。"

    def reader_next_step(self, item: Dict[str, Any]) -> str:
        if self.quality_tier(item) == "brief":
            return "只作为线索保留，等原始来源给出更多证据。"
        domain_key = str(item.get("domain_key") or _domain_key(item))
        if domain_key == "physical_ai":
            return "继续看真实机器人任务、部署环境、成功率和第三方评测。"
        if domain_key == "world_model":
            return "继续核对预测模块对规划、控制或智能体决策的实际增益。"
        if domain_key == "infra_open_source":
            return "继续看成本、延迟、许可证、代码活跃度和硬件适配。"
        return "继续看客户、价格、使用数据或产品细节是否补齐。"

    def quality_tier(self, item: Dict[str, Any]) -> str:
        if item.get("_codex_research_validated"):
            return "deep" if item.get("content_type") == "paper" else "standard"
        summary_quality_tier = str(item.get("summary_quality_tier") or "")
        if summary_quality_tier in {"index_only", "brief_only"}:
            return "brief"
        facts = _facts(item)
        evidence_quality = _safe_float(item.get("evidence_quality"))
        density = _safe_float(item.get("information_density"), evidence_quality)
        if not facts or not facts.get("who") or not facts.get("action") or not facts.get("target") or not facts.get("evidence"):
            return "brief"
        if evidence_quality < 0.45 or density < 0.45:
            return "brief"
        if item.get("content_type") == "paper" and paper_technical_intro_passes(str(item.get("paper_technical_intro") or "")):
            return "deep"
        return "standard"

    def brief_line(self, item: Dict[str, Any]) -> str:
        title = str(item.get("editorial_title") or self.editorial_title(item))
        if self.quality_tier(item) == "brief":
            return _clean(f"来源称{title}；目前缺少实验、客户、代码或指标证据。", 150)
        lead = str(item.get("editorial_lead") or title)
        evidence = str(item.get("evidence_line") or "")
        return _clean(f"{lead}；证据：{evidence}", 160)

    def editorial_flags(self, item: Dict[str, Any]) -> List[str]:
        fields = [
            item.get("editorial_title"),
            item.get("editorial_lead"),
            item.get("analysis_body"),
            item.get("evidence_line"),
            item.get("paper_plain_summary"),
            item.get("paper_technical_intro"),
            item.get("paper_compact_summary"),
            item.get("reader_next_step"),
        ]
        flags: List[str] = []
        if mixed_language_title(item.get("editorial_title") or item.get("title_cn") or item.get("title")):
            flags.append("mixed_language_title")
        if any(has_field_label_leak(value) for value in fields):
            flags.append("field_label_leak")
        if any(contains_mojibake(value) for value in fields):
            flags.append("mojibake_suspect")
        if any(has_bad_public_phrase(value) for value in fields):
            flags.append("bad_public_phrase")
        if any(has_untranslated_prose(value) for value in fields):
            flags.append("untranslated_fact")
        if item.get("quality_tier") == "brief" and len(str(item.get("analysis_body") or "")) > 170:
            flags.append("low_info_expanded")
        if item.get("content_type") == "paper" and not paper_technical_intro_passes(item.get("paper_technical_intro")):
            flags.append("paper_technical_intro_fail")
        if item.get("content_type") == "paper":
            facts = _facts(item)
            if not paper_plain_summary_passes(item.get("paper_plain_summary")):
                flags.append("paper_plain_summary_fail")
            if not str(facts.get("method") or "").strip():
                flags.append("paper_mechanism_missing")
            if not any(str(facts.get(key) or "").strip() for key in ("metric_result", "dataset_or_benchmark", "baseline")):
                flags.append("paper_result_context_missing")
        flags.extend(str(reason) for reason in item.get("summary_quality_reasons") or [])
        return sorted(set(flags))

    def decorate_items(self, items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.decorate_item(item) for item in items]

    def quality_metrics(self, items: Iterable[Dict[str, Any]], *, focus_sections: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        focus_sections = set(focus_sections or {"must_read", "physical_ai", "watch", "featured_papers"})
        decorated = [self.decorate_item(item) for item in items]

        def visible_untranslated(item: Dict[str, Any]) -> bool:
            if item.get("source_grounded_brief") and str(item.get("report_section") or "") == "brief":
                return False
            if str(item.get("report_section") or "") != "paper_appendix":
                return "untranslated_fact" in item.get("editorial_flags", [])
            return any(
                has_untranslated_prose(value)
                for value in (
                    item.get("editorial_title") or item.get("title_cn") or item.get("title"),
                    item.get("paper_compact_summary"),
                )
            )

        paper_items = [
            item
            for item in decorated
            if item.get("content_type") == "paper"
            and str(item.get("report_section") or "featured_papers") != "paper_appendix"
        ]
        paper_pass = sum(1 for item in paper_items if paper_technical_intro_passes(item.get("paper_technical_intro")))
        paper_plain_pass = sum(1 for item in paper_items if paper_plain_summary_passes(item.get("paper_plain_summary")))
        focus_items = [item for item in decorated if str(item.get("report_section") or "") in focus_sections]
        physical_ai_featured = sum(
            1
            for item in decorated
            if (
                str(item.get("report_section") or "") == "physical_ai"
                or str(item.get("domain_key") or _domain_key(item)) == "physical_ai"
            )
            and str(item.get("report_section") or "") != "brief"
            and (
                str(item.get("report_section") or "") == "physical_ai"
                or item.get("quality_tier") != "brief"
            )
        )
        gpt_items = [
            item for item in decorated
            if str(item.get("model_used") or "").lower().startswith("gpt-")
        ]
        llm_items = [
            item for item in decorated
            if str(item.get("model_used") or "")
            and str(item.get("model_used") or "") != "template_fallback"
        ]
        metrics = {
            "editorial_quality_status": "passed",
            "mixed_language_title_count": sum(1 for item in decorated if "mixed_language_title" in item.get("editorial_flags", [])),
            "field_label_leak_count": sum(1 for item in decorated if "field_label_leak" in item.get("editorial_flags", [])),
            "low_info_expanded_count": sum(1 for item in decorated if "low_info_expanded" in item.get("editorial_flags", [])),
            "mojibake_suspect_count": sum(1 for item in decorated if "mojibake_suspect" in item.get("editorial_flags", [])),
            "untranslated_fact_count": sum(1 for item in decorated if visible_untranslated(item)),
            "paper_mechanism_missing_count": sum(
                1 for item in paper_items if "paper_mechanism_missing" in item.get("editorial_flags", [])
            ),
            "paper_result_context_missing_count": sum(
                1 for item in paper_items if "paper_result_context_missing" in item.get("editorial_flags", [])
            ),
            "paper_technical_intro_pass_count": paper_pass,
            "paper_technical_intro_fail_count": max(0, len(paper_items) - paper_pass),
            "paper_plain_summary_pass_count": paper_plain_pass,
            "paper_plain_summary_fail_count": max(0, len(paper_items) - paper_plain_pass),
            "paper_plain_summary_pass_rate": round(paper_plain_pass / len(paper_items), 3) if paper_items else 1.0,
            "physical_ai_featured_count": physical_ai_featured,
            "deepseek_schema_valid_count": sum(1 for item in decorated if str(item.get("model_used") or "") == "deepseek-v4-pro" and bool(_facts(item))),
            "deepseek_empty_facts_count": sum(1 for item in decorated if str(item.get("model_used") or "") == "deepseek-v4-pro" and not bool(_facts(item))),
            "deepseek_key_field_missing_count": sum(
                1
                for item in decorated
                if str(item.get("model_used") or "") == "deepseek-v4-pro"
                and not all(_facts(item).get(key) for key in ("who", "action", "target"))
            ),
            "gpt_schema_valid_count": sum(1 for item in gpt_items if bool(_facts(item))),
            "gpt_empty_facts_count": sum(1 for item in gpt_items if not bool(_facts(item))),
            "gpt_key_field_missing_count": sum(
                1
                for item in gpt_items
                if not all(_facts(item).get(key) for key in ("who", "action", "target"))
            ),
            "llm_schema_valid_count": sum(1 for item in llm_items if bool(_facts(item))),
            "llm_empty_facts_count": sum(1 for item in llm_items if not bool(_facts(item))),
            "llm_key_field_missing_count": sum(
                1
                for item in llm_items
                if not all(_facts(item).get(key) for key in ("who", "action", "target"))
            ),
            "template_fallback_count": sum(
                1 for item in decorated if str(item.get("model_used") or "") == "template_fallback"
            ),
            "index_only_paper_count": sum(
                1 for item in decorated if str(item.get("summary_quality_tier") or "") == "index_only"
            ),
            "generic_fact_bundle_count": sum(
                1 for item in decorated if "generic_method" in item.get("editorial_flags", [])
            ),
            "unsupported_numeric_claim_count": sum(
                1 for item in decorated if "unsupported_numeric_claim" in item.get("editorial_flags", [])
            ),
            "failed_editorial_urls": [
                str(item.get("url") or "")
                for item in focus_items
                if item.get("quality_tier") == "brief" or item.get("editorial_flags")
            ],
        }
        if (
            metrics["mixed_language_title_count"]
            or metrics["field_label_leak_count"]
            or metrics["low_info_expanded_count"]
            or metrics["mojibake_suspect_count"]
            or metrics["untranslated_fact_count"]
            or metrics["paper_technical_intro_fail_count"] > max(0, len(paper_items) - self.paper_technical_intro_min_count)
            or metrics["paper_plain_summary_fail_count"]
            or metrics["physical_ai_featured_count"] < self.physical_ai_min_items
        ):
            metrics["editorial_quality_status"] = "failed"
        if metrics["deepseek_schema_valid_count"] and metrics["deepseek_key_field_missing_count"]:
            metrics["deepseek_health_hint"] = "deepseek_low_fact_quality"
        elif metrics["deepseek_schema_valid_count"]:
            metrics["deepseek_health_hint"] = "deepseek_ok"
        else:
            metrics["deepseek_health_hint"] = "deepseek_not_observed"
        if metrics["gpt_schema_valid_count"] and metrics["gpt_key_field_missing_count"]:
            metrics["gpt_health_hint"] = "gpt_low_fact_quality"
        elif metrics["gpt_schema_valid_count"]:
            metrics["gpt_health_hint"] = "gpt_ok"
        else:
            metrics["gpt_health_hint"] = "gpt_not_observed"
        if metrics["llm_schema_valid_count"] and metrics["llm_key_field_missing_count"]:
            metrics["llm_health_hint"] = "llm_low_fact_quality"
        elif metrics["llm_schema_valid_count"]:
            metrics["llm_health_hint"] = "llm_ok"
        else:
            metrics["llm_health_hint"] = "llm_not_observed"
        return metrics


def enrich_editorial_fields(
    item: Dict[str, Any],
    *,
    physical_ai_min_items: int = 4,
    paper_technical_intro_min_count: int = 12,
) -> Dict[str, Any]:
    return EditorialEngine(
        physical_ai_min_items=physical_ai_min_items,
        paper_technical_intro_min_count=paper_technical_intro_min_count,
    ).decorate_item(item)


def build_editorial_quality_metrics(
    items: Iterable[Dict[str, Any]],
    *,
    physical_ai_min_items: int = 4,
    paper_technical_intro_min_count: int = 12,
) -> Dict[str, Any]:
    return EditorialEngine(
        physical_ai_min_items=physical_ai_min_items,
        paper_technical_intro_min_count=paper_technical_intro_min_count,
    ).quality_metrics(items)
