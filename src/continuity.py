from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Tuple


TOPIC_LABELS = {
    "world_model_training": "World Model 训练",
    "physical_ai": "Physical AI",
    "vla": "VLA",
    "agent": "Agent",
    "inference_infra": "推理基础设施",
    "products_business": "产品与商业",
}

TOPIC_DOMAINS = {
    "world_model_training": "world_model",
    "physical_ai": "physical_ai",
    "vla": "physical_ai",
    "agent": "agent_models",
    "inference_infra": "infra_open_source",
    "products_business": "products_business",
}

FACT_FIELDS = (
    "who",
    "action",
    "target",
    "method",
    "dataset_or_benchmark",
    "metric_result",
    "baseline",
    "limitation",
    "code_or_project",
    "deployment_context",
)

NOVELTY_FIELDS = (
    "action",
    "target",
    "method",
    "dataset_or_benchmark",
    "metric_result",
    "baseline",
    "limitation",
    "code_or_project",
    "deployment_context",
)

TOPIC_PATTERNS = (
    (
        "world_model_training",
        re.compile(
            r"world models?|世界模型|latent dynamics|dynamics model|video prediction|future prediction|"
            r"predictive model|jepa|rollout|trajectory prediction|策略预测|视频预测|潜空间动力学",
            re.IGNORECASE,
        ),
    ),
    (
        "vla",
        re.compile(r"\bvla\b|vision[- ]language[- ]action|视觉语言动作|具身大模型", re.IGNORECASE),
    ),
    (
        "physical_ai",
        re.compile(
            r"physical ai|embodied|robot(?:ics)?|humanoid|manipulation|locomotion|具身|机器人|人形|机械臂",
            re.IGNORECASE,
        ),
    ),
    (
        "agent",
        re.compile(r"\bagents?\b|agentic|tool use|workflow|computer use|智能体|工具调用|工作流", re.IGNORECASE),
    ),
    (
        "inference_infra",
        re.compile(
            r"inference|serving|quantization|gpu|accelerator|datacenter|open source|license|推理|量化|算力|芯片|部署|开源",
            re.IGNORECASE,
        ),
    ),
)


def _clean(value: Any, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" -:：;；,，。")
    if len(text) <= limit:
        return text
    clipped = text[:limit]
    boundary = max(clipped.rfind(mark) for mark in ("。", "；", ";", "，", ","))
    if boundary >= max(32, int(limit * 0.55)):
        return clipped[: boundary + 1]
    return clipped.rstrip(" -:：;；,，。")


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").lower())


def _facts(item: Dict[str, Any]) -> Dict[str, Any]:
    value = item.get("facts_cn") or item.get("facts") or {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = {}
    return value if isinstance(value, dict) else {}


def _fact_text(item: Dict[str, Any]) -> str:
    facts = _facts(item)
    evidence = facts.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    return " ".join(
        [str(item.get("title_cn") or item.get("title") or "")]
        + [str(facts.get(key) or "") for key in FACT_FIELDS]
        + [str(value or "") for value in evidence[:3]]
        + [str(item.get("topic") or ""), str(item.get("category") or "")]
    )


def infer_topic_key(item: Dict[str, Any]) -> str:
    existing = str(item.get("continuity_topic_key") or "").strip()
    if existing:
        return existing
    text = _fact_text(item)
    for topic_key, pattern in TOPIC_PATTERNS:
        if pattern.search(text):
            return topic_key
    domain_key = str(item.get("domain_key") or "").strip()
    return {
        "world_model": "world_model_training",
        "physical_ai": "physical_ai",
        "agent_models": "agent",
        "infra_open_source": "inference_infra",
        "products_business": "products_business",
    }.get(domain_key, "products_business")


def entity_key(item: Dict[str, Any]) -> str:
    facts = _facts(item)
    raw = facts.get("who") or facts.get("code_or_project") or item.get("source_detail") or ""
    value = _normalize(raw)
    if value in {"", "thispaper", "researchers", "该论文", "研究团队", "研究者"}:
        value = _normalize(item.get("title_cn") or item.get("title") or "")[:48]
    return value[:80]


def event_fingerprint(item: Dict[str, Any]) -> str:
    facts = _facts(item)
    payload = "|".join(
        [infer_topic_key(item), entity_key(item)]
        + [_normalize(facts.get(key)) for key in ("action", "target", "method", "metric_result", "code_or_project")]
    )
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:20]


def confidence_for_item(item: Dict[str, Any]) -> Tuple[str, str]:
    facts = _facts(item)
    text = _fact_text(item)
    content_type = str(item.get("content_type") or "").lower()
    evidence = facts.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    source_tier = str(item.get("source_tier") or "").lower()
    source_text = " ".join(
        str(item.get(key) or "") for key in ("source", "source_detail", "platform", "canonical_url", "url")
    ).lower()

    if content_type == "paper" and (
        facts.get("metric_result")
        or facts.get("dataset_or_benchmark")
        or re.search(r"\d|experiment|benchmark|实验|指标|基准", " ".join(map(str, evidence)), re.IGNORECASE)
    ):
        return "已有实验验证", "论文给出了实验、基准或量化结果，但结论仍受任务设置和数据范围约束。"
    if re.search(r"interview|podcast|opinion|essay|观点|访谈|采访|预测|认为", text, re.IGNORECASE):
        return "观点或推测", "这是观点材料，用来理解判断框架，不等同于已验证事实。"
    if "google news" in source_text or "aggregator" in source_tier or source_tier in {"tier_3", "low"}:
        return "媒体报道，尚待原始材料", "当前证据来自媒体或聚合入口，关键数字和结论需回到原始材料核对。"
    if any(token in source_tier for token in ("official", "tier_1")) or any(
        host in source_text for host in ("openai.com", "anthropic.com", "deepmind.google", "microsoft.com", "nvidia.com")
    ):
        return "只有官方声明", "主体已公开发布相关信息，但尚缺少独立测试或外部采用数据。"
    return "公开材料可确认", "当前结论由公开材料支撑，证据强度取决于原文是否披露方法、数字或可复现资源。"


def _item_similarity(current: Dict[str, Any], previous: Dict[str, Any]) -> float:
    if str(current.get("canonical_url") or current.get("url") or "") == str(
        previous.get("canonical_url") or previous.get("url") or ""
    ):
        return 1.0
    if infer_topic_key(current) != infer_topic_key(previous):
        return 0.0
    score = 0.28
    current_entity = entity_key(current)
    previous_entity = entity_key(previous)
    if current_entity and previous_entity and current_entity == previous_entity:
        score += 0.42
    current_title = _normalize(current.get("title_cn") or current.get("title"))
    previous_title = _normalize(previous.get("title_cn") or previous.get("title"))
    if current_title and previous_title:
        score += 0.3 * SequenceMatcher(None, current_title, previous_title).ratio()
    return min(1.0, score)


def _new_fact_pairs(current: Dict[str, Any], previous: Dict[str, Any]) -> List[Tuple[str, str]]:
    current_facts = _facts(current)
    previous_facts = _facts(previous)
    result: List[Tuple[str, str]] = []
    for key in NOVELTY_FIELDS:
        value = _clean(current_facts.get(key), 150)
        if not value:
            continue
        old_value = _normalize(previous_facts.get(key))
        new_value = _normalize(value)
        if not old_value or (new_value not in old_value and old_value not in new_value):
            result.append((key, value))
    return result


def _delta_text(item: Dict[str, Any], previous: Optional[Dict[str, Any]], novel: List[Tuple[str, str]]) -> str:
    facts = _facts(item)
    if previous is None:
        subject = _clean(facts.get("who") or item.get("editorial_title") or item.get("title_cn") or item.get("title"), 60)
        action = _clean(facts.get("action"), 42)
        target = _clean(facts.get("target") or facts.get("method"), 90)
        statement = "".join(part for part in (subject, action, target) if part)
        return _clean(f"首次进入日报跟踪：{statement}" if statement else "首次进入日报跟踪。", 170)
    if not novel:
        return "与前次记录相比，没有出现可验证的新事实。"
    labels = {
        "action": "动作",
        "target": "对象",
        "method": "方法",
        "dataset_or_benchmark": "评测",
        "metric_result": "结果",
        "baseline": "对照",
        "limitation": "局限",
        "code_or_project": "资源",
        "deployment_context": "部署",
    }
    changes = [f"{labels.get(key, key)}新增“{value}”" for key, value in novel[:2]]
    return _clean("相较前次，" + "；".join(changes) + "。", 190)


def _paper_context(item: Dict[str, Any], related: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    facts = _facts(item)
    method = _clean(facts.get("method"), 100)
    baseline = _clean(facts.get("baseline"), 90)
    metric = _clean(facts.get("metric_result") or facts.get("dataset_or_benchmark"), 100)
    limitation = _clean(facts.get("limitation"), 90)
    route = _clean(
        f"它位于“{TOPIC_LABELS.get(infer_topic_key(item), infer_topic_key(item))}”路线中，核心机制是{method}。"
        if method
        else f"它属于“{TOPIC_LABELS.get(infer_topic_key(item), infer_topic_key(item))}”路线。",
        150,
    )
    prior = _clean(f"论文把 {baseline} 作为主要对照。" if baseline else "原文未明确给出足以定位技术谱系的主要 baseline。", 120)
    difference = _clean(
        f"相较于 {baseline}，它把 {method} 加入方法链路。" if baseline and method else "现有事实还不足以精确判断它与前序工作的机制差异。",
        150,
    )
    judgment = _clean(
        f"{metric}支持这条方法在论文设定内有效；{limitation or '能否跨数据集和真实场景成立仍需继续验证'}。"
        if metric
        else "当前材料没有充分量化结果，暂时只能理解方法，不能据此确认效果优势。",
        170,
    )
    related_reading = {}
    if related:
        related_reading = {
            "title": str(related.get("editorial_title") or related.get("title_cn") or related.get("title") or ""),
            "url": str(related.get("canonical_url") or related.get("url") or ""),
        }
    return {
        "technical_lineage": route,
        "prior_work_context": prior,
        "difference_from_prior": difference,
        "judgment_effect": judgment,
        "related_reading": related_reading,
    }


def enrich_continuity(
    items: Iterable[Dict[str, Any]],
    previous_items: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    previous = [dict(item) for item in previous_items]
    result: List[Dict[str, Any]] = []
    for raw_item in items:
        item = dict(raw_item)
        topic_key = infer_topic_key(item)
        candidates = sorted(
            ((candidate, _item_similarity(item, candidate)) for candidate in previous),
            key=lambda pair: pair[1],
            reverse=True,
        )
        best, similarity = candidates[0] if candidates else (None, 0.0)
        matched = best if similarity >= 0.66 else None
        novel = _new_fact_pairs(item, matched) if matched else []
        if matched is None:
            status = "new"
        elif novel:
            status = "updated"
        else:
            status = "repeated"
        confidence_label, confidence_reason = confidence_for_item(item)
        item.update(
            {
                "continuity_topic_key": topic_key,
                "continuity_topic_label": TOPIC_LABELS.get(topic_key, topic_key),
                "continuity_entity_key": entity_key(item),
                "event_fingerprint": event_fingerprint(item),
                "continuity_status": status,
                "delta_summary": _delta_text(item, matched, novel),
                "previous_conclusion": _clean((matched or {}).get("editorial_lead"), 150),
                "confidence_label": confidence_label,
                "confidence_reason": confidence_reason,
                "new_fact_fields": [key for key, _ in novel],
            }
        )
        if item.get("content_type") == "paper":
            related = next(
                (
                    candidate
                    for candidate in previous
                    if candidate.get("content_type") == "paper"
                    and infer_topic_key(candidate) == topic_key
                    and str(candidate.get("canonical_url") or candidate.get("url") or "")
                    != str(item.get("canonical_url") or item.get("url") or "")
                ),
                None,
            )
            item.update(_paper_context(item, related))
        result.append(item)
    return result


def build_topic_dossiers(items: Iterable[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[infer_topic_key(item)].append(item)
    dossiers: List[Dict[str, Any]] = []
    priority = ["world_model_training", "physical_ai", "vla", "agent", "inference_infra"]
    for topic_key in priority[: max(1, int(limit))]:
        topic_items = grouped.get(topic_key, [])
        if not topic_items:
            continue
        routes: List[str] = []
        changes: List[str] = []
        open_questions: List[str] = []
        reading: List[Dict[str, str]] = []
        for item in topic_items:
            facts = _facts(item)
            method = _clean(facts.get("method"), 110)
            if method and method not in routes:
                routes.append(method)
            delta = _clean(item.get("delta_summary"), 150)
            if delta and item.get("continuity_status") != "repeated" and delta not in changes:
                changes.append(delta)
            limitation = _clean(facts.get("limitation"), 120)
            if limitation and limitation not in open_questions:
                open_questions.append(limitation)
            title = str(item.get("editorial_title") or item.get("title_cn") or item.get("title") or "")
            url = str(item.get("canonical_url") or item.get("url") or "")
            if title and url and not any(entry["url"] == url for entry in reading):
                reading.append({"title": title, "url": url})
        if not open_questions:
            open_questions.append("这条路线能否在更广的数据、任务或真实部署条件下保持结果？")
        representatives = [
            {
                "title": str(item.get("editorial_title") or item.get("title_cn") or item.get("title") or ""),
                "url": str(item.get("canonical_url") or item.get("url") or ""),
                "confidence": str(item.get("confidence_label") or ""),
            }
            for item in topic_items[:3]
        ]
        dossiers.append(
            {
                "topic_key": topic_key,
                "label": TOPIC_LABELS.get(topic_key, topic_key),
                "domain_key": TOPIC_DOMAINS.get(topic_key, ""),
                "current_routes": routes[:3],
                "representative_items": representatives,
                "recent_changes": changes[:3],
                "open_questions": open_questions[:2],
                "recommended_reading": reading[:4],
            }
        )
    return dossiers


def build_closing_memory(items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    visible = [item for item in items if item.get("continuity_status") != "repeated"]
    facts: List[str] = []
    mechanism = ""
    question = ""
    for item in visible:
        item_facts = _facts(item)
        delta = _clean(item.get("delta_summary"), 95)
        if delta and delta not in facts:
            facts.append(delta)
        if not mechanism and item_facts.get("method"):
            mechanism = _clean(f"机制：{item_facts.get('method')}", 130)
        if not question and item_facts.get("limitation"):
            question = _clean(f"未决问题：{item_facts.get('limitation')}", 130)
        if len(facts) >= 3 and mechanism and question:
            break
    if not mechanism:
        mechanism = "机制：本期尚无同时具备明确方法和结果证据的新机制。"
    if not question:
        question = "未决问题：哪些结果能跨越单一基准，进入真实任务或稳定部署？"
    return {"facts": facts[:3], "mechanism": mechanism, "open_question": question}


def reading_queue_context(
    queue_rows: Iterable[Dict[str, Any]],
    current_items: Iterable[Dict[str, Any]],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    current = list(current_items)
    result: List[Dict[str, Any]] = []
    for row in queue_rows:
        snapshot = row.get("snapshot") if isinstance(row.get("snapshot"), dict) else {}
        topic_key = str(row.get("topic_key") or infer_topic_key(snapshot))
        changes = [
            item
            for item in current
            if infer_topic_key(item) == topic_key and item.get("continuity_status") in {"new", "updated"}
        ]
        result.append(
            {
                "article_id": row.get("article_id"),
                "report_id": row.get("report_id"),
                "topic_key": topic_key,
                "topic_label": TOPIC_LABELS.get(topic_key, topic_key),
                "title": str(snapshot.get("editorial_title") or snapshot.get("title_cn") or snapshot.get("title") or ""),
                "url": str(snapshot.get("canonical_url") or snapshot.get("url") or ""),
                "new_changes": [str(item.get("delta_summary") or "") for item in changes[:2]],
                "has_new_evidence": bool(changes),
                "status": row.get("status", "tracked"),
            }
        )
        if len(result) >= max(1, int(limit)):
            break
    return result


def build_weekly_digest(items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    candidates = list(items)
    confirmed = [
        _clean(item.get("delta_summary") or item.get("editorial_lead"), 120)
        for item in candidates
        if item.get("confidence_label") == "已有实验验证" and item.get("continuity_status") != "repeated"
    ]
    emerging = [
        _clean(item.get("delta_summary"), 120)
        for item in candidates
        if item.get("continuity_status") == "updated"
    ]
    papers = [
        {
            "title": str(item.get("editorial_title") or item.get("title_cn") or item.get("title") or ""),
            "url": str(item.get("canonical_url") or item.get("url") or ""),
            "route": str(item.get("technical_lineage") or ""),
        }
        for item in candidates
        if item.get("content_type") == "paper"
    ][:5]
    cooled = [
        _clean(item.get("previous_conclusion") or item.get("editorial_lead"), 110)
        for item in candidates
        if item.get("continuity_status") == "repeated"
    ]
    questions = []
    for item in candidates:
        limitation = _clean(_facts(item).get("limitation"), 120)
        if limitation and limitation not in questions:
            questions.append(limitation)
    return {
        "confirmed": [value for value in confirmed if value][:4],
        "cooled_or_unconfirmed": [value for value in cooled if value][:3],
        "emerging_routes": [value for value in emerging if value][:4],
        "best_papers": papers,
        "next_week_watchlist": questions[:4],
    }

