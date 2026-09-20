from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qsl, urlencode, urlparse, urlsplit, urlunsplit

from .base import BaseCollector
from ..editorial_engine import (
    attribution_opener_pattern,
    contains_mojibake,
    has_bad_public_phrase,
    has_untranslated_prose,
    mixed_language_title,
    paper_plain_summary_passes,
    paper_technical_intro_passes,
)
from ..relevance import clean_snippet, infer_platform, infer_source_tier, is_ai_web_content

CODEX_RESEARCH_ANALYSIS_VERSION = "codex-research-v3"
CODEX_RESEARCH_SCHEMA_VERSION = "codex-research-v3"
SUPPORTED_CODEX_RESEARCH_ANALYSIS_VERSIONS = frozenset({
    "codex-research-v2",
    CODEX_RESEARCH_ANALYSIS_VERSION,
})


class CodexResearchInboxCollector(BaseCollector):
    """Import source-linked research produced by a scheduled Codex task."""

    ITEM_REJECTION_DIAGNOSTICS = (
        "content_schema_error_count",
        "bad_editorial_title_count",
        "bad_editorial_summary_count",
        "truncated_editorial_copy_count",
        "ai_relevance_error_count",
        "missing_source_evidence_count",
        "short_source_excerpt_count",
        "structured_fact_missing_count",
        "technical_contract_missing_count",
        "paper_contract_missing_count",
        "numeric_context_missing_count",
        "key_number_contract_missing_count",
        "key_number_evidence_missing_count",
        "key_number_public_copy_missing_count",
        "unsupported_summary_numeric_count",
        "claim_type_error_count",
        "claim_language_mismatch_count",
        "interview_locator_missing_count",
        "interview_evidence_substance_missing_count",
        "missing_publish_date_count",
        "future_publish_date_count",
        "stale_source_count",
        "supplemental_too_old_count",
        "technical_category_error_count",
        "short_summary_count",
        "long_summary_count",
        "non_chinese_summary_count",
        "insufficient_sentence_count",
        "paragraph_structure_missing_count",
        "opening_substance_missing_count",
        "repetitive_summary_count",
    )

    def __init__(
        self,
        inbox_path: str,
        *,
        max_age_minutes: int = 240,
        minimum_items: int = 20,
        minimum_papers: int = 15,
        minimum_news: int = 20,
        minimum_technical: int = 20,
        technical_primary_source_ratio_min: float = 0.0,
        minimum_discovery_candidates: int = 0,
        minimum_discovered_by_section: Dict[str, int] | None = None,
        minimum_submitted_by_section: Dict[str, int] | None = None,
        minimum_key_number_items_by_section: Dict[str, int] | None = None,
        required_schema_version: str = "",
        technical_category_quotas: Dict[str, int] | None = None,
        technical_category_minimums: Dict[str, int] | None = None,
        news_format_quotas: Dict[str, int] | None = None,
        paper_domain_quotas: Dict[str, Dict[str, int]] | None = None,
        minimum_fresh_by_section: Dict[str, int] | None = None,
        supplemental_max_age_hours_by_section: Dict[str, int] | None = None,
        max_attribution_opener_count: int = 8,
        max_cross_item_template_repeat_count: int = 10000,
        cross_item_template_similarity_threshold: float = 0.92,
    ):
        self.inbox_path = Path(inbox_path)
        self.max_age_minutes = max(1, int(max_age_minutes))
        self.minimum_items = max(1, int(minimum_items))
        self.minimum_papers = max(0, int(minimum_papers))
        self.minimum_news = max(0, int(minimum_news))
        self.minimum_technical = max(0, int(minimum_technical))
        self.technical_primary_source_ratio_min = max(
            0.0,
            min(1.0, float(technical_primary_source_ratio_min or 0.0)),
        )
        self.minimum_discovery_candidates = max(0, int(minimum_discovery_candidates or 0))
        self.minimum_discovered_by_section = {
            str(section): max(0, int(minimum or 0))
            for section, minimum in dict(minimum_discovered_by_section or {}).items()
            if str(section) in {"news", "technical", "paper"}
        }
        self.minimum_submitted_by_section = {
            str(section): max(0, int(minimum or 0))
            for section, minimum in dict(minimum_submitted_by_section or {}).items()
            if str(section) in {"news", "technical", "paper"}
        }
        self.minimum_key_number_items_by_section = {
            str(section): max(0, int(minimum or 0))
            for section, minimum in dict(minimum_key_number_items_by_section or {}).items()
            if str(section) in {"news", "technical", "paper"}
        }
        self.required_schema_version = str(required_schema_version or "").strip()
        self.technical_category_quotas = dict(technical_category_quotas) if technical_category_quotas is not None else {
            "embodied_world_model": 6,
            "agent_systems": 4,
            "training_data": 4,
            "inference_deployment": 3,
            "multimodal_architecture": 3,
        }
        self.technical_category_minimums = (
            dict(technical_category_minimums)
            if technical_category_minimums is not None
            else dict(self.technical_category_quotas)
        )
        self.news_format_quotas = dict(news_format_quotas or {})
        self.paper_domain_quotas = dict(paper_domain_quotas or {})
        self.minimum_fresh_by_section = {
            str(section): max(0, int(minimum or 0))
            for section, minimum in dict(minimum_fresh_by_section or {}).items()
        }
        configured_supplemental_ages = dict(
            supplemental_max_age_hours_by_section
            or {"news": 168, "technical": 720, "paper": 720}
        )
        self.supplemental_max_age_hours_by_section = {
            section: max(1, int(configured_supplemental_ages.get(section, 0) or 0))
            for section in ("news", "technical", "paper")
        }
        self.label = "CodexResearchInboxCollector"
        self.run_in_subprocess = False
        self.require_key_numbers = False
        self.max_attribution_opener_count = max(
            1,
            int(max_attribution_opener_count),
        )
        self.max_cross_item_template_repeat_count = max(
            0,
            int(max_cross_item_template_repeat_count),
        )
        self.cross_item_template_similarity_threshold = max(
            0.75,
            min(1.0, float(cross_item_template_similarity_threshold)),
        )
        self.fetch_diagnostics: Dict[str, Any] = {
            "provider": "codex_automation",
            "inbox_path": self.inbox_path.as_posix(),
            "file_exists": self.inbox_path.exists(),
            "inbox_sha256": "",
            "fresh": False,
            "generated_at": "",
            "age_minutes": None,
            "schema_error_count": 0,
            "discovery_candidate_count": 0,
            "discovery_manifest_sha256": "",
            "discovery_section_counts": {},
            "discovery_duplicate_url_count": 0,
            "discovery_invalid_row_count": 0,
            "submitted_not_in_discovery_count": 0,
            "submitted_not_in_discovery_examples": [],
            "discovery_underfilled": [],
            "discovery_quota_status": "not_loaded",
            "submitted_item_count": 0,
            "submitted_section_counts": {},
            "submission_underfilled": [],
            "submission_quota_status": "not_loaded",
            "rejected_item_count": 0,
            "rejected_section_counts": {},
            "rejection_reason_counts_by_section": {},
            "accepted_section_rates": {},
            "paper_count": 0,
            "news_count": 0,
            "technical_count": 0,
            "technical_primary_source_count": 0,
            "technical_primary_source_ratio": 0.0,
            "technical_primary_source_status": "not_loaded",
            "duplicate_url_count": 0,
            "duplicate_url_examples": [],
            "duplicate_event_count": 0,
            "duplicate_event_examples": [],
            "content_schema_error_count": 0,
            "bad_editorial_title_count": 0,
            "bad_editorial_summary_count": 0,
            "truncated_editorial_copy_count": 0,
            "truncated_editorial_copy_examples": [],
            "ai_relevance_error_count": 0,
            "short_summary_count": 0,
            "long_summary_count": 0,
            "non_chinese_summary_count": 0,
            "insufficient_sentence_count": 0,
            "paragraph_structure_missing_count": 0,
            "opening_substance_missing_count": 0,
            "repetitive_summary_count": 0,
            "attribution_opener_counts": {},
            "attribution_opener_overuse_count": 0,
            "attribution_opener_overuse_examples": [],
            "cross_item_template_repeat_count": 0,
            "cross_item_template_repeat_examples": [],
            "stale_source_count": 0,
            "supplemental_too_old_count": 0,
            "missing_publish_date_count": 0,
            "future_publish_date_count": 0,
            "missing_source_evidence_count": 0,
            "short_source_excerpt_count": 0,
            "structured_fact_missing_count": 0,
            "technical_contract_missing_count": 0,
            "technical_contract_missing_examples": [],
            "paper_contract_missing_count": 0,
            "paper_contract_missing_examples": [],
            "numeric_context_missing_count": 0,
            "schema_version": "",
            "required_schema_version": self.required_schema_version,
            "schema_version_status": "not_loaded",
            "key_number_contract_missing_count": 0,
            "key_number_contract_missing_examples": [],
            "key_number_evidence_missing_count": 0,
            "key_number_evidence_missing_examples": [],
            "key_number_public_copy_missing_count": 0,
            "key_number_public_copy_missing_examples": [],
            "key_number_item_counts": {},
            "key_number_underfilled": [],
            "key_number_quota_status": "not_loaded",
            "unsupported_summary_numeric_count": 0,
            "unsupported_summary_numeric_examples": [],
            "claim_type_error_count": 0,
            "claim_language_mismatch_count": 0,
            "claim_language_mismatch_examples": [],
            "interview_locator_missing_count": 0,
            "interview_evidence_substance_missing_count": 0,
            "technical_category_error_count": 0,
            "technical_quota_status": "not_loaded",
            "technical_category_counts": {},
            "technical_category_underfilled": [],
            "technical_category_target_underfilled": [],
            "technical_target_status": "not_loaded",
            "news_format_counts": {},
            "news_format_underfilled": [],
            "news_format_quota_status": "not_loaded",
            "paper_domain_counts": {},
            "paper_domain_underfilled": [],
            "paper_domain_exceeded": [],
            "paper_domain_quota_status": "not_loaded",
            "fresh_source_counts": {},
            "supplemental_older_counts": {},
            "freshness_underfilled": [],
            "freshness_quota_status": "not_loaded",
            "quality_status": "not_loaded",
        }

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.astimezone()
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _chinese_ratio(value: Any) -> float:
        text = str(value or "")
        chinese_count = len(re.findall(r"[\u3400-\u9fff]", text))
        language_count = chinese_count + len(re.findall(r"[A-Za-z]", text))
        return chinese_count / language_count if language_count else 0.0

    @staticmethod
    def _summary_sentences(value: Any) -> List[str]:
        return [
            re.sub(r"\s+", "", sentence).strip("，,：:、 ")
            for sentence in re.split(r"[。！？!?；;]+", str(value or ""))
            if re.sub(r"\s+", "", sentence).strip("，,：:、 ")
        ]

    @staticmethod
    def _summary_template_signature(item: Dict[str, Any]) -> str:
        summary = str(
            (item.get("_codex_research_analysis") or {}).get("summary")
            or item.get("summary")
            or ""
        )
        facts = dict(item.get("facts") or {})
        replacements = []
        for key in (
            "who",
            "target",
            "method",
            "architecture",
            "training_objective",
            "input_output",
            "dataset_or_benchmark",
            "metric_result",
            "baseline",
            "limitation",
            "code_or_project",
            "deployment_context",
        ):
            value = str(facts.get(key) or "").strip()
            if len(value) >= 3:
                replacements.append(value)
        who = str(facts.get("who") or "").strip()
        for suffix in ("研究实验室", "研究室", "实验室", "研究院", "团队", "公司"):
            if who.endswith(suffix) and len(who) > len(suffix) + 1:
                replacements.append(who[: -len(suffix)])
        for value in sorted(set(replacements), key=len, reverse=True):
            summary = re.sub(re.escape(value), "<事实>", summary, flags=re.IGNORECASE)
        summary = re.sub(r"\d+(?:\.\d+)?%?", "<数值>", summary)
        return re.sub(r"[\W_]+", "", summary).lower()

    def _cross_item_template_repeats(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        seen_by_section: Dict[str, List[tuple[str, Dict[str, Any]]]] = {}
        repeated = []
        for item in items:
            section = str(item.get("primary_section") or "unknown")
            signature = self._summary_template_signature(item)
            if len(signature) < 80:
                continue
            best_ratio = 0.0
            best_item = None
            for previous_signature, previous_item in seen_by_section.setdefault(section, []):
                ratio = SequenceMatcher(None, signature, previous_signature).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_item = previous_item
            if best_item is not None and best_ratio >= self.cross_item_template_similarity_threshold:
                repeated.append(
                    {
                        "primary_section": section,
                        "similarity": round(best_ratio, 3),
                        "title": str(item.get("title_cn") or item.get("title") or ""),
                        "matches": str(
                            best_item.get("title_cn") or best_item.get("title") or ""
                        ),
                    }
                )
            seen_by_section[section].append((signature, item))
        return repeated

    @staticmethod
    def _canonical_number(value: Any) -> str:
        text = re.sub(r"[\s,]", "", str(value or "")).strip()
        if not text:
            return ""
        if re.fullmatch(r"\d+(?:\.\d+)?", text):
            try:
                normalized = format(Decimal(text).normalize(), "f")
            except InvalidOperation:
                return ""
            return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized

        digits = {
            "零": 0,
            "〇": 0,
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
        }
        if not re.fullmatch(r"[零〇一二两三四五六七八九十百千万亿点]+", text):
            return ""

        integer_text, separator, decimal_text = text.partition("点")
        if separator and (not integer_text or not decimal_text or "点" in decimal_text):
            return ""

        if not re.search(r"[十百千万亿]", integer_text):
            integer_digits = "".join(str(digits[char]) for char in integer_text)
            integer_value = int(integer_digits or "0")
        else:
            small_units = {"十": 10, "百": 100, "千": 1000}
            large_units = {"万": 10000, "亿": 100000000}
            total = 0
            section = 0
            number = 0
            for char in integer_text:
                if char in digits:
                    number = digits[char]
                elif char in small_units:
                    section += (number or 1) * small_units[char]
                    number = 0
                elif char in large_units:
                    section += number
                    total += (section or 1) * large_units[char]
                    section = 0
                    number = 0
            integer_value = total + section + number

        if not separator:
            return str(integer_value)
        if any(char not in digits for char in decimal_text):
            return ""
        decimal_digits = "".join(str(digits[char]) for char in decimal_text).rstrip("0")
        return f"{integer_value}.{decimal_digits}" if decimal_digits else str(integer_value)

    @classmethod
    def _metric_tokens(cls, value: Any) -> set[str]:
        text = str(value or "")
        number_pattern = r"(?:\d[\d,]*(?:\.\d+)?|[零〇一二两三四五六七八九十百千万亿点]+)"
        unit_pattern = (
            r"(?:个百分点|tokens?\s*/?\s*s|tok\s*/?\s*s|亿美元|万美元|亿元|万元|"
            r"毫秒|小时|分钟|美元|元|%|％|倍|x|ms|秒|gb|mb|项)"
        )
        unit_names = {
            "%": "percent",
            "％": "percent",
            "个百分点": "percentage_point",
            "倍": "multiple",
            "x": "multiple",
            "ms": "millisecond",
            "毫秒": "millisecond",
            "秒": "second",
            "小时": "hour",
            "分钟": "minute",
            "tokens": "token_per_second",
            "token": "token_per_second",
            "toks": "token_per_second",
            "tok": "token_per_second",
            "gb": "gb",
            "mb": "mb",
            "项": "item_count",
        }
        currency_units = {
            "亿美元": ("usd", Decimal("100000000")),
            "万美元": ("usd", Decimal("10000")),
            "美元": ("usd", Decimal("1")),
            "亿元": ("cny", Decimal("100000000")),
            "万元": ("cny", Decimal("10000")),
            "元": ("cny", Decimal("1")),
        }
        tokens: set[str] = set()

        percent_pattern = re.compile(rf"百分之\s*(?P<number>{number_pattern})", re.IGNORECASE)
        for match in percent_pattern.finditer(text):
            number = cls._canonical_number(match.group("number"))
            if number:
                tokens.add(f"percent:{number}")

        metric_pattern = re.compile(
            rf"(?P<number>{number_pattern})\s*(?P<unit>{unit_pattern})",
            re.IGNORECASE,
        )
        for match in metric_pattern.finditer(text):
            raw_number = match.group("number")
            number = cls._canonical_number(raw_number)
            unit = re.sub(r"\s|/", "", match.group("unit")).lower()
            if unit == "项" and raw_number == "一":
                continue
            if unit in currency_units and number:
                unit_name, scale = currency_units[unit]
                number = cls._canonical_number(str(Decimal(number) * scale))
            else:
                unit_name = "token_per_second" if unit.startswith("tok") else unit_names.get(unit)
            if number and unit_name:
                tokens.add(f"{unit_name}:{number}")

        count_units = {
            "家公司": "company_count",
            "公司": "company_count",
            "种模态": "modality_count",
            "个陨石坑": "crater_count",
            "组数据": "group_count",
            "个样本": "sample_count",
            "个任务": "task_count",
            "个客户": "customer_count",
            "篇论文": "document_count",
            "台服务器": "machine_count",
            "名参与者": "person_count",
            "模态": "modality_count",
            "陨石坑": "crater_count",
            "样本": "sample_count",
            "任务": "task_count",
            "客户": "customer_count",
        }
        count_pattern = re.compile(
            r"(?P<number>\d[\d,]*(?:\.\d+)?)\s*(?P<scale>万|亿)?\s*"
            rf"(?P<unit>家公司|种模态|个陨石坑|组数据|个样本|个任务|个客户|"
            rf"篇论文|台服务器|名参与者|公司|模态|陨石坑|样本|任务|客户)",
            re.IGNORECASE,
        )
        count_scales = {"": Decimal("1"), "万": Decimal("10000"), "亿": Decimal("100000000")}
        for match in count_pattern.finditer(text):
            number = cls._canonical_number(match.group("number"))
            unit_name = count_units.get(match.group("unit"))
            if number and unit_name:
                scaled = Decimal(number) * count_scales[match.group("scale") or ""]
                tokens.add(f"{unit_name}:{cls._canonical_number(str(scaled))}")

        english_currency_pattern = re.compile(
            r"\$\s*(?P<number>\d[\d,]*(?:\.\d+)?)\s*"
            r"(?P<scale>k|m|b|million|billion)?\b",
            re.IGNORECASE,
        )
        english_scales = {
            "": Decimal("1"),
            "k": Decimal("1000"),
            "m": Decimal("1000000"),
            "b": Decimal("1000000000"),
            "million": Decimal("1000000"),
            "billion": Decimal("1000000000"),
        }
        for match in english_currency_pattern.finditer(text):
            number = cls._canonical_number(match.group("number"))
            if number:
                scaled = Decimal(number) * english_scales[(match.group("scale") or "").lower()]
                tokens.add(f"usd:{cls._canonical_number(str(scaled))}")

        english_count_units = {
            "companies": "company_count",
            "company": "company_count",
            "samples": "sample_count",
            "sample": "sample_count",
            "tasks": "task_count",
            "task": "task_count",
            "modalities": "modality_count",
            "modality": "modality_count",
            "customers": "customer_count",
            "customer": "customer_count",
            "craters": "crater_count",
            "crater": "crater_count",
            "bundles": "group_count",
            "bundle": "group_count",
        }
        english_count_pattern = re.compile(
            r"(?P<number>\d[\d,]*(?:\.\d+)?)\s*"
            r"(?P<scale>k|m|b|million|billion)?\s*"
            r"(?P<unit>companies|company|samples|sample|tasks|task|modalities|modality|"
            r"customers|customer|craters|crater|bundles|bundle)\b",
            re.IGNORECASE,
        )
        for match in english_count_pattern.finditer(text):
            number = cls._canonical_number(match.group("number"))
            unit_name = english_count_units.get(match.group("unit").lower())
            if number and unit_name:
                scaled = Decimal(number) * english_scales[(match.group("scale") or "").lower()]
                tokens.add(f"{unit_name}:{cls._canonical_number(str(scaled))}")
        return tokens

    @staticmethod
    def _editorial_text_digest(value: Any) -> str:
        normalized = re.sub(r"\s+", " ", str(value or "")).strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""

    @staticmethod
    def _clean_editorial_copy(value: Any) -> str:
        """Clean approved copy without collapsing its paragraph structure."""
        raw = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
        raw = re.sub(
            r"<\s*(?:br\s*/?|/?(?:p|div|li|h[1-6]))\b[^>]*>",
            "\n",
            raw,
            flags=re.IGNORECASE,
        )
        paragraphs = []
        for part in re.split(r"\n+", raw):
            cleaned = clean_snippet(part, limit=max(1, len(part) + 1))
            if cleaned:
                paragraphs.append(cleaned)
        return "\n\n".join(paragraphs)

    @staticmethod
    def _editorial_paragraph_digest(value: Any) -> str:
        paragraphs = [
            re.sub(r"\s+", " ", part).strip()
            for part in re.split(r"(?:\r\n|\r|\n)+", str(value or ""))
            if re.sub(r"\s+", " ", part).strip()
        ]
        if len(paragraphs) < 2:
            return ""
        normalized = "\n".join(paragraphs)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _valid_source_url(value: Any) -> bool:
        parsed = urlparse(str(value or "").strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    @staticmethod
    def _integer_stats(values: List[int]) -> Dict[str, Any]:
        cleaned = [max(0, int(value)) for value in values]
        if not cleaned:
            return {"count": 0, "min": 0, "max": 0, "average": 0.0}
        return {
            "count": len(cleaned),
            "min": min(cleaned),
            "max": max(cleaned),
            "average": round(sum(cleaned) / len(cleaned), 1),
        }

    @staticmethod
    def _opening_is_substantive(
        summary: str,
        facts: Dict[str, Any],
        primary_section: str,
        content_type: str,
    ) -> bool:
        first_paragraph = next(
            (part.strip() for part in str(summary or "").split("\n\n") if part.strip()),
            "",
        )
        minimum_length = 90 if content_type in {"interview", "podcast", "video"} else 48
        if len(first_paragraph) < minimum_length:
            return False
        anchor_text = " ".join(
            str(facts.get(key) or "")
            for key in ("who", "target", "method", "code_or_project")
        )
        anchor_tokens = [
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9._+-]{1,}|[\u4e00-\u9fff]{2,}", anchor_text)
            if token.lower()
            not in {
                "人工智能", "模型", "系统", "团队", "公司", "研究", "方法",
                "技术", "项目", "产品", "论文", "资料", "平台", "功能",
            }
        ]
        normalized_opening = first_paragraph.lower()
        has_anchor = any(token in normalized_opening for token in anchor_tokens)
        action_pattern = (
            r"发布|宣布|推出|公开|开源|更新|上线|披露|确认|报告|展示|"
            r"解释|复盘|主张|认为|指出|采用|通过|训练|部署|编码|解码|"
            r"检索|规划|控制|缓存|路由|量化|蒸馏|优化|构建|实现|接入|调用|替换"
        )
        if primary_section == "technical":
            action_pattern += r"|输入|输出|模块|架构|约束|推理"
        has_concrete_action = bool(re.search(action_pattern, first_paragraph, re.IGNORECASE))
        generic_opening = bool(re.search(
            r"^(?:本文|这篇文章|本期内容|这则内容|这份资料).{0,18}(?:介绍|讨论|关注|围绕)",
            first_paragraph,
        ))
        return has_anchor and has_concrete_action and not generic_opening

    @staticmethod
    def _claim_language_matches(summary: str, claim_type: str) -> bool:
        text = str(summary or "")
        if claim_type == "official_claim":
            return bool(re.search(
                r"(?:公司|团队|官方|发布方|项目方|作者|机构).{0,10}"
                r"(?:称|表示|介绍|宣称|自报|自述|披露|报告|公告)|"
                r"据.{0,12}(?:官方|公司|团队|公告|博客|技术文档|新闻稿)",
                text,
            ))
        if claim_type == "interview_opinion":
            return bool(re.search(
                r"(?:嘉宾|受访者|主持人|\b[A-Z][A-Za-z .'-]{1,30})?.{0,8}"
                r"(?:说|表示|认为|主张|解释|指出|判断|强调|反对|支持|提到)",
                text,
            ))
        if claim_type == "analysis":
            return bool(re.search(
                r"分析|推测|判断|可能|尚不能|不能据此|据此推演|作者认为|媒体认为",
                text,
            ))
        return True

    @staticmethod
    def _editorial_copy_is_complete(value: Any) -> bool:
        text = str(value or "").strip()
        if not text or "..." in text or "…" in text:
            return False
        return bool(re.search(r"[。！？.!?][）】》”’\"']?$", text))

    @staticmethod
    def _url_identity(value: Any) -> str:
        parsed = urlsplit(str(value or "").strip())
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        path = re.sub(r"/+$", "", parsed.path or "/") or "/"
        if host in {"arxiv.org", "export.arxiv.org"}:
            host = "arxiv.org"
            path = re.sub(r"v\d+$", "", path, flags=re.IGNORECASE)
        tracking_keys = {"ref", "source", "fbclid", "gclid", "mc_cid", "mc_eid"}
        query = urlencode(sorted(
            (key, item)
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in tracking_keys
        ))
        return urlunsplit((parsed.scheme.lower(), host, path, query, ""))

    @staticmethod
    def _event_identity_component(value: Any) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"(?<=\d)\.0\b", "", text)
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)

    @classmethod
    def _event_identity(cls, item: Dict[str, Any]) -> str:
        if str(item.get("primary_section") or "") == "paper":
            return ""
        facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
        who = cls._event_identity_component(facts.get("who"))
        who = re.sub(
            r"(?:研究团队|实验室|研究院|团队|公司|corporation|team|labs?|inc|corp)$",
            "",
            who,
        )
        raw_action = str(facts.get("action") or "").strip().lower()
        action_groups = (
            (r"发布|推出|上线|公开|开源|开放|release|launch|open.?source", "release"),
            (r"更新|升级|改进|新增|update|upgrade", "update"),
            (r"宣布|披露|公布|announce|reveal", "announce"),
            (r"合作|联手|携手|partner|collaborat", "partner"),
            (r"收购|并购|acquir", "acquire"),
            (r"融资|投资|fund|invest", "funding"),
        )
        action = next(
            (
                normalized
                for pattern, normalized in action_groups
                if re.search(pattern, raw_action, re.IGNORECASE)
            ),
            cls._event_identity_component(raw_action),
        )
        target_text = re.sub(
            r"^(?:全新|最新|新版|新的|新一代)+",
            "",
            str(facts.get("target") or "").strip(),
        )
        target = cls._event_identity_component(target_text)
        identity = "|".join((who, action, target))
        return identity if len(identity.replace("|", "")) >= 12 else ""

    @staticmethod
    def _duplicate_example(
        rejected: Dict[str, Any], matched: Dict[str, Any]
    ) -> Dict[str, str]:
        def title(item: Dict[str, Any]) -> str:
            analysis = (
                item.get("_codex_research_analysis")
                if isinstance(item.get("_codex_research_analysis"), dict)
                else {}
            )
            return str(
                analysis.get("title_cn")
                or item.get("title_cn")
                or item.get("title")
                or ""
            )

        return {
            "rejected_section": str(rejected.get("primary_section") or ""),
            "rejected_title": title(rejected),
            "rejected_url": str(rejected.get("canonical_url") or rejected.get("url") or ""),
            "matched_section": str(matched.get("primary_section") or ""),
            "matched_title": title(matched),
            "matched_url": str(matched.get("canonical_url") or matched.get("url") or ""),
        }

    @classmethod
    def _facts(cls, row: Dict[str, Any]) -> Dict[str, Any]:
        facts = row.get("facts") if isinstance(row.get("facts"), dict) else {}
        evidence = facts.get("evidence") or row.get("evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        facts = dict(facts)
        facts["evidence"] = [clean_snippet(str(item), limit=320) for item in evidence if str(item).strip()][:4]
        if "key_numbers" in facts:
            key_numbers = facts.get("key_numbers")
            if isinstance(key_numbers, list):
                facts["key_numbers"] = [
                    clean_snippet(str(item), limit=120)
                    for item in key_numbers
                    if str(item).strip()
                ][:3]
        for key in (
            "source_excerpt",
            "evidence_locator",
            "primary_section",
            "technical_category",
            "claim_type",
        ):
            value = row.get(key) or facts.get(key)
            if value:
                facts[key] = clean_snippet(str(value), limit=520)
        for key in ("paper_plain_summary", "paper_technical_intro"):
            value = row.get(key) or facts.get(key)
            if value:
                facts[key] = cls._clean_editorial_copy(value)
        return facts

    @staticmethod
    def _primary_section(row: Dict[str, Any], content_type: str) -> str:
        section = str(row.get("primary_section") or "").strip().lower()
        if section in {"news", "technical", "paper"}:
            return section
        if content_type == "paper":
            return "paper"
        if content_type in {"project", "open_source", "opensource"}:
            return "technical"
        return "news"

    @staticmethod
    def _paper_domain_key(item: Dict[str, Any]) -> str:
        facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
        text = " ".join(
            str(value or "").lower()
            for value in (
                item.get("topic"),
                item.get("category"),
                item.get("title"),
                item.get("title_cn"),
                facts.get("target"),
                facts.get("method"),
            )
        )
        if re.search(r"world model|世界模型|video prediction|环境预测", text):
            return "world_model"
        if re.search(r"physical ai|robot|robotics|具身|机器人|vla|manipulation|navigation", text):
            return "physical_ai"
        if re.search(r"agent|智能体|tool use|工具调用|language model|语言模型", text):
            return "agent_models"
        if re.search(r"infra|inference|deployment|serving|open source|开源|推理|部署|系统优化", text):
            return "infra_open_source"
        return "other"

    def _normalize_item(self, row: Dict[str, Any]) -> Dict[str, Any] | None:
        title = clean_snippet(str(row.get("title") or ""), limit=240)
        title_cn = clean_snippet(str(row.get("title_cn") or title), limit=240)
        url = str(row.get("canonical_url") or row.get("url") or "").strip()
        summary = self._clean_editorial_copy(row.get("summary"))
        facts = self._facts(row)
        evidence = facts.get("evidence") or []
        if not title or not self._valid_source_url(url) or not summary or not evidence:
            self.fetch_diagnostics["content_schema_error_count"] += 1
            return None
        if (
            not 6 <= len(title_cn) <= 56
            or any(marker in title_cn for marker in ("...", "…"))
            or contains_mojibake(title_cn)
            or has_bad_public_phrase(title_cn)
            or has_untranslated_prose(title_cn)
            or mixed_language_title(title_cn)
            or re.search(r"[\u4e00-\u9fff]\s+[\u4e00-\u9fff]", title_cn)
        ):
            self.fetch_diagnostics["bad_editorial_title_count"] += 1
            return None
        source_excerpt = str(facts.get("source_excerpt") or "").strip()
        evidence_locator = str(facts.get("evidence_locator") or "").strip()
        if not source_excerpt or not evidence_locator:
            self.fetch_diagnostics["missing_source_evidence_count"] += 1
            return None
        if len(source_excerpt) < 24:
            self.fetch_diagnostics["short_source_excerpt_count"] += 1
            return None
        if not is_ai_web_content(title, summary):
            self.fetch_diagnostics["ai_relevance_error_count"] += 1
            return None

        content_type = str(row.get("content_type") or "news").strip().lower()
        if content_type not in {"news", "paper", "interview", "podcast", "video", "project"}:
            content_type = "news"
        primary_section = self._primary_section(row, content_type)
        if primary_section == "paper":
            content_type = "paper"
        facts["primary_section"] = primary_section
        public_copy = [summary]
        if primary_section == "paper":
            public_copy.extend(
                [
                    str(facts.get("paper_plain_summary") or ""),
                    str(facts.get("paper_technical_intro") or ""),
                ]
            )
        incomplete_copy_fields = [
            field
            for field, value in (
                ("summary", summary),
                ("paper_plain_summary", facts.get("paper_plain_summary")),
                ("paper_technical_intro", facts.get("paper_technical_intro")),
            )
            if value and not self._editorial_copy_is_complete(value)
        ]
        if incomplete_copy_fields:
            self.fetch_diagnostics["truncated_editorial_copy_count"] += 1
            examples = self.fetch_diagnostics["truncated_editorial_copy_examples"]
            if len(examples) < 10:
                examples.append({
                    "title": title_cn,
                    "fields": incomplete_copy_fields,
                })
            return None
        if any(has_bad_public_phrase(value) for value in public_copy if value):
            self.fetch_diagnostics["bad_editorial_summary_count"] += 1
            return None
        claim_type = str(row.get("claim_type") or facts.get("claim_type") or "").strip().lower()
        allowed_claim_types = {
            "verified_fact",
            "official_claim",
            "interview_opinion",
            "analysis",
            "research_result",
        }
        claim_type_valid = claim_type in allowed_claim_types
        if content_type == "paper":
            claim_type_valid = claim_type == "research_result"
        elif content_type in {"interview", "podcast", "video"}:
            claim_type_valid = claim_type == "interview_opinion"
        elif primary_section == "technical":
            claim_type_valid = claim_type in {"verified_fact", "official_claim", "analysis"}
        if not claim_type_valid:
            self.fetch_diagnostics["claim_type_error_count"] += 1
            return None
        facts["claim_type"] = claim_type
        if content_type in {"interview", "podcast", "video"}:
            evidence_locator = str(facts.get("evidence_locator") or "").strip()
            has_precise_locator = bool(
                re.search(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", evidence_locator)
                or re.search(
                    r"文字稿|逐字稿|字幕|章节|小标题|第\s*\d+\s*段|transcript|chapter|section|timestamp|timecode",
                    evidence_locator,
                    re.IGNORECASE,
                )
            )
            if not has_precise_locator:
                self.fetch_diagnostics["interview_locator_missing_count"] += 1
                return None
        if primary_section == "paper":
            plain_summary = str(facts.get("paper_plain_summary") or "").strip()
            technical_intro = str(facts.get("paper_technical_intro") or "").strip()
            if (
                not 90 <= len(plain_summary) <= 180
                or not 160 <= len(technical_intro) <= 300
                or not paper_plain_summary_passes(plain_summary)
                or not paper_technical_intro_passes(technical_intro)
            ):
                self.fetch_diagnostics["structured_fact_missing_count"] += 1
                return None
        common_fact_keys = ("who", "action", "target")
        common_facts_ready = all(str(facts.get(key) or "").strip() for key in common_fact_keys)
        if primary_section == "technical":
            supporting_keys = (
                "architecture",
                "input_output",
                "dataset_or_benchmark",
                "metric_result",
                "code_or_project",
                "deployment_context",
            )
            structured_facts_ready = (
                common_facts_ready
                and bool(str(facts.get("method") or "").strip())
                and any(str(facts.get(key) or "").strip() for key in supporting_keys)
            )
        elif primary_section == "paper":
            supporting_keys = ("dataset_or_benchmark", "metric_result", "baseline", "limitation", "code_or_project")
            structured_facts_ready = (
                common_facts_ready
                and bool(str(facts.get("method") or "").strip())
                and any(str(facts.get(key) or "").strip() for key in supporting_keys)
            )
        else:
            structured_facts_ready = common_facts_ready
        if not structured_facts_ready:
            self.fetch_diagnostics["structured_fact_missing_count"] += 1
            return None
        metric_result = str(facts.get("metric_result") or "").strip()
        if (
            primary_section in {"technical", "paper"}
            and re.search(r"\d", metric_result)
            and not any(
                str(facts.get(key) or "").strip()
                for key in ("dataset_or_benchmark", "deployment_context", "baseline")
            )
        ):
            self.fetch_diagnostics["numeric_context_missing_count"] += 1
            return None
        quality_flags = list(row.get("quality_flags") or [])
        published_at = self._parse_datetime(row.get("publish_date"))
        if published_at is None:
            self.fetch_diagnostics["missing_publish_date_count"] += 1
            return None
        freshness_hours = {"news": 48, "technical": 168, "paper": 168}[primary_section]
        source_age_hours = (
            (datetime.now(timezone.utc) - published_at).total_seconds() / 3600
            if published_at is not None
            else 0.0
        )
        if source_age_hours < -6:
            self.fetch_diagnostics["future_publish_date_count"] += 1
            return None
        supplemental_source = "supplemental_older_source" in quality_flags
        if source_age_hours > freshness_hours and not supplemental_source:
            self.fetch_diagnostics["stale_source_count"] += 1
            return None
        if (
            supplemental_source
            and source_age_hours
            > self.supplemental_max_age_hours_by_section[primary_section]
        ):
            self.fetch_diagnostics["supplemental_too_old_count"] += 1
            return None
        technical_category = str(row.get("technical_category") or facts.get("technical_category") or "").strip().lower()
        allowed_technical_categories = {
            "embodied_world_model",
            "agent_systems",
            "training_data",
            "inference_deployment",
            "multimodal_architecture",
        }
        if primary_section == "technical" and technical_category not in allowed_technical_categories:
            self.fetch_diagnostics["technical_category_error_count"] += 1
            return None
        if technical_category:
            facts["technical_category"] = technical_category
        if primary_section == "news" and content_type in {"interview", "podcast", "video"}:
            minimum_summary_chars = 300
            maximum_summary_chars = 500
        else:
            minimum_summary_chars = {"news": 180, "technical": 220, "paper": 100}[primary_section]
            maximum_summary_chars = {"news": 300, "technical": 380, "paper": 1000}[primary_section]
        if len(summary) < minimum_summary_chars:
            self.fetch_diagnostics["short_summary_count"] += 1
            return None
        if len(summary) > maximum_summary_chars:
            self.fetch_diagnostics["long_summary_count"] += 1
            return None
        if self._chinese_ratio(summary) < 0.35:
            self.fetch_diagnostics["non_chinese_summary_count"] += 1
            return None
        summary_sentences = self._summary_sentences(summary)
        minimum_sentence_count = 4 if content_type in {"interview", "podcast", "video"} else 3
        if len(summary_sentences) < minimum_sentence_count:
            self.fetch_diagnostics["insufficient_sentence_count"] += 1
            return None
        if (
            content_type in {"interview", "podcast", "video"}
            and len([part for part in summary.split("\n\n") if part.strip()]) < 2
        ):
            self.fetch_diagnostics["paragraph_structure_missing_count"] += 1
            return None
        normalized_sentences = [
            re.sub(r"\W+", "", sentence).lower()
            for sentence in summary_sentences
            if len(re.sub(r"\W+", "", sentence)) >= 12
        ]
        if len(normalized_sentences) != len(set(normalized_sentences)):
            self.fetch_diagnostics["repetitive_summary_count"] += 1
            return None
        if primary_section != "paper" and not self._opening_is_substantive(
            summary,
            facts,
            primary_section,
            content_type,
        ):
            self.fetch_diagnostics["opening_substance_missing_count"] += 1
            return None
        if content_type in {"interview", "podcast", "video"}:
            interview_evidence = " ".join(
                [source_excerpt] + [str(point or "") for point in evidence]
            )
            has_substantive_evidence = bool(re.search(
                r"表示|认为|主张|解释|指出|判断|反对|支持|依据|因为|原因|"
                r"实验|结果|方法|架构|模型|训练|推理|部署|数据|性能|限制|"
                r"风险|争议|分歧|失败|成功|恢复|权限|成本|延迟|吞吐|基线|对照",
                interview_evidence,
                re.IGNORECASE,
            ))
            metadata_only = bool(re.fullmatch(
                r"[\s\S]{0,40}(?:节目|视频|播客).{0,30}(?:时长|上线|发布|播出|日期)"
                r"[\s\S]{0,80}",
                interview_evidence,
            )) and not has_substantive_evidence
            if not has_substantive_evidence or metadata_only:
                self.fetch_diagnostics["interview_evidence_substance_missing_count"] += 1
                return None
        if not self._claim_language_matches(summary, claim_type):
            self.fetch_diagnostics["claim_language_mismatch_count"] += 1
            examples = self.fetch_diagnostics["claim_language_mismatch_examples"]
            if len(examples) < 10:
                examples.append(
                    {
                        "primary_section": primary_section,
                        "title": title_cn,
                        "claim_type": claim_type,
                        "summary_opening": summary[:120],
                        "required_attribution": {
                            "official_claim": "公司称／团队介绍／发布方披露／官方公告显示",
                            "interview_opinion": "受访者认为／嘉宾解释／某人指出",
                            "analysis": "分析认为／据此判断／可能",
                        }.get(claim_type, ""),
                    }
                )
            return None
        if primary_section == "technical":
            validation_keys = (
                "dataset_or_benchmark",
                "metric_result",
                "code_or_project",
                "deployment_context",
            )
            missing_contract_parts = []
            if not str(facts.get("method") or "").strip():
                missing_contract_parts.append("method")
            if not str(facts.get("baseline") or "").strip():
                missing_contract_parts.append("baseline")
            if not str(facts.get("limitation") or "").strip():
                missing_contract_parts.append("limitation")
            if not any(str(facts.get(key) or "").strip() for key in validation_keys):
                missing_contract_parts.append("validation")
            if missing_contract_parts:
                self.fetch_diagnostics["technical_contract_missing_count"] += 1
                examples = self.fetch_diagnostics["technical_contract_missing_examples"]
                if len(examples) < 10:
                    examples.append({
                        "title": title_cn,
                        "missing": missing_contract_parts,
                    })
                return None
        if primary_section == "paper":
            validation_keys = (
                "dataset_or_benchmark",
                "metric_result",
                "code_or_project",
                "deployment_context",
            )
            missing_contract_parts = []
            if not str(facts.get("method") or "").strip():
                missing_contract_parts.append("method")
            if not str(facts.get("baseline") or "").strip():
                missing_contract_parts.append("baseline")
            if not str(facts.get("limitation") or "").strip():
                missing_contract_parts.append("limitation")
            if not any(str(facts.get(key) or "").strip() for key in validation_keys):
                missing_contract_parts.append("validation")
            if missing_contract_parts:
                self.fetch_diagnostics["paper_contract_missing_count"] += 1
                examples = self.fetch_diagnostics["paper_contract_missing_examples"]
                if len(examples) < 10:
                    examples.append({
                        "title": title_cn,
                        "missing": missing_contract_parts,
                    })
                return None
        if primary_section == "paper":
            numeric_public_text = " ".join(
                [
                    str(facts.get("paper_plain_summary") or ""),
                    str(facts.get("paper_technical_intro") or ""),
                ]
            )
        else:
            numeric_public_text = summary
        public_metric_tokens = self._metric_tokens(numeric_public_text)
        numeric_support_text = " ".join(
            [
                source_excerpt,
                str(facts.get("metric_result") or ""),
                str(facts.get("dataset_or_benchmark") or ""),
                str(facts.get("baseline") or ""),
                str(facts.get("deployment_context") or ""),
            ]
            + [str(point or "") for point in evidence]
        )
        supported_metric_tokens = self._metric_tokens(numeric_support_text)
        unsupported_metric_tokens = public_metric_tokens - supported_metric_tokens
        if unsupported_metric_tokens:
            self.fetch_diagnostics["unsupported_summary_numeric_count"] += 1
            examples = self.fetch_diagnostics["unsupported_summary_numeric_examples"]
            if len(examples) < 10:
                examples.append({
                    "primary_section": primary_section,
                    "title": title_cn,
                    "unsupported_tokens": sorted(unsupported_metric_tokens),
                })
            return None
        if self.require_key_numbers:
            raw_key_numbers = facts.get("key_numbers")
            if not isinstance(raw_key_numbers, list):
                self.fetch_diagnostics["key_number_contract_missing_count"] += 1
                examples = self.fetch_diagnostics["key_number_contract_missing_examples"]
                if len(examples) < 10:
                    examples.append({"primary_section": primary_section, "title": title_cn})
                return None
            key_number_tokens = set().union(
                *(self._metric_tokens(value) for value in raw_key_numbers)
            ) if raw_key_numbers else set()
            if raw_key_numbers and not key_number_tokens:
                self.fetch_diagnostics["key_number_contract_missing_count"] += 1
                examples = self.fetch_diagnostics["key_number_contract_missing_examples"]
                if len(examples) < 10:
                    examples.append({
                        "primary_section": primary_section,
                        "title": title_cn,
                        "key_numbers": raw_key_numbers,
                    })
                return None
            if supported_metric_tokens and not key_number_tokens:
                self.fetch_diagnostics["key_number_contract_missing_count"] += 1
                examples = self.fetch_diagnostics["key_number_contract_missing_examples"]
                if len(examples) < 10:
                    examples.append({
                        "primary_section": primary_section,
                        "title": title_cn,
                        "supported_tokens": sorted(supported_metric_tokens),
                    })
                return None
            unsupported_key_tokens = key_number_tokens - supported_metric_tokens
            if unsupported_key_tokens:
                self.fetch_diagnostics["key_number_evidence_missing_count"] += 1
                examples = self.fetch_diagnostics["key_number_evidence_missing_examples"]
                if len(examples) < 10:
                    examples.append({
                        "primary_section": primary_section,
                        "title": title_cn,
                        "unsupported_tokens": sorted(unsupported_key_tokens),
                    })
                return None
            missing_public_tokens = key_number_tokens - public_metric_tokens
            if missing_public_tokens:
                self.fetch_diagnostics["key_number_public_copy_missing_count"] += 1
                examples = self.fetch_diagnostics["key_number_public_copy_missing_examples"]
                if len(examples) < 10:
                    examples.append({
                        "primary_section": primary_section,
                        "title": title_cn,
                        "missing_tokens": sorted(missing_public_tokens),
                    })
                return None
        source_detail = clean_snippet(str(row.get("source_detail") or "Codex Research"), limit=160)
        category = str(row.get("category") or ("模型/研究" if content_type == "paper" else "行业动态"))
        editorial_source_hashes = {
            "title_cn": self._editorial_text_digest(title_cn),
            "summary": self._editorial_text_digest(summary),
            "source_excerpt": self._editorial_text_digest(source_excerpt),
            "evidence_locator": self._editorial_text_digest(evidence_locator),
        }
        if isinstance(facts.get("key_numbers"), list):
            editorial_source_hashes["key_numbers"] = self._editorial_text_digest(
                json.dumps(
                    facts.get("key_numbers"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        if primary_section == "paper":
            editorial_source_hashes.update(
                {
                    "paper_plain_summary": self._editorial_text_digest(
                        facts.get("paper_plain_summary")
                    ),
                    "paper_technical_intro": self._editorial_text_digest(
                        facts.get("paper_technical_intro")
                    ),
                }
            )
        facts["editorial_source_hashes"] = editorial_source_hashes
        paragraph_values = {"summary": summary}
        if primary_section == "paper":
            paragraph_values.update(
                {
                    "paper_plain_summary": facts.get("paper_plain_summary"),
                    "paper_technical_intro": facts.get("paper_technical_intro"),
                }
            )
        editorial_paragraph_hashes = {
            key: digest
            for key, value in paragraph_values.items()
            if (digest := self._editorial_paragraph_digest(value))
        }
        if editorial_paragraph_hashes:
            facts["editorial_paragraph_hashes"] = editorial_paragraph_hashes
        paper_domain_key = ""
        if primary_section == "paper":
            paper_domain_key = self._paper_domain_key({**row, "facts": facts})
            facts["paper_domain_key"] = paper_domain_key
        score = max(0.0, min(10.0, float(row.get("score", 7.0) or 7.0)))
        evidence_quality = max(0.0, min(1.0, float(row.get("evidence_quality", 0.65) or 0.65)))
        information_density = max(0.0, min(1.0, float(row.get("information_density", 0.65) or 0.65)))
        keywords = row.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [item.strip() for item in keywords.split(",") if item.strip()]
        analysis = {
            "summary": summary,
            "score": score,
            "keywords": list(keywords)[:6],
            "category": category,
            "title_cn": title_cn,
            "summary_preview": str(row.get("summary_preview") or ""),
            "why_it_matters": str(row.get("why_it_matters") or ""),
            "why_now": str(row.get("why_now") or ""),
            "expected_effect": str(row.get("expected_effect") or ""),
            "future_impact": str(row.get("future_impact") or ""),
            "facts": facts,
            "evidence_quality": evidence_quality,
            "information_density": information_density,
            "model_used": "codex-automation",
            "analysis_version": CODEX_RESEARCH_ANALYSIS_VERSION,
            "quality_flags": quality_flags,
            "primary_section": primary_section,
            "claim_type": claim_type,
            "paper_domain_key": paper_domain_key,
        }
        item = {
            "source": "Codex Research",
            "source_detail": source_detail,
            "title": title,
            "url": url,
            "canonical_url": url,
            "content": " ".join([summary, str(facts.get("source_excerpt") or "")] + list(evidence)),
            "publish_date": str(row.get("publish_date") or ""),
            "author": str(row.get("author") or source_detail),
            "content_type": content_type,
            "platform": str(row.get("platform") or infer_platform(url, source_detail) or "Website"),
            "topic": str(row.get("topic") or row.get("domain_key") or "AI"),
            "initial_score": score,
            "score": score,
            "facts": facts,
            "evidence_quality": evidence_quality,
            "information_density": information_density,
            "model_used": "codex-automation",
            "analysis_version": CODEX_RESEARCH_ANALYSIS_VERSION,
            "quality_flags": quality_flags,
            "primary_section": primary_section,
            "claim_type": claim_type,
            "paper_domain_key": paper_domain_key,
            "_codex_research_analysis": analysis,
        }
        item["source_tier"] = infer_source_tier(item)
        return item

    def collect(self) -> List[Dict[str, Any]]:
        if not self.inbox_path.exists():
            self.fetch_diagnostics["quality_status"] = "missing_inbox"
            return []
        try:
            raw_bytes = self.inbox_path.read_bytes()
            self.fetch_diagnostics["inbox_sha256"] = hashlib.sha256(
                raw_bytes
            ).hexdigest()
            raw_payload = raw_bytes.decode("utf-8-sig")
            payload = json.loads(raw_payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.fetch_diagnostics["schema_error_count"] = 1
            self.fetch_diagnostics["quality_status"] = "invalid_json"
            self.fetch_diagnostics["error"] = str(exc)
            return []

        generated_at = self._parse_datetime(
            payload.get("generated_at") if isinstance(payload, dict) else ""
        )
        if generated_at is not None:
            self.fetch_diagnostics["generated_at"] = generated_at.isoformat()
            age_minutes = max(
                0.0,
                (datetime.now(timezone.utc) - generated_at).total_seconds() / 60,
            )
            self.fetch_diagnostics["age_minutes"] = round(age_minutes, 1)
            self.fetch_diagnostics["fresh"] = age_minutes <= self.max_age_minutes

        schema_version = str(payload.get("schema_version") or "").strip() if isinstance(payload, dict) else ""
        self.fetch_diagnostics["schema_version"] = schema_version
        self.fetch_diagnostics["schema_version_status"] = (
            "passed"
            if not self.required_schema_version
            or schema_version == self.required_schema_version
            else "failed"
        )
        self.require_key_numbers = schema_version == CODEX_RESEARCH_SCHEMA_VERSION
        if self.required_schema_version and schema_version != self.required_schema_version:
            self.fetch_diagnostics["schema_error_count"] = 1
            self.fetch_diagnostics["quality_status"] = "schema_version_mismatch"
            return []

        if generated_at is None:
            self.fetch_diagnostics["schema_error_count"] = 1
            self.fetch_diagnostics["quality_status"] = "missing_generated_at"
            return []
        if not self.fetch_diagnostics["fresh"]:
            self.fetch_diagnostics["quality_status"] = "stale_inbox"
            return []

        rows = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            self.fetch_diagnostics["schema_error_count"] = 1
            self.fetch_diagnostics["quality_status"] = "missing_items"
            return []

        discovery_required = bool(
            self.minimum_discovery_candidates or self.minimum_discovered_by_section
        )
        discovery_rows = payload.get("discovery_candidates") if isinstance(payload, dict) else None
        if not isinstance(discovery_rows, list):
            discovery_rows = []
        discovered_url_ids = set()
        discovery_section_counts = {section: 0 for section in ("news", "technical", "paper")}
        discovery_duplicate_url_count = 0
        discovery_invalid_row_count = 0
        for candidate in discovery_rows:
            if not isinstance(candidate, dict):
                discovery_invalid_row_count += 1
                continue
            section = str(candidate.get("primary_section") or "").strip().lower()
            url_identity = self._url_identity(candidate.get("url"))
            if section not in discovery_section_counts or not url_identity:
                discovery_invalid_row_count += 1
                continue
            if url_identity in discovered_url_ids:
                discovery_duplicate_url_count += 1
                continue
            discovered_url_ids.add(url_identity)
            discovery_section_counts[section] += 1
        discovery_underfilled = []
        if len(discovered_url_ids) < self.minimum_discovery_candidates:
            discovery_underfilled.append(
                f"total:{len(discovered_url_ids)}/{self.minimum_discovery_candidates}"
            )
        discovery_underfilled.extend(
            f"{section}:{discovery_section_counts.get(section, 0)}/{minimum}"
            for section, minimum in self.minimum_discovered_by_section.items()
            if discovery_section_counts.get(section, 0) < minimum
        )
        submitted_not_in_discovery = []
        if discovery_required:
            for row in rows:
                if not isinstance(row, dict):
                    continue
                url_identity = self._url_identity(row.get("url"))
                if url_identity and url_identity not in discovered_url_ids:
                    submitted_not_in_discovery.append(str(row.get("url") or ""))
        discovery_quota_passed = (
            not discovery_underfilled
            and not submitted_not_in_discovery
            and (not discovery_required or not discovery_invalid_row_count)
        )
        self.fetch_diagnostics["discovery_candidate_count"] = len(discovered_url_ids)
        discovery_manifest_material = "\n".join(sorted(
            f"{str(candidate.get('primary_section') or '').strip().lower()}|"
            f"{self._url_identity(candidate.get('url'))}"
            for candidate in discovery_rows
            if isinstance(candidate, dict)
            and str(candidate.get("primary_section") or "").strip().lower()
            in discovery_section_counts
            and self._url_identity(candidate.get("url"))
        ))
        self.fetch_diagnostics["discovery_manifest_sha256"] = (
            hashlib.sha256(discovery_manifest_material.encode("utf-8")).hexdigest()
            if discovery_manifest_material
            else ""
        )
        self.fetch_diagnostics["discovery_section_counts"] = discovery_section_counts
        self.fetch_diagnostics["discovery_duplicate_url_count"] = discovery_duplicate_url_count
        self.fetch_diagnostics["discovery_invalid_row_count"] = discovery_invalid_row_count
        self.fetch_diagnostics["submitted_not_in_discovery_count"] = len(
            submitted_not_in_discovery
        )
        self.fetch_diagnostics["submitted_not_in_discovery_examples"] = (
            submitted_not_in_discovery[:10]
        )
        self.fetch_diagnostics["discovery_underfilled"] = discovery_underfilled
        self.fetch_diagnostics["discovery_quota_status"] = (
            "passed" if discovery_quota_passed else "failed"
        )

        submitted_section_counts = {section: 0 for section in ("news", "technical", "paper")}
        for row in rows:
            if not isinstance(row, dict):
                continue
            section = str(row.get("primary_section") or "").strip().lower()
            if section in submitted_section_counts:
                submitted_section_counts[section] += 1
        rejected_section_counts = {section: 0 for section in ("news", "technical", "paper", "unknown")}
        rejection_reason_counts_by_section: Dict[str, Dict[str, int]] = {
            section: {} for section in rejected_section_counts
        }
        self.fetch_diagnostics["submitted_item_count"] = len(rows)
        self.fetch_diagnostics["submitted_section_counts"] = submitted_section_counts
        submission_underfilled = [
            f"{section}:{submitted_section_counts.get(section, 0)}/{minimum}"
            for section, minimum in self.minimum_submitted_by_section.items()
            if submitted_section_counts.get(section, 0) < minimum
        ]
        self.fetch_diagnostics["submission_underfilled"] = submission_underfilled
        self.fetch_diagnostics["submission_quota_status"] = (
            "failed" if submission_underfilled else "passed"
        )

        items: List[Dict[str, Any]] = []
        seen_urls: Dict[str, Dict[str, Any]] = {}
        seen_events: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            rejection_counts_before = {
                key: int(self.fetch_diagnostics.get(key, 0) or 0)
                for key in self.ITEM_REJECTION_DIAGNOSTICS
            }
            item = self._normalize_item(row) if isinstance(row, dict) else None
            url_identity = self._url_identity(item["url"]) if item else ""
            event_identity = self._event_identity(item) if item else ""
            if not item or url_identity in seen_urls or (event_identity and event_identity in seen_events):
                self.fetch_diagnostics["rejected_item_count"] += 1
                submitted_section = (
                    str(row.get("primary_section") or "").strip().lower()
                    if isinstance(row, dict)
                    else "unknown"
                )
                if submitted_section not in rejected_section_counts:
                    submitted_section = "unknown"
                rejected_section_counts[submitted_section] += 1
                if item and url_identity in seen_urls:
                    self.fetch_diagnostics["duplicate_url_count"] += 1
                    examples = self.fetch_diagnostics["duplicate_url_examples"]
                    if len(examples) < 10:
                        examples.append(
                            self._duplicate_example(item, seen_urls[url_identity])
                        )
                    rejection_reasons = ["duplicate_url"]
                elif item and event_identity and event_identity in seen_events:
                    self.fetch_diagnostics["duplicate_event_count"] += 1
                    examples = self.fetch_diagnostics["duplicate_event_examples"]
                    if len(examples) < 10:
                        examples.append(
                            self._duplicate_example(item, seen_events[event_identity])
                        )
                    rejection_reasons = ["duplicate_event"]
                else:
                    rejection_reasons = [
                        key.removesuffix("_count")
                        for key in self.ITEM_REJECTION_DIAGNOSTICS
                        if int(self.fetch_diagnostics.get(key, 0) or 0) > rejection_counts_before[key]
                    ] or ["unknown_validation_error"]
                section_reasons = rejection_reason_counts_by_section[submitted_section]
                for reason in rejection_reasons:
                    section_reasons[reason] = section_reasons.get(reason, 0) + 1
                continue
            seen_urls[url_identity] = item
            if event_identity:
                seen_events[event_identity] = item
            items.append(item)

        paper_count = sum(1 for item in items if item.get("primary_section") == "paper")
        news_count = sum(1 for item in items if item.get("primary_section") == "news")
        technical_count = sum(1 for item in items if item.get("primary_section") == "technical")
        technical_primary_source_count = sum(
            1
            for item in items
            if item.get("primary_section") == "technical"
            and str(item.get("source_tier") or "").lower() in {"official", "research", "primary"}
        )
        technical_primary_source_ratio = (
            round(technical_primary_source_count / technical_count, 3)
            if technical_count
            else 1.0
        )
        technical_primary_source_status = (
            "passed"
            if technical_primary_source_ratio >= self.technical_primary_source_ratio_min
            else "failed"
        )
        accepted_section_counts = {
            "news": news_count,
            "technical": technical_count,
            "paper": paper_count,
        }
        accepted_section_rates = {
            section: round(accepted_section_counts[section] / submitted, 3) if submitted else 0.0
            for section, submitted in submitted_section_counts.items()
        }
        technical_category_counts: Dict[str, int] = {}
        for item in items:
            if item.get("primary_section") != "technical":
                continue
            facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
            key = str(facts.get("technical_category") or "")
            technical_category_counts[key] = technical_category_counts.get(key, 0) + 1
        technical_category_target_underfilled = [
            f"{key}:{technical_category_counts.get(key, 0)}/{target}"
            for key, target in self.technical_category_quotas.items()
            if technical_category_counts.get(key, 0) < target
        ]
        technical_category_underfilled = [
            f"{key}:{technical_category_counts.get(key, 0)}/{minimum}"
            for key, minimum in self.technical_category_minimums.items()
            if technical_category_counts.get(key, 0) < minimum
        ]
        news_format_counts = {
            "interview_or_podcast": sum(
                1
                for item in items
                if item.get("primary_section") == "news"
                and str(item.get("content_type") or "").lower() in {"interview", "podcast", "video"}
            ),
            "blog": sum(
                1
                for item in items
                if item.get("primary_section") == "news"
                and str(item.get("platform") or "").lower() == "blog"
            ),
        }
        news_format_underfilled = [
            f"{key}:{news_format_counts.get(key, 0)}/{minimum}"
            for key, minimum in self.news_format_quotas.items()
            if news_format_counts.get(key, 0) < minimum
        ]
        paper_domain_counts: Dict[str, int] = {}
        for item in items:
            if item.get("primary_section") != "paper":
                continue
            key = self._paper_domain_key(item)
            paper_domain_counts[key] = paper_domain_counts.get(key, 0) + 1
        paper_domain_underfilled = []
        paper_domain_exceeded = []
        for key, limits in self.paper_domain_quotas.items():
            limits = dict(limits or {})
            count = paper_domain_counts.get(key, 0)
            minimum = max(0, int(limits.get("min", 0) or 0))
            maximum = max(minimum, int(limits.get("max", max(paper_count, minimum)) or max(paper_count, minimum)))
            if count < minimum:
                paper_domain_underfilled.append(f"{key}:{count}/{minimum}")
            if count > maximum:
                paper_domain_exceeded.append(f"{key}:{count}/{maximum}")
        fresh_source_counts = {section: 0 for section in ("news", "technical", "paper")}
        supplemental_older_counts = {section: 0 for section in ("news", "technical", "paper")}
        for item in items:
            section = str(item.get("primary_section") or "")
            if section not in fresh_source_counts:
                continue
            if "supplemental_older_source" in set(item.get("quality_flags") or []):
                supplemental_older_counts[section] += 1
            else:
                fresh_source_counts[section] += 1
        freshness_underfilled = [
            f"{section}:{fresh_source_counts.get(section, 0)}/{minimum}"
            for section, minimum in self.minimum_fresh_by_section.items()
            if fresh_source_counts.get(section, 0) < minimum
        ]
        key_number_item_counts = {
            section: sum(
                1
                for item in items
                if item.get("primary_section") == section
                and bool((item.get("facts") or {}).get("key_numbers"))
            )
            for section in ("news", "technical", "paper")
        }
        key_number_underfilled = (
            [
                f"{section}:{key_number_item_counts.get(section, 0)}/{minimum}"
                for section, minimum in self.minimum_key_number_items_by_section.items()
                if key_number_item_counts.get(section, 0) < minimum
            ]
            if self.require_key_numbers
            else []
        )
        summary_char_stats_by_section = {}
        summary_paragraph_stats_by_section = {}
        for section in ("news", "technical", "paper"):
            section_items = [
                item for item in items if item.get("primary_section") == section
            ]
            summaries = [
                str(
                    (item.get("_codex_research_analysis") or {}).get("summary")
                    or item.get("summary")
                    or ""
                )
                for item in section_items
            ]
            summary_char_stats_by_section[section] = self._integer_stats(
                [len(value) for value in summaries]
            )
            summary_paragraph_stats_by_section[section] = self._integer_stats(
                [len([part for part in value.split("\n\n") if part.strip()]) for value in summaries]
            )
        accepted_content_type_counts: Dict[str, int] = {}
        for item in items:
            key = str(item.get("content_type") or "unknown").strip().lower()
            accepted_content_type_counts[key] = accepted_content_type_counts.get(key, 0) + 1
        paper_plain_summary_char_stats = self._integer_stats([
            len(str((item.get("facts") or {}).get("paper_plain_summary") or ""))
            for item in items
            if item.get("primary_section") == "paper"
        ])
        paper_technical_intro_char_stats = self._integer_stats([
            len(str((item.get("facts") or {}).get("paper_technical_intro") or ""))
            for item in items
            if item.get("primary_section") == "paper"
        ])
        attribution_opener_counts: Dict[str, int] = {}
        attribution_opener_examples: Dict[str, List[str]] = {}
        for item in items:
            if item.get("primary_section") == "paper":
                continue
            body = str(
                (item.get("_codex_research_analysis") or {}).get("summary")
                or item.get("summary")
                or ""
            )
            pattern = attribution_opener_pattern(body)
            if not pattern:
                continue
            attribution_opener_counts[pattern] = (
                attribution_opener_counts.get(pattern, 0) + 1
            )
            attribution_opener_examples.setdefault(pattern, []).append(
                str(item.get("title_cn") or item.get("title") or "")
            )
        attribution_opener_overuse = {
            pattern: count
            for pattern, count in attribution_opener_counts.items()
            if count > self.max_attribution_opener_count
        }
        attribution_opener_overuse_examples = [
            {
                "pattern": pattern,
                "count": count,
                "titles": attribution_opener_examples.get(pattern, [])[:5],
            }
            for pattern, count in sorted(
                attribution_opener_overuse.items(),
                key=lambda row: (-row[1], row[0]),
            )
        ]
        cross_item_template_repeats = self._cross_item_template_repeats(items)
        self.fetch_diagnostics["paper_count"] = paper_count
        self.fetch_diagnostics["news_count"] = news_count
        self.fetch_diagnostics["technical_count"] = technical_count
        self.fetch_diagnostics["technical_primary_source_count"] = technical_primary_source_count
        self.fetch_diagnostics["technical_primary_source_ratio"] = technical_primary_source_ratio
        self.fetch_diagnostics["technical_primary_source_status"] = technical_primary_source_status
        self.fetch_diagnostics["rejected_section_counts"] = rejected_section_counts
        self.fetch_diagnostics["rejection_reason_counts_by_section"] = rejection_reason_counts_by_section
        self.fetch_diagnostics["accepted_section_rates"] = accepted_section_rates
        self.fetch_diagnostics["technical_category_counts"] = technical_category_counts
        self.fetch_diagnostics["technical_category_underfilled"] = technical_category_underfilled
        self.fetch_diagnostics["technical_category_target_underfilled"] = (
            technical_category_target_underfilled
        )
        self.fetch_diagnostics["technical_target_status"] = (
            "failed" if technical_category_target_underfilled else "passed"
        )
        self.fetch_diagnostics["technical_quota_status"] = "failed" if technical_category_underfilled else "passed"
        self.fetch_diagnostics["news_format_counts"] = news_format_counts
        self.fetch_diagnostics["news_format_underfilled"] = news_format_underfilled
        self.fetch_diagnostics["news_format_quota_status"] = "failed" if news_format_underfilled else "passed"
        self.fetch_diagnostics["paper_domain_counts"] = paper_domain_counts
        self.fetch_diagnostics["paper_domain_underfilled"] = paper_domain_underfilled
        self.fetch_diagnostics["paper_domain_exceeded"] = paper_domain_exceeded
        self.fetch_diagnostics["paper_domain_quota_status"] = (
            "failed" if paper_domain_underfilled or paper_domain_exceeded else "passed"
        )
        self.fetch_diagnostics["fresh_source_counts"] = fresh_source_counts
        self.fetch_diagnostics["supplemental_older_counts"] = supplemental_older_counts
        self.fetch_diagnostics["freshness_underfilled"] = freshness_underfilled
        self.fetch_diagnostics["freshness_quota_status"] = "failed" if freshness_underfilled else "passed"
        self.fetch_diagnostics["key_number_item_counts"] = key_number_item_counts
        self.fetch_diagnostics["key_number_underfilled"] = key_number_underfilled
        self.fetch_diagnostics["key_number_quota_status"] = (
            "failed" if key_number_underfilled else "passed"
        )
        self.fetch_diagnostics["summary_char_stats_by_section"] = summary_char_stats_by_section
        self.fetch_diagnostics["summary_paragraph_stats_by_section"] = summary_paragraph_stats_by_section
        self.fetch_diagnostics["accepted_content_type_counts"] = accepted_content_type_counts
        self.fetch_diagnostics["paper_plain_summary_char_stats"] = paper_plain_summary_char_stats
        self.fetch_diagnostics["paper_technical_intro_char_stats"] = paper_technical_intro_char_stats
        self.fetch_diagnostics["attribution_opener_counts"] = attribution_opener_counts
        self.fetch_diagnostics["attribution_opener_overuse_count"] = len(
            attribution_opener_overuse
        )
        self.fetch_diagnostics["attribution_opener_overuse_examples"] = (
            attribution_opener_overuse_examples
        )
        self.fetch_diagnostics["cross_item_template_repeat_count"] = len(
            cross_item_template_repeats
        )
        self.fetch_diagnostics["cross_item_template_repeat_examples"] = (
            cross_item_template_repeats[:10]
        )
        meets_minimums = (
            len(items) >= self.minimum_items
            and paper_count >= self.minimum_papers
            and news_count >= self.minimum_news
            and technical_count >= self.minimum_technical
            and technical_primary_source_status == "passed"
            and not technical_category_underfilled
            and not news_format_underfilled
            and not paper_domain_underfilled
            and not paper_domain_exceeded
            and not freshness_underfilled
            and not key_number_underfilled
            and not submission_underfilled
            and discovery_quota_passed
            and not attribution_opener_overuse
            and len(cross_item_template_repeats)
            <= self.max_cross_item_template_repeat_count
        )
        self.fetch_diagnostics["quality_status"] = "passed" if meets_minimums else "underfilled"
        self.fetch_diagnostics["collected_count"] = len(items)
        return items if meets_minimums else []


def build_codex_research_readiness_summary(
    diagnostics: Dict[str, Any],
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Condense the research inbox contract into one operator-facing decision."""
    config = dict(config or {})
    submitted = {
        section: int((diagnostics.get("submitted_section_counts") or {}).get(section, 0) or 0)
        for section in ("news", "technical", "paper")
    }
    accepted = {
        "news": int(diagnostics.get("news_count", 0) or 0),
        "technical": int(diagnostics.get("technical_count", 0) or 0),
        "paper": int(diagnostics.get("paper_count", 0) or 0),
    }
    accepted_minimums = {
        "news": int(
            20 if config.get("minimum_news") is None else config["minimum_news"]
        ),
        "technical": int(
            20
            if config.get("minimum_technical") is None
            else config["minimum_technical"]
        ),
        "paper": int(
            15 if config.get("minimum_papers") is None else config["minimum_papers"]
        ),
    }
    accepted_underfilled = [
        f"{section}:{accepted[section]}/{minimum}"
        for section, minimum in accepted_minimums.items()
        if accepted[section] < minimum
    ]
    accepted_rates = dict(diagnostics.get("accepted_section_rates") or {})
    for section in submitted:
        if section not in accepted_rates:
            accepted_rates[section] = (
                round(accepted[section] / submitted[section], 3)
                if submitted[section]
                else 0.0
            )

    rejection_rows = []
    for section, reasons in dict(
        diagnostics.get("rejection_reason_counts_by_section") or {}
    ).items():
        for reason, count in dict(reasons or {}).items():
            numeric_count = int(count or 0)
            if numeric_count:
                rejection_rows.append(
                    {"section": str(section), "reason": str(reason), "count": numeric_count}
                )
    rejection_rows.sort(key=lambda row: (-row["count"], row["section"], row["reason"]))

    quality_status = str(diagnostics.get("quality_status") or "unknown")
    production_ready_status = str(
        diagnostics.get("production_ready_status") or "not_checked"
    )
    blockers = []
    required_schema_version = str(config.get("required_schema_version") or "").strip()
    actual_schema_version = str(diagnostics.get("schema_version") or "").strip()
    schema_version_blocked = bool(
        required_schema_version and actual_schema_version != required_schema_version
    )
    if schema_version_blocked:
        blockers.append(
            f"schema_version={actual_schema_version or 'missing'}"
        )
    if not schema_version_blocked and not bool(diagnostics.get("fresh", False)):
        blockers.append("stale_inbox")
    if quality_status != "passed":
        blockers.append(f"quality_status={quality_status}")
    if not schema_version_blocked and str(
        diagnostics.get("submission_quota_status") or "not_loaded"
    ) != "passed":
        blockers.append("submission_quota")
    if not schema_version_blocked and str(
        diagnostics.get("discovery_quota_status") or "not_loaded"
    ) != "passed":
        blockers.append("discovery_quota")
    if (
        not schema_version_blocked
        and str(diagnostics.get("schema_version") or "")
        == CODEX_RESEARCH_SCHEMA_VERSION
        and str(diagnostics.get("key_number_quota_status") or "not_loaded") != "passed"
    ):
        blockers.append("key_number_quota")
    if not schema_version_blocked and accepted_underfilled:
        blockers.append("accepted_section_minimums")
    if not schema_version_blocked and int(
        diagnostics.get("attribution_opener_overuse_count", 0) or 0
    ):
        blockers.append("attribution_opener_overuse")
    maximum_template_repeats = int(
        config.get("max_cross_item_template_repeat_count", 10000) or 0
    )
    if not schema_version_blocked and int(
        diagnostics.get("cross_item_template_repeat_count", 0) or 0
    ) > maximum_template_repeats:
        blockers.append("cross_item_template_repetition")
    if not schema_version_blocked and production_ready_status != "passed":
        blockers.append(f"production_ready_status={production_ready_status}")

    ready_for_dry_run = not blockers
    return {
        "status": "ready" if ready_for_dry_run else "blocked",
        "ready_for_dry_run": ready_for_dry_run,
        "fresh": bool(diagnostics.get("fresh", False)),
        "generated_at": str(diagnostics.get("generated_at") or ""),
        "age_minutes": diagnostics.get("age_minutes"),
        "inbox_sha256": str(diagnostics.get("inbox_sha256") or ""),
        "schema_version": actual_schema_version,
        "required_schema_version": required_schema_version,
        "schema_version_status": str(
            diagnostics.get("schema_version_status") or "not_loaded"
        ),
        "submitted_counts": submitted,
        "accepted_counts": accepted,
        "accepted_minimums": accepted_minimums,
        "accepted_rates": accepted_rates,
        "rejected_count": int(diagnostics.get("rejected_item_count", 0) or 0),
        "top_rejection_reasons": rejection_rows[:10],
        "discovery_candidate_count": int(
            diagnostics.get("discovery_candidate_count", 0) or 0
        ),
        "discovery_manifest_sha256": str(
            diagnostics.get("discovery_manifest_sha256") or ""
        ),
        "discovery_section_counts": dict(
            diagnostics.get("discovery_section_counts") or {}
        ),
        "discovery_duplicate_url_count": int(
            diagnostics.get("discovery_duplicate_url_count", 0) or 0
        ),
        "discovery_invalid_row_count": int(
            diagnostics.get("discovery_invalid_row_count", 0) or 0
        ),
        "submitted_not_in_discovery_count": int(
            diagnostics.get("submitted_not_in_discovery_count", 0) or 0
        ),
        "discovery_underfilled": list(diagnostics.get("discovery_underfilled") or []),
        "submission_underfilled": list(diagnostics.get("submission_underfilled") or []),
        "accepted_underfilled": accepted_underfilled,
        "technical_category_underfilled": list(
            diagnostics.get("technical_category_underfilled") or []
        ),
        "news_format_underfilled": list(diagnostics.get("news_format_underfilled") or []),
        "paper_domain_underfilled": list(diagnostics.get("paper_domain_underfilled") or []),
        "paper_domain_exceeded": list(diagnostics.get("paper_domain_exceeded") or []),
        "freshness_underfilled": list(diagnostics.get("freshness_underfilled") or []),
        "key_number_item_counts": dict(diagnostics.get("key_number_item_counts") or {}),
        "key_number_underfilled": list(diagnostics.get("key_number_underfilled") or []),
        "key_number_quota_status": str(
            diagnostics.get("key_number_quota_status") or "not_loaded"
        ),
        "attribution_opener_counts": dict(
            diagnostics.get("attribution_opener_counts") or {}
        ),
        "attribution_opener_overuse_count": int(
            diagnostics.get("attribution_opener_overuse_count", 0) or 0
        ),
        "attribution_opener_overuse_examples": list(
            diagnostics.get("attribution_opener_overuse_examples") or []
        ),
        "cross_item_template_repeat_count": int(
            diagnostics.get("cross_item_template_repeat_count", 0) or 0
        ),
        "cross_item_template_repeat_examples": list(
            diagnostics.get("cross_item_template_repeat_examples") or []
        ),
        "sent_history_overlap_count": diagnostics.get("sent_history_overlap_count"),
        "sent_history_overlap_by_section": dict(
            diagnostics.get("sent_history_overlap_by_section") or {}
        ),
        "blockers": list(dict.fromkeys(blockers)),
    }


def build_codex_research_inbox_collector(
    config: Dict[str, Any],
    *,
    root: Path,
    inbox_path_override: Path | None = None,
) -> CodexResearchInboxCollector:
    """Build the inbox collector from the single authoritative config mapping."""
    inbox_path = Path(inbox_path_override) if inbox_path_override is not None else Path(
        str(
            os.getenv("WEB_AGENT_CODEX_RESEARCH_INBOX_PATH", "")
            or config.get("path")
            or "data/codex_research/latest.json"
        )
    )
    if not inbox_path.is_absolute():
        inbox_path = root / inbox_path
    return CodexResearchInboxCollector(
        str(inbox_path),
        max_age_minutes=int(config.get("max_age_minutes", 240) or 240),
        minimum_items=int(config.get("minimum_items", 55) or 55),
        minimum_papers=int(
            15 if config.get("minimum_papers") is None else config["minimum_papers"]
        ),
        minimum_news=int(
            20 if config.get("minimum_news") is None else config["minimum_news"]
        ),
        minimum_technical=int(
            20
            if config.get("minimum_technical") is None
            else config["minimum_technical"]
        ),
        technical_primary_source_ratio_min=float(
            config.get("technical_primary_source_ratio_min", 0.0) or 0.0
        ),
        minimum_discovery_candidates=int(
            config.get("minimum_discovery_candidates", 0) or 0
        ),
        minimum_discovered_by_section=dict(
            config.get("minimum_discovered_by_section") or {}
        ),
        minimum_submitted_by_section=dict(config.get("minimum_submitted_by_section") or {}),
        minimum_key_number_items_by_section=dict(
            config.get("minimum_key_number_items_by_section") or {}
        ),
        required_schema_version=str(config.get("required_schema_version") or ""),
        technical_category_quotas=dict(config.get("technical_category_quotas") or {}),
        technical_category_minimums=(
            dict(config.get("technical_category_minimums") or {})
            if "technical_category_minimums" in config
            else None
        ),
        news_format_quotas=dict(config.get("news_format_quotas") or {}),
        paper_domain_quotas=dict(config.get("paper_domain_quotas") or {}),
        minimum_fresh_by_section=dict(config.get("minimum_fresh_by_section") or {}),
        supplemental_max_age_hours_by_section=dict(
            config.get("supplemental_max_age_hours_by_section") or {}
        ),
        max_attribution_opener_count=int(
            config.get("max_attribution_opener_count", 8) or 8
        ),
        max_cross_item_template_repeat_count=int(
            config.get("max_cross_item_template_repeat_count", 10000)
        ),
        cross_item_template_similarity_threshold=float(
            config.get("cross_item_template_similarity_threshold", 0.92)
        ),
    )


def evaluate_sent_history_overlap(
    items: List[Dict[str, Any]],
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    history_urls = {
        CodexResearchInboxCollector._url_identity(item.get("canonical_url") or item.get("url"))
        for item in history
        if item.get("canonical_url") or item.get("url")
    }
    history_events = {
        CodexResearchInboxCollector._event_identity(item)
        for item in history
        if CodexResearchInboxCollector._event_identity(item)
    }
    overlaps = []
    for item in items:
        url_identity = CodexResearchInboxCollector._url_identity(
            item.get("canonical_url") or item.get("url")
        )
        event_identity = CodexResearchInboxCollector._event_identity(item)
        if url_identity in history_urls or (event_identity and event_identity in history_events):
            overlaps.append(item)
    counts: Dict[str, int] = {}
    for item in overlaps:
        section = str(item.get("primary_section") or "unknown")
        counts[section] = counts.get(section, 0) + 1
    return {
        "sent_history_overlap_count": len(overlaps),
        "sent_history_overlap_by_section": counts,
        "sent_history_overlap_examples": [
            str(item.get("canonical_url") or item.get("url") or "")
            for item in overlaps[:10]
        ],
        "production_ready_status": "failed" if overlaps else "passed",
    }
