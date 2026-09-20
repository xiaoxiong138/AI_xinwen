from __future__ import annotations

import os
import re
import hashlib
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.editorial_engine import (
    contains_mojibake,
    enrich_editorial_fields,
    paper_plain_summary_passes,
    paper_technical_intro_passes,
)


def editorial_source_identity(item: Dict[str, Any]) -> str:
    """Return the stable report identity used to trace a guide row to its body item."""
    url = str(item.get("canonical_url") or item.get("url") or "").strip()
    if url:
        return f"url:{url}"
    article_id = str(item.get("id") or item.get("article_id") or "").strip()
    if article_id:
        return f"article:{article_id}"
    title = re.sub(
        r"\W+",
        "",
        str(item.get("editorial_title") or item.get("title_cn") or item.get("title") or "").lower(),
    )
    return f"title:{title}" if title else ""


def editorial_item_render_key(item: Dict[str, Any]) -> str:
    """Return a compact stable key for matching a rendered card to its source item."""
    identity = editorial_source_identity(item)
    if not identity:
        return ""
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


class ReportGenerator:
    DESIGN_VERSION = "v7-classic-briefing"
    INTERLEAVE_PATTERN = ("update", "paper", "update")
    MAX_CONSECUTIVE_SAME_KIND = 2
    REPORT_SECTION_ORDER = ("must_read", "physical_ai", "watch", "featured_papers", "paper_appendix", "brief")
    LEARNING_DOMAIN_ORDER = (
        "world_model",
        "physical_ai",
        "agent_models",
        "infra_open_source",
        "products_business",
    )
    LEARNING_DOMAIN_LABELS = {
        "world_model": "World Model",
        "physical_ai": "Physical AI / Robotics",
        "agent_models": "Agent / Models",
        "infra_open_source": "Infra / Open Source",
        "products_business": "Products / Business",
    }
    V11_DOMAIN_LABELS = {
        "world_model": "世界模型",
        "physical_ai": "具身智能 / 机器人",
        "agent_models": "智能体 / 模型",
        "infra_open_source": "推理基础设施 / 开源",
        "products_business": "产品 / 产业",
    }
    V11_INLINE_TAG_STYLES = {
        "body": "margin:0;padding:10px;background:#f3f5f6;color:#17212b;font-family:Segoe UI,PingFang SC,Microsoft YaHei,Arial,sans-serif;line-height:1.62",
        "h1": "margin:0 0 9px;font-size:24px;line-height:1.22;font-weight:800;color:#17212b",
    }
    V11_INLINE_CLASS_STYLES = {
        "container": "width:100%;max-width:720px;margin:0 auto;background:#ffffff;border:1px solid #dfe5e8;overflow:hidden",
        "masthead": "padding:26px 18px 20px;background:#fbfcfc;border-bottom:1px solid #dfe5e8",
        "eyebrow": "color:#0f665f;font-size:12px;font-weight:800;margin-bottom:10px",
        "edition-title": "margin:-2px 0 9px;color:#314454;font-size:15px;line-height:1.4;font-weight:700",
        "sub": "color:#66717f;font-size:12px",
        "v10-lead": "margin-top:22px;padding-top:17px;border-top:2px solid #0f665f",
        "v10-lead-title": "margin-bottom:9px;color:#17212b;font-size:15px;font-weight:800",
        "v10-decision": "padding:10px 0;border-bottom:1px solid #e7ecee",
        "v10-decision-domain": "color:#0f665f;font-size:11px;font-weight:800",
        "v10-decision-text": "margin-top:2px;color:#263544;font-size:14px;line-height:1.62",
        "v10-decision-evidence": "margin-top:4px;color:#66717f;font-size:12px;line-height:1.55",
        "v11-counts": "width:100%;margin-top:14px;border-collapse:collapse;border-top:1px solid #dce5e3;border-bottom:1px solid #dce5e3",
        "v11-count-value": "color:#17212b;font-size:20px;font-weight:900;line-height:1.15",
        "v11-count-label": "margin-top:3px;color:#66717f;font-size:11px",
        "v11-edition-total": "margin-top:8px;color:#66717f;font-size:11px;line-height:1.5",
        "v10-nav": "margin-top:13px;font-size:12px;line-height:1.8",
        "content": "padding:26px 18px 34px",
        "section": "margin-top:32px",
        "section-head": "margin-bottom:16px;border-bottom:1px solid #dfe5e8;padding-bottom:10px",
        "v11-section-intro": "margin:-4px 0 9px;padding:11px 13px;border-left:3px solid #0f665f;background:#f5f8f7;color:#465663;font-size:13px;line-height:1.6",
        "v10-entry": "padding:17px 0;border-bottom:1px solid #e4eaec",
        "v10-kicker": "margin-bottom:5px;color:#0f665f;font-size:11px;font-weight:800",
        "v11-claim-label": "color:#315d4a;font-weight:800",
        "v10-title": "margin:0 0 9px;color:#17212b;font-size:17px;line-height:1.4;font-weight:800",
        "v10-body": "color:#263544;font-size:16px;line-height:1.72;white-space:pre-line",
        "v10-paper-plain": "margin:2px 0 0;padding:12px 13px 13px;border-left:4px solid #2f7d6d;background:#f2f8f6;color:#1d2a36;font-size:16px;line-height:1.78;font-weight:600;white-space:pre-line",
        "v10-paper-tech": "margin-top:13px;padding:0 2px;color:#526170;font-size:16px;line-height:1.75;white-space:pre-line",
        "v10-evidence": "margin-top:11px;padding:9px 11px;border-left:3px solid #82a99f;background:#f5f8f7;color:#3e514e;font-size:12px;line-height:1.58",
        "v11-source-note": "margin-top:9px;color:#6a7580;font-size:11px;line-height:1.55",
        "v10-actions": "margin-top:10px",
        "v10-more-paper": "padding:13px 0;border-bottom:1px solid #e9edef",
        "v10-more-title": "color:#244f75;font-size:16px;line-height:1.5;font-weight:800;text-decoration:none",
        "v10-more-summary": "margin-top:4px;color:#596775;font-size:13px;line-height:1.62",
        "v10-more-plain": "margin-top:9px;color:#263544;font-size:16px;line-height:1.72;white-space:pre-line",
        "v10-more-tech": "margin-top:8px;color:#596775;font-size:16px;line-height:1.72;white-space:pre-line",
        "footer": "padding:17px 18px;border-top:1px solid #dfe5e8;color:#66717f;font-size:12px;background:#fbfcfc",
    }
    PHYSICAL_AI_TERMS = (
        "physical ai",
        "embodied ai",
        "embodied intelligence",
        "robot",
        "robotics",
        "humanoid",
        "vla",
        "vision-language-action",
        "manipulation",
        "locomotion",
        "warehouse automation",
        "industrial automation",
        "具身",
        "物理ai",
        "物理 ai",
        "机器人",
        "人形机器人",
        "机械臂",
        "操作任务",
        "真实环境",
    )
    WORLD_MODEL_TECH_TERMS = (
        "world model",
        "world models",
        "世界模型",
        "latent dynamics",
        "dynamics model",
        "video prediction",
        "future prediction",
        "predictive model",
        "jepa",
        "joint embedding predictive architecture",
        "latent space",
        "simulation",
        "simulator",
        "planning",
        "rollout",
        "trajectory",
        "仿真",
        "潜空间",
        "动态模型",
        "预测模型",
        "视频预测",
        "长时预测",
        "规划",
        "轨迹",
        "训练",
    )
    AI_LEADER_VIEW_TERMS = (
        "interview",
        "podcast",
        "conversation",
        "talk",
        "opinion",
        "essay",
        "观点",
        "访谈",
        "采访",
        "演讲",
        "yann lecun",
        "lecun",
        "demis hassabis",
        "hassabis",
        "ilya sutskever",
        "sutskever",
        "andrej karpathy",
        "karpathy",
        "sam altman",
        "altman",
        "dario amodei",
        "amodei",
        "jensen huang",
        "huang",
        "fei-fei li",
        "李飞飞",
        "杨立昆",
    )
    SUMMARY_LABEL_PATTERN = re.compile(r"(发生了什么|证据是什么|对谁有影响|下一步看什么)[：:]")
    GENERIC_ANALYSIS_PATTERN = re.compile(r"客户采用和落地数据|后续产品和融资动作|先影响哪一层|形成共识|值得关注|未来可能")
    BAD_PUBLIC_SENTENCE_PATTERN = re.compile(
        r"学习重点是|技术上，它主要围绕|需要回看原文确认|当前摘要还缺少|是否真正|值得持续关注|未来可能"
    )
    METHOD_WORD_PATTERN = re.compile(r"方法|框架|模型|策略|训练|预测|规划|控制|生成|对齐|评测|基准|Transformer|VLA|LiDAR|扩散", re.IGNORECASE)
    RESULT_WORD_PATTERN = re.compile(r"实验|成功率|提升|达到|优于|基线|基准|数据集|参数|复现|开源|\d|%|倍|x", re.IGNORECASE)

    CLASSIC_DESIGN_VERSION = "v7-classic-briefing"
    V8_DESIGN_VERSION = "v8-editorial-reader"
    V9_DESIGN_VERSION = "v9-continuous-learning"
    V10_DESIGN_VERSION = "v10-learning-digest"
    V11_DESIGN_VERSION = "v11-editorial-library"

    def __init__(
        self,
        template_dir: str = "templates",
        design_version: Optional[str] = None,
        report_config: Optional[Dict[str, Any]] = None,
    ):
        os.makedirs(template_dir, exist_ok=True)
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(enabled_extensions=("html", "xml")),
        )
        self.design_version = design_version or self.DESIGN_VERSION
        self.report_config = dict(report_config or {})

    def _is_classic_briefing(self) -> bool:
        return self.design_version == self.CLASSIC_DESIGN_VERSION

    def _is_v8_reader(self) -> bool:
        return self.design_version in {
            self.V8_DESIGN_VERSION,
            self.V9_DESIGN_VERSION,
            self.V10_DESIGN_VERSION,
            self.V11_DESIGN_VERSION,
        }

    def _is_v9_reader(self) -> bool:
        return self.design_version == self.V9_DESIGN_VERSION

    def _is_v10_reader(self) -> bool:
        return self.design_version in {self.V10_DESIGN_VERSION, self.V11_DESIGN_VERSION}

    def _is_v11_product(self) -> bool:
        return str(self.report_config.get("product_mode") or "") == "intelligence_v11_editorial_library"

    def _v11_domain_label(self, item_or_key: Any) -> str:
        key = str(item_or_key if isinstance(item_or_key, str) else self._domain_key(item_or_key))
        return self.V11_DOMAIN_LABELS.get(key, "产品 / 产业")

    @classmethod
    def _inline_v11_critical_styles(cls, html: str) -> str:
        opening_tag = re.compile(
            r"<(?P<tag>body|div|table|h1|h2|h3|article|a)\b(?P<attrs>[^<>]*?)>",
            re.IGNORECASE,
        )

        def apply(match: re.Match[str]) -> str:
            tag = match.group("tag")
            attrs = match.group("attrs") or ""
            styles: List[str] = []
            tag_style = cls.V11_INLINE_TAG_STYLES.get(tag.lower())
            if tag_style:
                styles.append(tag_style)
            class_match = re.search(r'\bclass=["\']([^"\']+)["\']', attrs, re.IGNORECASE)
            if class_match:
                for class_name in class_match.group(1).split():
                    class_style = cls.V11_INLINE_CLASS_STYLES.get(class_name)
                    if class_style:
                        styles.append(class_style)
            if not styles:
                return match.group(0)
            inline_style = ";".join(styles).rstrip(";")
            style_match = re.search(r'\sstyle=(["\'])(.*?)\1', attrs, re.IGNORECASE | re.DOTALL)
            if style_match:
                existing = style_match.group(2).strip().rstrip(";")
                merged = f"{inline_style};{existing}" if existing else inline_style
                attrs = attrs[: style_match.start()] + f' style="{merged}"' + attrs[style_match.end() :]
            else:
                attrs = f'{attrs} style="{inline_style}"'
            return f"<{tag}{attrs}>"

        return opening_tag.sub(apply, html)

    def _score_value(self, item: Dict[str, Any]) -> float:
        return float(item.get("selection_score", item.get("score", 0)) or 0)

    def _sort_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            items,
            key=lambda item: (
                self._score_value(item),
                float(item.get("score", 0) or 0),
                item.get("publish_date", ""),
            ),
            reverse=True,
        )

    def build_mixed_items(self, papers: List[Dict[str, Any]], updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        paper_queue = self._sort_items(papers)
        update_queue = self._sort_items(updates)
        mixed: List[Dict[str, Any]] = []

        last_kind = ""
        same_kind_streak = 0
        pattern_index = 0

        while paper_queue or update_queue:
            desired_kind = self.INTERLEAVE_PATTERN[pattern_index % len(self.INTERLEAVE_PATTERN)]
            forced_switch = last_kind and same_kind_streak >= self.MAX_CONSECUTIVE_SAME_KIND

            next_item = None
            next_kind = ""

            if not forced_switch:
                if desired_kind == "paper" and paper_queue:
                    next_item = paper_queue.pop(0)
                    next_kind = "paper"
                elif desired_kind == "update" and update_queue:
                    next_item = update_queue.pop(0)
                    next_kind = "update"

            if next_item is None:
                if forced_switch and last_kind == "paper" and update_queue:
                    next_item = update_queue.pop(0)
                    next_kind = "update"
                elif forced_switch and last_kind == "update" and paper_queue:
                    next_item = paper_queue.pop(0)
                    next_kind = "paper"
                elif paper_queue and update_queue:
                    top_paper = self._score_value(paper_queue[0])
                    top_update = self._score_value(update_queue[0])
                    if last_kind == "paper":
                        if top_update >= top_paper - 0.35:
                            next_item = update_queue.pop(0)
                            next_kind = "update"
                        else:
                            next_item = paper_queue.pop(0)
                            next_kind = "paper"
                    elif last_kind == "update":
                        if top_paper >= top_update - 0.35:
                            next_item = paper_queue.pop(0)
                            next_kind = "paper"
                        else:
                            next_item = update_queue.pop(0)
                            next_kind = "update"
                    else:
                        if top_update >= top_paper:
                            next_item = update_queue.pop(0)
                            next_kind = "update"
                        else:
                            next_item = paper_queue.pop(0)
                            next_kind = "paper"
                elif update_queue:
                    next_item = update_queue.pop(0)
                    next_kind = "update"
                elif paper_queue:
                    next_item = paper_queue.pop(0)
                    next_kind = "paper"

            if next_item is None:
                break

            mixed.append(next_item)
            if next_kind == last_kind:
                same_kind_streak += 1
            else:
                last_kind = next_kind
                same_kind_streak = 1
            pattern_index += 1

        return mixed

    def _preview_text(self, item: Dict[str, Any]) -> str:
        preview = str(item.get("summary_preview", "") or "").strip()
        if preview:
            return preview

        summary = re.split(r"(?<=[。！？])\s*", str(item.get("summary", "") or "").strip())
        for sentence in summary:
            sentence = sentence.strip()
            if sentence:
                return sentence[:54] + ("..." if len(sentence) > 54 else "")
        return ""

    def _trim_reason(self, text: str, limit: int = 52) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        cleaned = re.sub(r"^[：:，,\-\s]+", "", cleaned)
        cleaned = re.sub(r"[。！？!?.]+$", "", cleaned)
        if not cleaned:
            return ""
        return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip("，、；： ") + "…"

    def _clean_public_sentence(self, text: str, limit: int = 240) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        cleaned = cleaned.replace("技术上，它", "这篇论文")
        cleaned = cleaned.replace("技术上，", "")
        cleaned = cleaned.replace("学习重点是", "关键在于")
        cleaned = cleaned.replace("是否真正", "能不能")
        cleaned = cleaned.replace("需要回看原文确认", "后续应核对")
        cleaned = cleaned.replace("当前摘要还缺少", "材料中暂时缺少")
        cleaned = cleaned.replace("世界模型s", "World Models")
        cleaned = re.sub(r"值得持续关注[^。！？]*[。！？]?", "", cleaned)
        cleaned = re.sub(r"未来可能[^。！？]*[。！？]?", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，；。")
        if not cleaned:
            return ""
        return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip("，、；： ") + "…"

    def _has_bad_public_sentence(self, text: str) -> bool:
        return bool(self.BAD_PUBLIC_SENTENCE_PATTERN.search(str(text or "")))

    def _join_sentences(self, sentences: List[str], limit: int = 260) -> str:
        parts: List[str] = []
        for sentence in sentences:
            cleaned = self._clean_public_sentence(sentence, limit)
            if not cleaned:
                continue
            if not re.search(r"[。！？]$", cleaned):
                cleaned += "。"
            parts.append(cleaned)
        paragraph = "".join(parts)
        return paragraph if len(paragraph) <= limit else paragraph[: limit - 1].rstrip("，、；： ") + "…"

    def _evidence_quality(self, item: Dict[str, Any]) -> float:
        try:
            return float(item.get("evidence_quality", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _information_density(self, item: Dict[str, Any]) -> float:
        try:
            return float(item.get("information_density", item.get("evidence_quality", 0.0)) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _is_high_evidence(self, item: Dict[str, Any]) -> bool:
        return self._evidence_quality(item) >= 0.45 and self._information_density(item) >= 0.45

    def _quality_label(self, item: Dict[str, Any]) -> str:
        evidence = self._evidence_quality(item)
        density = self._information_density(item)
        if evidence >= 0.65 and density >= 0.65:
            return "高证据"
        if evidence >= 0.45 and density >= 0.45:
            return "可采信"
        if evidence >= 0.35 and density >= 0.35:
            return "观察中"
        return "待确认"

    def _source_tier_label(self, item: Dict[str, Any]) -> str:
        tier = str(item.get("source_tier", "") or "").strip().lower()
        content_type = str(item.get("content_type", "") or "").strip().lower()
        if tier == "official":
            return "官方"
        if tier == "research" or content_type == "paper":
            return "论文"
        if tier == "aggregator":
            return "聚合发现"
        if tier == "low_signal":
            return "低信号"
        if tier == "media":
            return "媒体"
        return "来源"

    def _tone_label(self, item: Dict[str, Any]) -> str:
        tier = str(item.get("source_tier", "") or "").strip().lower()
        if tier in {"aggregator", "low_signal"}:
            return "待确认"
        if self._evidence_quality(item) < 0.45:
            return "谨慎看"
        return "可确认"

    def _display_label(self, value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            return "其他"
        mapping = {
            "Product Release": "产品发布",
            "Industry": "行业动态",
            "Application": "应用落地",
            "Partnership": "合作",
            "Infrastructure": "基础设施",
            "Open Source": "开源",
            "Model/Research": "模型研究",
            "Physical AI / Robotics": "Physical AI / Robotics",
            "World Model": "World Model",
            "Social": "社交平台",
            "Video": "视频生成",
            "Other": "其他",
        }
        return mapping.get(raw, raw)

    def _metric_text(self, value: float) -> str:
        return f"{max(0.0, min(1.0, value)):.2f}"

    def _topic_bucket(self, item: Dict[str, Any]) -> str:
        text = " ".join(
            str(item.get(key, "") or "")
            for key in ("display_topic", "topic_cn", "category", "title_cn", "title")
        ).lower()
        if any(token in text for token in ("agent", "代理", "工作流", "workflow")):
            return "Agent"
        if any(token in text for token in ("robot", "机器人", "具身", "physical ai", "vla")):
            return "Robotics"
        if any(token in text for token in ("world model", "世界模型", "视频生成", "planning")):
            return "World Model"
        if any(token in text for token in ("infrastructure", "gpu", "算力", "推理", "芯片", "部署", "成本")):
            return "Infrastructure"
        if any(token in text for token in ("open source", "开源", "github", "license")):
            return "Open Source"
        if str(item.get("content_type", "") or "") == "paper":
            return "Research"
        return "Business"

    def _domain_key(self, item: Dict[str, Any]) -> str:
        if item.get("domain_key"):
            return str(item.get("domain_key"))
        text = self._item_search_text(item)
        if any(token in text for token in ("world model", "world models", "世界模型", "latent dynamics", "jepa", "video prediction", "predictive model", "rollout")):
            return "world_model"
        if self._is_physical_ai_item(item) or any(token in text for token in ("robot", "robotics", "humanoid", "manipulation", "locomotion", "具身", "机器人", "机械臂")):
            return "physical_ai"
        if any(token in text for token in ("agent", "workflow", "reasoning", "llm", "gpt", "claude", "gemini", "模型", "智能体", "工作流")):
            return "agent_models"
        if any(token in text for token in ("gpu", "chip", "inference", "datacenter", "open source", "github", "license", "算力", "芯片", "推理", "开源", "基础设施")):
            return "infra_open_source"
        return "products_business"

    def _domain_label(self, item_or_key: Any) -> str:
        key = str(item_or_key if isinstance(item_or_key, str) else self._domain_key(item_or_key))
        return self.LEARNING_DOMAIN_LABELS.get(key, "Products / Business")

    def _domain_learning_focus(self, domain_key: str) -> str:
        return {
            "world_model": "重点理解预测、潜空间动态、仿真和规划之间的关系。",
            "physical_ai": "重点理解模型如何落到真实机器人任务、动作控制和部署证据。",
            "agent_models": "重点理解模型能力如何转成任务执行、推理流程和工作流自动化。",
            "infra_open_source": "重点理解算力、推理成本、部署栈和开源默认路线的变化。",
            "products_business": "重点理解产品入口、客户采用、商业化和行业资源配置。",
        }.get(domain_key, "重点理解这类内容改变了哪条 AI 学习主线。")

    def _facts_for_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return item.get("facts") if isinstance(item.get("facts"), dict) else {}

    def _public_takeaway_text(self, item: Dict[str, Any], text: str) -> str:
        cleaned = self._clean_public_sentence(text, 82)
        has_long_english = bool(re.search(r"[A-Za-z][A-Za-z0-9./_-]*(?:\s+[A-Za-z][A-Za-z0-9./_-]*){2,}", cleaned))
        if cleaned and not self._mostly_english(cleaned) and not has_long_english:
            return cleaned
        if item.get("content_type") == "paper":
            intro = self._paper_technical_intro(item)
            first_sentence = re.split(r"[。！？]", intro, 1)[0].strip()
            if first_sentence:
                return self._trim_reason(first_sentence, 82)
        return self._public_subject_text(
            cleaned,
            str(item.get("display_topic") or item.get("category") or "相关进展"),
        )

    def _learning_takeaway(self, item: Dict[str, Any]) -> str:
        existing = self._trim_reason(item.get("learning_takeaway", ""), 82)
        if existing:
            return self._public_takeaway_text(item, existing)
        facts = self._facts_for_item(item)
        subject = self._trim_reason(facts.get("who") or item.get("title_cn") or item.get("title"), 26)
        action = self._trim_reason(facts.get("action", ""), 24)
        target = self._trim_reason(facts.get("target", ""), 48)
        if action and target:
            return self._public_takeaway_text(item, f"{subject}{action}{target}")
        return self._public_takeaway_text(item, item.get("summary_preview") or item.get("title_cn") or item.get("title"))

    def _technical_context(self, item: Dict[str, Any]) -> str:
        existing = self._trim_reason(item.get("technical_context", ""), 140)
        if existing:
            return self._clean_public_sentence(existing, 150)
        if item.get("content_type") == "paper":
            return self._paper_technical_intro(item)
        facts = self._facts_for_item(item)
        who = self._trim_reason(facts.get("who") or item.get("title_cn") or item.get("title"), 32)
        target = self._trim_reason(facts.get("target") or item.get("display_topic"), 56)
        evidence = facts.get("evidence") if isinstance(facts, dict) else []
        if isinstance(evidence, str):
            evidence = [evidence]
        evidence_hint = self._trim_reason(str((evidence or [""])[0]), 70)
        text = self._item_search_text(item)
        domain_key = self._domain_key(item)
        if "interview" in text or "访谈" in text or "观点" in text:
            return self._join_sentences(
                [
                    f"{who}这类观点内容不能按发布消息理解，先看它主张的技术路线是{target}",
                    f"判断价值来自论据而不是立场，当前可引用的线索是{evidence_hint}" if evidence_hint else "如果原文没有给出实验或产品证据，只适合放在观点观察里",
                ],
                170,
            )
        category = str(item.get("display_topic") or item.get("category") or "")
        if "产品" in category or domain_key == "products_business":
            return self._join_sentences(
                [
                    f"{who}这条产品信息要看它替代了哪段旧流程，而不只是新增一个功能名",
                    f"目前最具体的信号是{evidence_hint}" if evidence_hint else f"现有材料还没给出客户、价格或使用数据，因此只能先看{target}",
                ],
                170,
            )
        if "合作" in category or "融资" in category:
            return self._join_sentences(
                [
                    f"{who}这类合作或融资信息的重点是资源如何进入{target}",
                    f"如果后续没有客户、交付或产品整合证据，就只能说明资源配置变化",
                ],
                170,
            )
        if "开源" in category or "github" in text or "open source" in text:
            return self._join_sentences(
                [
                    f"{who}的技术价值取决于开发者能不能直接复用{target}",
                    f"优先看许可证、代码活跃度、复现文档和社区采用，而不是只看发布热度",
                ],
                170,
            )
        if domain_key == "world_model":
            return self._join_sentences([f"这条内容值得记住的不是生成效果，而是它把预测结果用于{target}", f"关键验证是预测结果能否进入规划、控制或智能体决策闭环"], 170)
        if domain_key == "physical_ai":
            return self._join_sentences([f"{who}的价值取决于它能不能跨过仿真，进入{target}", f"方法上要看感知、规划和动作控制如何闭环，结果上要看真实机器人任务指标"], 170)
        if domain_key == "agent_models":
            return self._join_sentences([f"{who}这条线索要看模型从回答问题推进到哪类任务执行", f"如果能接入工具、权限、记忆或工作流编排，它才会改变真实使用流程"], 170)
        if domain_key == "infra_open_source":
            return self._join_sentences([f"{who}的核心变量是{target}会不会降低部署成本或切换成本", f"判断时优先看延迟、吞吐、价格、许可证和硬件适配证据"], 170)
        return self._join_sentences([f"{who}这条变化先影响的是{target}", "如果没有客户、实验或代码证据，就不应写成确定趋势"], 150)

    def _technical_digest_points(self, item: Dict[str, Any], limit: int = 4) -> List[str]:
        existing = item.get("technical_digest")
        if isinstance(existing, list):
            points = [self._trim_reason(str(point), 120) for point in existing if str(point).strip()]
            if points:
                return points[:limit]
        facts = self._facts_for_item(item)
        evidence = facts.get("evidence") if isinstance(facts, dict) else []
        if isinstance(evidence, str):
            evidence = [evidence]
        evidence = [self._trim_reason(str(point), 92) for point in (evidence or []) if str(point).strip()]
        target = self._trim_reason(facts.get("target", "") if isinstance(facts, dict) else "", 72)
        action = self._trim_reason(facts.get("action", "") if isinstance(facts, dict) else "", 34)
        title = self._trim_reason(item.get("title_cn") or item.get("title"), 72)
        domain_key = self._domain_key(item)
        text = self._item_search_text(item)
        points: List[str] = []

        if item.get("content_type") == "paper":
            paper_text = " ".join(
                str(value or "")
                for value in (
                    item.get("title_cn"),
                    item.get("summary"),
                    item.get("summary_preview"),
                    facts.get("target", "") if isinstance(facts, dict) else "",
                    " ".join(evidence[:3]),
                )
            ).lower()
            method = self._paper_method_clause(item, facts, self._paper_translate_english_target(target or title), paper_text)
            if "从标题和摘要看" in method:
                short_name = self._paper_short_name(item, facts)
                if "policy evaluation correlation" in paper_text or "real-world success rate" in paper_text:
                    method = f"{short_name}用世界模型预测策略表现，并用真实机器人成功率相关性校验评估是否可信"
                elif "benchmark" in paper_text or "dataset" in paper_text:
                    method = f"{short_name}把任务拆成可比较的评测单元，用统一基准衡量模型差异"
                else:
                    method = f"{short_name}围绕{self._trim_reason(target or title, 46)}梳理新的模型或训练流程"
            if method:
                points.append(f"方法路线：{method}。")
            evidence_bits = [self._paper_translate_english_evidence(point) for point in evidence[:2]]
            evidence_bits = [bit for bit in evidence_bits if bit]
            if evidence_bits:
                points.append(f"实验信号：{evidence_bits[0]}。")
            if len(evidence_bits) > 1:
                points.append(f"对照依据：{evidence_bits[1]}。")
            if domain_key == "world_model":
                points.append("技术方向：关注预测模块能不能接入规划、控制或智能体决策，而不是只提升生成质量。")
            elif domain_key == "physical_ai":
                points.append("技术方向：关注仿真训练、VLA策略或控制模块能否迁移到真实机器人任务。")
            elif domain_key == "agent_models":
                points.append("技术方向：关注模型是否从单次推理扩展到工具调用、状态记忆和多步任务执行。")
            else:
                points.append("技术方向：优先看模型结构、训练数据、评测基准和复现材料是否支撑论文结论。")
        else:
            if action and target:
                points.append(f"路线变化：{action}{target}，需要看它具体改变的是模型能力、系统接口还是部署流程。")
            elif target:
                points.append(f"路线变化：围绕{target}展开，重点看技术机制是否已经被证据支撑。")
            else:
                points.append(f"路线变化：{title}。")
            if evidence:
                points.append(f"证据摘编：{evidence[0]}。")
            if domain_key == "world_model":
                points.append("技术看点：是否把潜空间动态、视频预测或仿真 rollout 变成可用于规划的中间表示。")
            elif domain_key == "physical_ai":
                points.append("技术看点：是否形成感知、规划、动作控制到真实部署的闭环，而不是只停留在演示。")
            elif domain_key == "agent_models":
                points.append("技术看点：是否把模型推理接到工具、权限、记忆和工作流编排。")
            elif domain_key == "infra_open_source":
                points.append("技术看点：是否改变推理成本、延迟、部署栈、许可证或开发者默认选择。")
            else:
                points.append("技术看点：区分产品包装、真实能力边界和可持续的商业化路径。")
            if "interview" in text or "访谈" in text or "观点" in text:
                points.append("观点摘编：把受访者的技术路线偏好和已经发生的事实分开看，避免把判断当成结论。")

        unique: List[str] = []
        seen = set()
        for point in points:
            cleaned = re.sub(r"\s+", " ", point).strip()
            cleaned = self._trim_reason(cleaned, 122)
            key = cleaned[:42]
            if cleaned and key not in seen:
                unique.append(cleaned)
                seen.add(key)
            if len(unique) >= limit:
                break
        return unique

    def _public_evidence_point(self, item: Dict[str, Any], point: Any) -> str:
        raw = re.sub(r"\s+", " ", str(point or "")).strip()
        if not raw:
            return ""
        if item.get("content_type") == "paper":
            return self._paper_translate_english_evidence(raw)
        if not self._mostly_english(raw):
            return self._trim_reason(raw, 90)
        lowered = raw.lower()
        numbers = re.findall(r"\d[\d,]*(?:\.\d+)?\s?(?:%|x|倍|m|b|k)?", raw, re.IGNORECASE)
        number_hint = numbers[0].strip() if numbers else ""
        if "released" in lowered or "launch" in lowered:
            return self._trim_reason("来源披露了新发布或新上线内容", 90)
        if "multitasking" in lowered or "parental control" in lowered:
            return "新增多任务、家长控制等系统功能"
        if "api" in lowered and ("safeguard" in lowered or "guardrail" in lowered):
            return "API支持按场景应用安全防护"
        if "agentic ai" in lowered and "application" in lowered:
            return "可在智能体应用的不同环节接入"
        if "real conversation data" in lowered:
            return "使用真实对话数据验证"
        if "predict" in lowered and "behavior" in lowered:
            return "方法用于在部署前预测模型行为"
        if "blackwell" in lowered and ("performance" in lowered or "result" in lowered):
            return f"Blackwell在基准中给出性能结果{number_hint}" if number_hint else "Blackwell在基准中给出性能结果"
        if "up to" in lowered and number_hint:
            return f"最高达到{number_hint}的性能或效率提升"
        if "prototype" in lowered:
            return "正在构建AI原型验证应用场景"
        if "house-building" in lowered or "housing" in lowered:
            return "目标是推动住房建设相关流程"
        if "customer" in lowered or "client" in lowered:
            return "材料提到客户或使用场景"
        if number_hint:
            return f"材料给出{number_hint}这一量化信号"
        return "英文材料给出功能或场景线索，建议打开原文核对"

    def _background_context(self, item: Dict[str, Any]) -> str:
        existing = self._trim_reason(item.get("background_context", ""), 128)
        if existing:
            existing = re.sub(r"^事实依据[：:]\s*", "", existing).strip()
            existing_points = [part for part in re.split(r"[；;]", existing) if part.strip()]
            translated = [self._public_evidence_point(item, part) for part in existing_points[:2]]
            translated = [part for part in translated if part]
            return "；".join(translated) if translated else existing
        points = item.get("evidence_points") or []
        if points:
            return self._trim_reason("；".join(str(point) for point in points[:2]), 128)
        if self._evidence_quality(item) < 0.35:
            return "当前只能作为待确认线索。"
        return "来自标题、来源和正文摘要，仍建议点开原文核对细节。"

    def _deep_dive_prompt(self, item: Dict[str, Any]) -> str:
        existing = self._trim_reason(item.get("deep_dive_prompt", ""), 118)
        if existing:
            return re.sub(r"^继续深挖[：:]\s*", "", existing).strip()
        if item.get("content_type") == "paper":
            return self._paper_deep_dive_hint(item)
        text = self._item_search_text(item)
        if "interview" in text or "访谈" in text or "观点" in text:
            return "优先看原访谈上下文，区分事实、判断和个人路线偏好。"
        if "github" in text or "open source" in text or "开源" in text:
            return "看许可证、代码活跃度、复现文档和社区采用。"
        return "看原文里的数字、客户、实验或产品细节是否支撑结论。"

    def _editorial_brief(self, item: Dict[str, Any]) -> str:
        facts = self._facts_for_item(item)
        who = self._trim_reason(facts.get("who") or item.get("title_cn") or item.get("title"), 34)
        action = self._trim_reason(facts.get("action") or "披露", 16)
        target = self._trim_reason(facts.get("target") or item.get("display_topic") or "相关方向", 62)
        evidence = facts.get("evidence") if isinstance(facts, dict) else []
        if isinstance(evidence, str):
            evidence = [evidence]
        who = self._public_subject_text(who, "该来源")
        target = self._public_subject_text(target, str(item.get("display_topic") or "相关方向"))
        evidence_hint = self._public_evidence_point(item, (evidence or [""])[0])
        evidence_hint = self._trim_reason(evidence_hint, 74)
        category = str(item.get("display_topic") or item.get("category") or "")
        text = self._item_search_text(item)
        if item.get("content_type") == "paper":
            return self._paper_technical_intro(item)
        if self._evidence_quality(item) < 0.35:
            return self._join_sentences([f"来源称{self._fact_phrase(who, action, target)}", "目前缺少实验、客户或代码证据，因此只适合作为待确认线索"], 160)
        if "interview" in text or "访谈" in text or "观点" in text:
            return self._join_sentences([f"{who}的核心主张是{target}", f"这类内容要把观点和论据拆开看，当前论据是{evidence_hint}" if evidence_hint else "材料没有给出强实验或产品证据"], 190)
        if "产品" in category or self._domain_key(item) == "products_business":
            return self._join_sentences([self._fact_phrase(who, action, target), "这条信息的重点不是功能名，而是它可能替代哪段旧流程", f"当前证据是{evidence_hint}" if evidence_hint else "还需要客户、价格或真实使用数据支撑"], 190)
        if "开源" in category or "github" in text or "open source" in text:
            return self._join_sentences([self._fact_phrase(who, action, target), "判断它的技术价值，要看许可证、复现文档和社区维护能否让开发者直接复用"], 180)
        if "基础设施" in category or self._domain_key(item) == "infra_open_source":
            return self._join_sentences([self._fact_phrase(who, action, target), "基础设施类内容要先看成本、延迟、吞吐或部署兼容性，而不是只看发布动作"], 180)
        if "合作" in category or "融资" in category:
            return self._join_sentences([self._fact_phrase(who, action, target), "这类内容要看资源是否变成客户、渠道、算力或产品整合，而不是只看合作标题"], 180)
        return self._join_sentences([self._fact_phrase(who, action, target), self._technical_context(item)], 190)

    def _fact_phrase(self, who: str, action: str, target: str) -> str:
        action = action or "披露"
        separator = " " if re.search(r"[A-Za-z0-9]$", who or "") and re.search(r"^[A-Za-z0-9]", target or "") else ""
        return f"{who}{action}{separator}{target}".strip()

    def _public_subject_text(self, value: str, fallback: str) -> str:
        cleaned = self._clean_public_sentence(value, 72)
        if not cleaned:
            return fallback
        if not self._mostly_english(cleaned):
            return cleaned
        if len(re.findall(r"[A-Za-z][A-Za-z0-9.&-]*", cleaned)) <= 3 and len(cleaned) <= 28:
            return cleaned
        lowered = cleaned.lower()
        if "hermes agent skills" in lowered:
            return "Hermes Agent Skills 的 /learn 能力" if "/learn" in lowered else "Hermes Agent Skills"
        if "android" in lowered or "wear os" in lowered:
            return "Android 和 Wear OS 系统更新"
        if "blackwell" in lowered:
            return "Blackwell 基准性能"
        if "agent" in lowered or "workflow" in lowered:
            return "智能体工作流"
        if "robot" in lowered or "robotics" in lowered:
            return "机器人相关进展"
        return fallback

    def _research_group_label(self, item: Dict[str, Any]) -> str:
        text = " ".join(
            str(item.get(key, "") or "")
            for key in ("display_topic", "topic_cn", "category", "title_cn", "title")
        ).lower()
        if any(token in text for token in ("physical ai", "具身", "vla", "embodied")):
            return "Physical AI"
        if any(token in text for token in ("world model", "世界模型", "planning")):
            return "World Model"
        if any(token in text for token in ("robot", "机器人", "locomotion", "grasp")):
            return "Robotics"
        return "模型研究"

    def _is_physical_ai_item(self, item: Dict[str, Any]) -> bool:
        facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
        evidence = facts.get("evidence") if isinstance(facts, dict) else []
        if isinstance(evidence, str):
            evidence = [evidence]
        evidence = evidence or []
        text = " ".join(
            str(part or "")
            for part in (
                item.get("title_cn", ""),
                item.get("title", ""),
                item.get("summary_preview", ""),
                item.get("summary", ""),
                item.get("display_topic", ""),
                item.get("topic_cn", ""),
                item.get("category", ""),
                facts.get("who", "") if isinstance(facts, dict) else "",
                facts.get("target", "") if isinstance(facts, dict) else "",
                " ".join(str(point) for point in (evidence or [])),
            )
        ).lower()
        return any(term in text for term in self.PHYSICAL_AI_TERMS)

    def _build_dashboard(
        self,
        layers: Dict[str, List[Dict[str, Any]]],
        report_summary: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        visible_keys = ("must_read", "physical_ai", "watch", "featured_papers", "paper_appendix", "brief")
        all_items = [item for key in visible_keys for item in layers.get(key, [])]
        if not layers.get("featured_papers") and layers.get("research"):
            all_items.extend(layers.get("research", []))
        low_evidence_count = sum(1 for item in all_items if self._evidence_quality(item) < 0.35)
        quality_status = "通过" if not low_evidence_count else "已分层"
        return [
            {"label": "必读", "value": str(len(layers.get("must_read", []))), "hint": "高证据、高密度内容"},
            {"label": "具身", "value": str(len(layers.get("physical_ai", []))), "hint": "Physical AI 与机器人落地动态"},
            {
                "label": "论文",
                "value": str(len(layers.get("featured_papers", [])) + len(layers.get("paper_appendix", []))),
                "hint": "精选论文与附录覆盖",
            },
            {"label": "质量", "value": quality_status, "hint": str(report_summary.get("lead_summary", "") or "")[:34]},
        ]

    def _decision_entry(
        self,
        layers: Dict[str, List[Dict[str, Any]]],
        report_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        focus_items = (
            layers.get("must_read", [])
            + layers.get("physical_ai", [])
            + layers.get("featured_papers", [])
        )
        summary_lines = self._summary_lines(report_summary, limit=1)
        remember = (
            summary_lines[0]
            if summary_lines
            else self._editor_judgement(focus_items + layers.get("watch", []) + layers.get("brief", []))
        )

        open_first: List[Dict[str, str]] = []
        seen_urls = set()
        for item in self._sort_items(focus_items):
            title = str(item.get("title_cn") or item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title or not url or url in seen_urls:
                continue
            open_first.append(
                {
                    "title": self._trim_reason(title, 52),
                    "url": url,
                    "label": str(item.get("report_section_label") or item.get("display_topic") or ""),
                }
            )
            seen_urls.add(url)
            if len(open_first) >= 3:
                break

        brief_count = len(layers.get("brief", []))
        if brief_count:
            skip_note = f"{brief_count} 条低证据或短讯已放入快讯区，最后扫一眼即可。"
        else:
            skip_note = "本期低证据内容较少，可以直接读重点区和论文区。"
        return {"remember": remember, "open_first": open_first, "skip_note": skip_note}

    def _physical_ai_radar_label(self, item: Dict[str, Any]) -> str:
        facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
        evidence = facts.get("evidence") if isinstance(facts, dict) else []
        if isinstance(evidence, str):
            evidence = [evidence]
        text = " ".join(
            str(part or "")
            for part in (
                item.get("title_cn", ""),
                item.get("title", ""),
                item.get("summary_preview", ""),
                item.get("summary", ""),
                item.get("display_topic", ""),
                item.get("topic_cn", ""),
                item.get("category", ""),
                facts.get("who", "") if isinstance(facts, dict) else "",
                facts.get("target", "") if isinstance(facts, dict) else "",
                " ".join(str(point) for point in (evidence or [])),
            )
        ).lower()
        if any(token in text for token in ("humanoid", "人形", "whole-body", "全身控制")):
            return "人形机器人"
        if any(token in text for token in ("vla", "vision-language-action", "manipulation", "grasp", "机械臂", "抓取", "操作")):
            return "VLA / 操作模型"
        if any(token in text for token in ("warehouse", "industrial", "manufacturing", "logistics", "仓储", "工业", "制造", "物流")):
            return "工业 / 仓储"
        if any(token in text for token in ("deployment", "customer", "real-world", "pilot", "部署", "客户", "真实", "试点", "落地")):
            return "真实部署"
        return "其他动态"

    def _physical_ai_radar_groups(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in items:
            grouped.setdefault(self._physical_ai_radar_label(item), []).append(item)
        result: List[Dict[str, Any]] = []
        for label in ("真实部署", "人形机器人", "VLA / 操作模型", "工业 / 仓储", "其他动态"):
            group_items = self._sort_items(grouped.get(label, []))
            if not group_items:
                continue
            result.append({"label": label, "articles": group_items})
        return result

    def _item_search_text(self, item: Dict[str, Any]) -> str:
        facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
        evidence = facts.get("evidence") if isinstance(facts, dict) else []
        if isinstance(evidence, str):
            evidence = [evidence]
        return " ".join(
            str(part or "")
            for part in (
                item.get("title_cn", ""),
                item.get("title", ""),
                item.get("summary_preview", ""),
                item.get("summary", ""),
                item.get("content", ""),
                item.get("display_topic", ""),
                item.get("topic_cn", ""),
                item.get("category", ""),
                item.get("source_detail", ""),
                item.get("platform", ""),
                facts.get("who", "") if isinstance(facts, dict) else "",
                facts.get("action", "") if isinstance(facts, dict) else "",
                facts.get("target", "") if isinstance(facts, dict) else "",
                " ".join(str(point) for point in (evidence or [])),
            )
        ).lower()

    def _is_world_model_focus_item(self, item: Dict[str, Any]) -> bool:
        text = self._item_search_text(item)
        return any(term in text for term in self.WORLD_MODEL_TECH_TERMS)

    def _world_model_focus_label(self, item: Dict[str, Any]) -> str:
        text = self._item_search_text(item)
        if any(term in text for term in self.AI_LEADER_VIEW_TERMS):
            return "大佬访谈 / 观点"
        if any(term in text for term in ("training", "训练", "jepa", "latent", "潜空间", "dynamics", "动态模型", "rollout", "trajectory", "轨迹")):
            return "训练技术"
        return "文章 / 长文"

    def _world_model_focus_groups(self, items: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        seen_urls = set()
        for item in self._sort_items(items):
            url = str(item.get("url") or "").strip()
            if not url or url in seen_urls or not self._is_world_model_focus_item(item):
                continue
            grouped.setdefault(self._world_model_focus_label(item), []).append(item)
            seen_urls.add(url)
            if len(seen_urls) >= limit:
                break
        result: List[Dict[str, Any]] = []
        for label in ("训练技术", "大佬访谈 / 观点", "文章 / 长文"):
            group_items = grouped.get(label, [])
            if group_items:
                result.append({"label": label, "articles": group_items})
        return result

    def _summary_lines(self, report_summary: Dict[str, Any], limit: int = 3) -> List[str]:
        lead = re.sub(r"\s+", " ", str(report_summary.get("lead_summary", "") or "")).strip()
        if not lead:
            return []
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[。！？!?])\s*", lead) if sentence.strip()]
        if not sentences:
            sentences = [lead]
        return sentences[:limit]

    def _editor_judgement(self, items: List[Dict[str, Any]]) -> str:
        high_items = [item for item in items if self._is_high_evidence(item)]
        buckets: Dict[str, int] = {}
        for item in high_items[:10]:
            bucket = self._topic_bucket(item)
            buckets[bucket] = buckets.get(bucket, 0) + 1
        if not buckets:
            return "今天的重要性不在于单条新闻数量，而在于低证据内容已被压缩，重点区只保留能说清事实、证据和影响对象的变化。"
        leaders = sorted(buckets.items(), key=lambda pair: pair[1], reverse=True)[:2]
        names = "、".join(label for label, _ in leaders)
        return f"今天的重要性集中在 {names}：重点条目已经按证据和信息密度筛过，更适合先判断方向，再决定是否深读原文。"

    def _research_value_tags(self, item: Dict[str, Any]) -> List[str]:
        evidence = self._evidence_quality(item)
        density = self._information_density(item)
        text = " ".join(str(item.get(key, "") or "") for key in ("title_cn", "summary", "summary_preview", "category")).lower()
        reproducibility = "复现价值高" if density >= 0.65 or any(token in text for token in ("benchmark", "dataset", "github", "code", "开源", "数据集")) else "复现价值中"
        novelty = "方法新意高" if any(token in text for token in ("new", "novel", "propose", "提出", "framework", "architecture", "方法")) or evidence >= 0.7 else "方法新意中"
        if any(token in text for token in ("robot", "机器人", "deployment", "industry", "agent", "workflow", "应用")):
            distance = "应用距离近"
        elif evidence >= 0.55 and density >= 0.55:
            distance = "应用距离中"
        else:
            distance = "应用距离远"
        return [reproducibility, novelty, distance]

    def _mostly_english(self, text: str) -> bool:
        letters = len(re.findall(r"[A-Za-z]", text or ""))
        cjk = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
        return (letters >= 14 and letters > cjk * 2) or (letters >= 5 and cjk == 0)

    def _chinese_enough_for_paper_description(self, text: str) -> bool:
        letters = len(re.findall(r"[A-Za-z]", text or ""))
        cjk = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
        return cjk >= 30 and cjk >= max(letters // 2, 12)

    def _useful_plain_sentence(self, text: str) -> bool:
        cjk = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
        return bool(text) and cjk >= 8 and not self._mostly_english(text)

    def _paper_description_is_substantive(self, text: str) -> bool:
        normalized = re.sub(r"\s+", "", text or "")
        if any(phrase in text for phrase in ("白话总结", "可以把它当作", "论文的证据主要藏", "重点不在概念名", "关键看", "核心看", "深入时", "继续读原文", "论文来源", "适合谁读")):
            return False
        method_ok = bool(re.search(r"方法|框架|模型|策略|训练|预测|规划|控制|生成|对齐|强化学习|Transformer|VLA|LiDAR|扩散", text or "", re.IGNORECASE))
        result_ok = bool(re.search(r"实验|成功率|提升|达到|优于|基准|数据集|参数|复现|开源|\d|%|倍|x", text or "", re.IGNORECASE))
        meaning_ok = bool(re.search(r"说明|体现|意味着|意义|价值|影响|决定|主要影响|真实任务|真实机器人|部署|泛化|试错成本|可复现|稳定", text or ""))
        has_repeated_template = normalized.count("它的阅读重点") > 1 or normalized.count("继续读原文") > 1
        return method_ok and result_ok and meaning_ok and not has_repeated_template and self._chinese_enough_for_paper_description(text)

    def _paper_translate_english_target(self, text: str) -> str:
        lowered = (text or "").lower()
        if "predict-then-act" in lowered or "motion-aware latent world model" in lowered:
            return "用运动感知世界模型先预测物体未来位置，再让VLA执行动作"
        if "whole-body control" in lowered or "humanoid" in lowered:
            return "用大规模动作数据训练人形机器人的全身控制模型"
        if "lidar" in lowered and ("4d" in lowered or "scene" in lowered):
            return "按空间不确定性生成更可信的4D LiDAR场景"
        if "uav" in lowered or "aerial navigation" in lowered:
            return "让无人机用智能体自动设计奖励并改进导航策略"
        if "human-in-the-loop" in lowered or "preference-calibrated" in lowered:
            return "用人类干预产生的偏好信号重新分配机器人强化学习里的奖励信用"
        if "navigation" in lowered and ("function" in lowered or "motion planning" in lowered):
            return "把学习型导航目标嵌入结构化规划器，提升未见环境里的运动规划"
        if "speech-and-motion" in lowered or "dyadic interaction" in lowered:
            return "把流式语音理解和身体动作生成放进同一个双人交互模型"
        if "world model" in lowered and ("agent" in lowered or "policy" in lowered):
            return "让文本世界模型和智能体策略在交互中一起更新"
        if "affordance" in lowered:
            return "让机器人同时判断在哪里交互以及交互后该怎样运动"
        if "benchmark" in lowered or "dataset" in lowered:
            return "建立新的数据或评测方式来比较后续模型"
        if text and not self._mostly_english(text):
            return self._trim_reason(text, 70)
        return "从标题和摘要看，论文在改进模型或训练流程；当前材料不足以展开更多细节"

    def _paper_translate_english_evidence(self, text: str) -> str:
        raw = str(text or "").strip()
        lowered = raw.lower()
        numbers = re.findall(r"\d[\d,]*(?:\.\d+)?\s?(?:%|x|m|b|k|小时|hour|hours|frame|frames|参数|parameters)?", raw, re.IGNORECASE)
        number_hint = numbers[0].strip() if numbers else ""
        if "consistent gains" in lowered and any(token in lowered for token in ("libero", "robocasa", "robotwin", "real-robot")):
            return "在 LIBERO、RoboCasa24、RoboTwin2.0 和真机设置中都报告增益"
        if "largest improvements" in lowered and "spatial" in lowered:
            return "空间和物体敏感任务的提升最大"
        if "validated on" in lowered and "gr00t" in lowered:
            return "在多种 VLA 基座上完成验证"
        if "pretrained vla" in lowered and "geometric module" in lowered:
            return "没有几何模块的预训练 VLA 基线"
        if "4.9m" in lowered and "parameter" in lowered:
            return "只给冻结的7B OpenVLA额外增加4.9M参数"
        if "79 to 97" in lowered:
            return "20个动态仿真场景成功率达到79%-97%"
        if "31 to 58" in lowered:
            return "最强基线成功率只有31%-58%"
        if "policy evaluation correlation" in lowered:
            return f"策略评估与真实成功率相关性达到{number_hint}" if number_hint else "策略评估与真实成功率有相关性"
        if "policy improvement" in lowered and "real-world success" in lowered:
            return f"真实任务成功率提升{number_hint}" if number_hint else "真实任务成功率有提升"
        if "success" in lowered and "%" in raw:
            return f"报告了{number_hint or '明确'}成功率指标"
        if "2b-frame" in lowered or "2b frame" in lowered:
            return "使用2B帧动作语料进行预训练"
        if "4,000-hour" in lowered or "4000-hour" in lowered or "4000 hour" in lowered:
            return "使用4000小时交互数据训练"
        if "5x" in lowered or "\\times" in lowered or "times improvement" in lowered:
            return "相对部分学习型规划器最高提升5倍"
        if "16.75" in lowered:
            return "在多个智能体基准上相对基线提升16.75%"
        if "relative improvement" in lowered and number_hint:
            return f"在对比基准上相对提升{number_hint}"
        if "24.5" in lowered:
            return "五项真实机器人操作任务平均成功率提升24.5%"
        if "1.3" in lowered and "convergence" in lowered:
            return "收敛速度提升到1.3倍"
        if "43.5" in lowered:
            return "并行仿真和推理把 rollout 时间降低43.5%"
        if "29/30" in lowered or "30/30" in lowered:
            return "真实机械臂动态任务达到29/30到30/30成功"
        if "perception constraints" in lowered and "uav" in lowered:
            return "使用无人机动态导航中的感知约束"
        if "velocity-triggered" in lowered:
            return "通过速度触发机制平衡安全与任务推进"
        if "scanpath" in lowered and number_hint:
            return f"扫描路径预测指标为{number_hint}"
        if ("成功率" in raw or "success rate" in lowered) and number_hint:
            return f"成功率提升或达到{number_hint}"
        if "zero-shot" in lowered:
            return "展示了未见任务或未见环境的零样本泛化"
        if "code available" in lowered or "github" in lowered:
            return "作者公开了代码，便于复现检查"
        if "benchmark" in lowered or "dataset" in lowered:
            return f"在{number_hint + ' ' if number_hint else ''}数据集或基准上验证"
        if number_hint:
            return f"材料给出{number_hint}这一量化结果"
        return self._trim_reason(raw, 60) if raw and not self._mostly_english(raw) else "论文给出了实验、基准或真实任务验证"

    def _paper_short_name(self, item: Dict[str, Any], facts: Dict[str, Any]) -> str:
        who = str(facts.get("who") or item.get("title_cn") or item.get("title") or "这篇论文").strip()
        if who.lower() in {"this paper", "researchers", "unknown"}:
            who = str(item.get("title_cn") or item.get("title") or "这篇论文").strip()
        who = re.split(r"\s*\(|：|:|，|,", who, 1)[0].strip()
        if self._mostly_english(who) and len(re.findall(r"[A-Za-z][A-Za-z0-9./_-]*", who)) >= 4:
            who = "这篇论文"
        return self._trim_reason(who, 36) or "这篇论文"

    def _paper_method_sentence(self, item: Dict[str, Any], facts: Dict[str, Any], method: str, paper_text: str) -> str:
        who = self._paper_short_name(item, facts)
        if any(token in paper_text for token in ("predict-then-act", "motion-aware", "dynamic manipulation", "moving objects")):
            return f"{who}关注的是动态物体操作中的延迟问题：机器人不再只根据当前画面出动作，而是{method}。"
        if any(token in paper_text for token in ("whole-body", "humanoid", "motion tracking")):
            return f"{who}把重点放在人形机器人的全身控制上：它用更大规模的动作数据训练一个能覆盖更多动作和任务的控制模型。"
        if any(token in paper_text for token in ("uav", "aerial navigation", "reward design")):
            return f"{who}处理的是无人机导航里的奖励设计和策略修正问题：系统会{method}。"
        if any(token in paper_text for token in ("human-in-the-loop", "preference-calibrated", "intervention")):
            return f"{who}盯住的是人类介入后的奖励分配问题：它把干预信号转成偏好，用来校准机器人策略训练。"
        if any(token in paper_text for token in ("lidar", "4d world", "scene synthesis")):
            return f"{who}研究的是4D场景生成：它不再平均处理每个空间区域，而是{method}。"
        if any(token in paper_text for token in ("speech-and-motion", "dyadic interaction", "full-duplex")):
            return f"{who}把语音和动作看成同一个实时交互过程：模型一边理解对话，一边生成同步的身体运动。"
        if any(token in paper_text for token in ("textual world model", "agent policies", "future-aware")):
            return f"{who}处理的是智能体行动前如何预判环境：它让文本世界模型和智能体策略在交互轨迹中一起更新。"
        if any(token in paper_text for token in ("navigation function", "motion planning", "zero-shot")):
            return f"{who}研究的是未见环境里的运动规划：它把学习目标放进结构化规划器，让路径仍保持可解释的约束。"
        return f"{who}这篇论文的核心做法是{method}。"

    def _paper_method_clause(self, item: Dict[str, Any], facts: Dict[str, Any], method: str, paper_text: str) -> str:
        who = self._paper_short_name(item, facts)
        if any(token in paper_text for token in ("predict-then-act", "motion-aware", "dynamic manipulation", "moving objects")):
            return f"{who}用运动感知世界模型预判动态物体位置，再让VLA执行动作"
        if any(token in paper_text for token in ("whole-body", "humanoid", "motion tracking")):
            return f"{who}用大规模动作数据训练人形机器人的全身控制模型"
        if any(token in paper_text for token in ("uav", "aerial navigation", "reward design")):
            return f"{who}让无人机用智能体自动设计奖励并改进导航策略"
        if any(token in paper_text for token in ("human-in-the-loop", "preference-calibrated", "intervention")):
            return f"{who}把人类干预信号转成偏好，用来校准机器人策略训练"
        if any(token in paper_text for token in ("lidar", "4d world", "scene synthesis")):
            return f"{who}按空间不确定性生成更可信的4D LiDAR场景"
        if any(token in paper_text for token in ("speech-and-motion", "dyadic interaction", "full-duplex")):
            return f"{who}把流式语音理解和身体动作生成放进同一个交互模型"
        if any(token in paper_text for token in ("textual world model", "agent policies", "future-aware")):
            return f"{who}让文本世界模型和智能体策略在交互中一起更新"
        if any(token in paper_text for token in ("navigation function", "motion planning", "zero-shot")):
            return f"{who}把学习型导航目标嵌入结构化规划器"
        return f"{who}聚焦{self._trim_reason(method, 42)}"

    def _paper_core_watch_clause(self, item: Dict[str, Any], paper_text: str) -> str:
        if any(token in paper_text for token in ("uav", "aerial navigation", "reward design")):
            return "核心看真实飞行中的成功率和故障模式"
        if any(token in paper_text for token in ("human-in-the-loop", "preference-calibrated", "intervention")):
            return "核心看能否稳定提升更多真实操作任务"
        if any(token in paper_text for token in ("whole-body", "humanoid", "motion tracking")):
            return "核心看未见动作和长时间运行是否稳定"
        if any(token in paper_text for token in ("vla", "vision-language-action", "manipulation", "grasp", "动态物体操作")):
            return "核心看能否迁移到新物体、新指令和新场景"
        if any(token in paper_text for token in ("world model", "future prediction", "planning", "导航", "规划", "预测")):
            return "核心看是否真能降低规划和控制里的试错成本"
        if any(token in paper_text for token in ("dataset", "benchmark", "数据集", "基准", "leaderboard")):
            return "核心看数据覆盖和评测设计是否足够可靠"
        topic = self._research_group_label(item)
        if topic == "World Model":
            return "核心看评估任务是否接近真实决策"
        if topic in {"Physical AI", "Robotics"}:
            return "核心看实机复现、失败案例和跨场景表现"
        return "核心看对照实验、指标和复现条件是否扎实"

    def _paper_memory_point(self, item: Dict[str, Any]) -> str:
        facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
        title = str(item.get("title_cn") or item.get("title") or "").strip()
        target = str(facts.get("target") or "").strip()
        evidence = facts.get("evidence") if isinstance(facts, dict) else []
        if isinstance(evidence, str):
            evidence = [evidence]
        paper_text = " ".join([title, target, " ".join(str(point) for point in (evidence or []))]).lower()
        name = self._paper_short_name(item, facts)
        if any(token in paper_text for token in ("predict-then-act", "motion-aware", "dynamic manipulation", "moving objects")):
            concept = "给VLA加一个预测未来的动作前视镜"
        elif any(token in paper_text for token in ("whole-body", "humanoid", "motion tracking")):
            concept = "用大规模动作语料训练更通用的人形机器人控制器"
        elif any(token in paper_text for token in ("uav", "aerial navigation", "reward design")):
            concept = "让智能体自动设计奖励来训练无人机导航"
        elif any(token in paper_text for token in ("human-in-the-loop", "preference-calibrated", "intervention")):
            concept = "把人类接管动作变成机器人学习的偏好信号"
        elif any(token in paper_text for token in ("lidar", "4d world", "scene synthesis")):
            concept = "用不确定性决定4D LiDAR场景哪里该重点生成"
        elif any(token in paper_text for token in ("speech-and-motion", "dyadic interaction", "full-duplex")):
            concept = "把实时语音和身体动作合成到同一个交互模型"
        elif any(token in paper_text for token in ("textual world model", "agent policies", "future-aware")):
            concept = "让世界模型和智能体策略互相迭代"
        elif any(token in paper_text for token in ("navigation function", "motion planning", "zero-shot")):
            concept = "把学习目标放进有约束的运动规划器"
        elif any(token in paper_text for token in ("affordance", "functional mask", "post-contact motion")):
            concept = "让机器人同时知道在哪里动手和怎么继续动"
        else:
            method = self._paper_translate_english_target(target or title)
            concept = self._trim_reason(method, 34)
        return f"{name}：{concept}"

    def _paper_meaning_sentence(self, item: Dict[str, Any], paper_text: str) -> str:
        topic = self._research_group_label(item)
        if any(token in paper_text for token in ("uav", "aerial navigation", "reward design")):
            return "这说明无人机策略训练可以减少对人工奖励设计和反复调参的依赖，论文价值主要体现在真实飞行成功率和故障模式是否被系统验证。"
        if any(token in paper_text for token in ("human-in-the-loop", "preference-calibrated", "intervention")):
            return "这把人类干预从临时纠错变成训练信号，说明偏好校准有机会稳定提升更多真实操作任务。"
        if any(token in paper_text for token in ("whole-body", "humanoid", "motion tracking")):
            return "这说明大规模动作数据可能支撑更通用的人形机器人控制，论文价值体现在未见动作、复杂动作和长时间运行是否稳定。"
        if any(token in paper_text for token in ("vla", "vision-language-action", "manipulation", "grasp", "动态物体操作")):
            return "这说明机器人操作正在从静态演示走向更接近真实环境的任务，论文价值体现在能否迁移到新物体、新指令和新场景。"
        if any(token in paper_text for token in ("world model", "future prediction", "planning", "导航", "规划", "预测")):
            return "这把预测未来状态变成规划依据，说明世界模型的价值不只是生成画面，而是降低导航、控制或工具使用里的试错成本。"
        if any(token in paper_text for token in ("control", "locomotion", "whole-body", "力控", "控制", "运动", "四足", "腿式", "humanoid")):
            return "它的意义在于验证控制策略能否承受扰动、地形变化和长时间运行，这决定它离真实机器人部署还有多远。"
        if any(token in paper_text for token in ("dataset", "benchmark", "数据集", "基准", "leaderboard")):
            return "这类工作给后续研究提供可比较的公共参照，论文价值取决于数据覆盖、评测设计和失败案例是否完整。"
        if topic == "Physical AI":
            return "它的意义在于把模型输出和真实任务结果连起来，后续要看第三方复现或真实环境验证是否支撑论文结论。"
        if topic == "Robotics":
            return "它的意义在于减少只看演示的误判，关键仍是实机成功率、失败案例和跨场景表现。"
        if topic == "World Model":
            return "这把世界模型从生成或预测能力推进到可用的决策中间层，论文价值取决于评估任务是否接近真实决策。"
        return "这篇论文的价值不在新名词本身，而在方法改动是否带来可复现的真实收益，尤其是对照实验、指标和失败条件是否完整。"

    def _paper_fact_story_summary(self, item: Dict[str, Any], facts: Dict[str, Any], evidence: List[str], paper_text: str) -> str:
        target = str(facts.get("target") or "").strip()
        method = self._paper_translate_english_target(target or paper_text)
        method_clause = self._paper_method_clause(item, facts, method, paper_text)
        evidence_bits = [self._paper_translate_english_evidence(point) for point in evidence[:3]]
        evidence_bits = [bit for bit in evidence_bits if bit]
        deduped_bits = []
        seen_bits = set()
        for bit in evidence_bits:
            key = re.sub(r"\s+", "", bit.lower())
            if key and key not in seen_bits:
                seen_bits.add(key)
                deduped_bits.append(self._trim_reason(bit, 42))
        if deduped_bits:
            evidence_clause = "证据是" + "、".join(deduped_bits[:2])
        else:
            evidence_clause = "证据来自实验或基准线索"
        meaning = self._paper_meaning_sentence(item, paper_text)
        return f"{method_clause}。{evidence_clause}。{meaning}"

    def _paper_substantive_description(self, item: Dict[str, Any]) -> str:
        facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
        preview = re.sub(r"\s+", " ", str(item.get("summary_preview", "") or "")).strip()
        summary = re.sub(r"\s+", " ", str(item.get("summary_display") or item.get("summary") or "")).strip()
        title = str(item.get("title_cn") or item.get("title") or "").strip()
        target = str(facts.get("target", "") or "").strip() if isinstance(facts, dict) else ""
        action = str(facts.get("action", "") or "").strip() if isinstance(facts, dict) else ""
        evidence = facts.get("evidence") if isinstance(facts, dict) else []
        if isinstance(evidence, str):
            evidence = [evidence]
        evidence = [re.sub(r"\s+", " ", str(point or "")).strip() for point in (evidence or []) if str(point or "").strip()]
        if self._chinese_enough_for_paper_description(summary):
            source = summary
        elif self._chinese_enough_for_paper_description(preview):
            source = preview
        else:
            source = summary or preview or title
        source = re.sub(r"^(这篇论文|该论文|这项研究|本文|论文)\s*", "", source)
        sentences = [sentence.strip() for sentence in re.split(r"[。！？.!?]+", source) if sentence.strip()]
        paper_text = " ".join([title, preview, summary, target, action, " ".join(evidence)]).lower()
        fact_story = self._paper_fact_story_summary(item, facts, evidence, paper_text)
        has_english_facts = bool(target and self._mostly_english(target)) or any(self._mostly_english(point) for point in evidence)

        useful_sentences = [sentence for sentence in sentences if self._useful_plain_sentence(sentence)]
        if useful_sentences:
            opener = self._trim_reason(useful_sentences[0], 112)
            if len(useful_sentences) > 1 and len(opener) < 82:
                opener = f"{opener}。{self._trim_reason(useful_sentences[1], 86)}"
            if opener and not opener.endswith(("。", "！", "？", ".", "!", "?")):
                opener += "。"
        elif target and action and not self._mostly_english(target + action):
            opener = f"这篇论文的主线是把{self._trim_reason(target, 44)}做得更清楚，并用{self._trim_reason(action, 36)}作为方法入口。"
        elif target and not self._mostly_english(target):
            opener = f"这篇论文围绕{self._trim_reason(target, 56)}展开，重点是让模型在这个任务上更可控、更可验证。"
        elif title:
            opener = f"这篇论文围绕「{self._trim_reason(title, 54)}」展开，主要描述一种可检查的模型或训练方法改动。"
        else:
            opener = ""

        evidence_text = ""
        if evidence:
            readable_evidence = [point for point in evidence if not self._mostly_english(point)]
            if readable_evidence:
                concise_evidence = [self._trim_reason(point, 58) for point in readable_evidence[:2]]
                evidence_joined = "、".join(point for point in concise_evidence if point)
                if evidence_joined:
                    evidence_text = "论文用" + evidence_joined + "支撑结论，说明它不是只提出概念，而是尝试给出可核对的结果。"
                else:
                    evidence_text = "论文给出了可核对的实验线索，主要信息来自对照组、评测任务和失败案例。"
            else:
                evidence_text = "论文的证据主要来自基准、成功率或真实/仿真任务评估，这些指标决定结论是否站得住。"
        elif len(sentences) > 1:
            evidence_text = self._trim_reason(sentences[1], 100)
            if evidence_text and not evidence_text.endswith(("。", "！", "？", ".", "!", "?")):
                evidence_text += "。"

        close = self._paper_meaning_sentence(item, paper_text)

        paragraph = " ".join(part for part in (opener, evidence_text, close) if part)
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        target_cn = self._trim_reason(target, 28) if target and not self._mostly_english(target) else ""
        missing_target = bool(target_cn and len(target_cn) >= 6 and target_cn not in paragraph)
        if has_english_facts and self._paper_description_is_substantive(fact_story):
            paragraph = fact_story
        elif missing_target or not self._paper_description_is_substantive(paragraph):
            paragraph = fact_story
        paragraph = re.sub(
            r"\b[A-Za-z][A-Za-z0-9./_-]*(?:\s+[A-Za-z][A-Za-z0-9./_-]*){3,}\b",
            "这篇论文",
            paragraph,
        )
        paragraph = re.sub(r"(这篇论文){2,}", "这篇论文", paragraph)
        if (
            "围绕论文中的核心任务" in paragraph
            or "论文给出了" in paragraph and "等量化证据" in paragraph
            or not self._paper_description_is_substantive(paragraph)
        ):
            subject = self._paper_short_name(item, facts)
            target_hint = self._paper_translate_english_target(target or title)
            evidence_hint = self._paper_translate_english_evidence(evidence[0]) if evidence else ""
            if evidence_hint:
                paragraph = f"{subject}的核心问题是{target_hint}。材料中能确认的证据是{evidence_hint}。现有摘要还不足以判断结论强度，应优先查看实验设置、对照基线和失败案例。"
            else:
                paragraph = f"{subject}主要围绕{target_hint}展开。现有材料没有给出足够实验细节，因此只适合作为论文线索，深读时应先看方法、指标和限制条件。"
        if len(paragraph) > 190:
            paragraph = paragraph[:189].rstrip(" ，；。") + "…"
        return paragraph

    def _paper_clean_fact_field(self, value: str, field: str, fallback: str = "") -> str:
        cleaned = self._clean_public_sentence(value, 90)
        if not cleaned:
            return fallback
        if contains_mojibake(cleaned):
            return fallback
        if self._mostly_english(cleaned):
            if field in {"metric", "baseline", "dataset"}:
                cleaned = self._paper_translate_english_evidence(cleaned)
            elif "limited" in cleaned.lower() or "limitation" in cleaned.lower():
                cleaned = "真实环境、失败案例或复现条件仍需要原文核对"
            else:
                cleaned = self._paper_translate_english_target(cleaned)
        if re.search(r"(^[A-Za-z]\b|对应技术模块|从标题和摘要看|当前材料不足)", cleaned):
            return fallback
        return self._clean_public_sentence(cleaned, 90)

    def _paper_specific_mechanism(self, item: Dict[str, Any], facts: Dict[str, Any], paper_text: str) -> Tuple[str, str]:
        title = str(item.get("title_cn") or item.get("title") or "").strip()
        target = str(facts.get("target") or "").strip() if isinstance(facts, dict) else ""
        readable_target = self._paper_translate_english_target(target or title)
        if any(token in paper_text for token in ("gazelnn", "scanpath", "gaze", "visual attention", "扫描路径")):
            return (
                "用轻量级液态神经网络预测视觉扫描路径，把人的注视轨迹当成随时间演化的动态系统来建模",
                "重点在于用更小的模型捕捉视觉注意变化，而不是继续堆更大的通用序列模型",
            )
        if "指令" in paper_text and "对齐" in paper_text:
            return (
                "通过细粒度指令对齐把语言目标拆成更具体的操作约束，再让 VLA 策略按这些约束执行动作",
                "关键在于减少自然语言指令和低层动作之间的歧义，而不是只扩大训练数据规模",
            )
        if any(token in paper_text for token in ("predict-then-act", "motion-aware", "dynamic manipulation", "moving objects")):
            return (
                "先用运动感知世界模型预测动态物体的下一步位置，再把预测结果交给冻结 VLA 做动作决策",
                "好处是不用重训底层策略模型，只在前面补一个负责预判的模块",
            )
        if any(token in paper_text for token in ("g$^3$vla", "g3vla", "camera-aware geometric", "prope", "ray embeddings", "cross-view fusion")):
            return (
                "把相机内参编码成射线嵌入，再用 PRoPE 和双向跨视角融合把几何信息送进 VLA 动作生成链路",
                "关键不是再加一种视觉特征，而是让策略知道同一物体在不同相机视角下对应的空间关系",
            )
        if any(token in paper_text for token in ("lie-algebra", "lie algebra", "矩阵李群", "李代数")):
            return (
                "把 token 表示成矩阵李群上的变换，再在李代数空间里做注意力计算",
                "它想解决的是普通注意力缺少几何结构约束的问题，关键看这种结构化表示是否带来稳定收益",
            )
        if "eventvla" in paper_text or "event vla" in paper_text:
            return (
                "把事件相机的高时间分辨率信号接入 VLA 策略，让机器人在快速变化场景里更早感知动作线索",
                "价值取决于事件流是否真的改善动态操作，而不是只增加一种传感器输入",
            )
        if "atom-bench" in paper_text or "atom bench" in paper_text:
            return (
                "把真实世界原子级任务整理成统一基准，用同一套任务和指标比较不同模型",
                "它的贡献主要是评测设计，能减少每篇论文各自定义任务带来的比较噪声",
            )
        if "bintrack" in paper_text:
            return (
                "围绕目标轨迹和查询表示做跟踪建模，用更明确的匹配关系减少长时跟踪漂移",
                "关键看它在遮挡、目标切换和长序列场景里是否比常规跟踪基线更稳",
            )
        if "slow brain" in paper_text:
            return (
                "把推理过程拆成更慢的规划或反思步骤，让模型在行动前先形成中间判断",
                "重点不是生成更长答案，而是这些额外思考步骤能否降低错误决策",
            )
        if "handtouch" in paper_text or "ht-bench" in paper_text:
            return (
                "把手部接触、触觉线索和交互任务整理成可评测问题，用来检验机器人是否真的理解接触过程",
                "技术价值在于把触觉和操作结果连接起来，而不是只看视觉识别是否正确",
            )
        if any(token in paper_text for token in ("grasp", "抓取", "partial point cloud", "点云")):
            return (
                "从部分观测点云中生成更稳的抓取候选，并用约束或评分机制筛掉不可靠姿态",
                "核心问题是信息不完整时还能不能合成可执行抓取，而不是只在完整模型上做规划",
            )
        if any(token in paper_text for token in ("操作理解", "机器人操作", "robot manipulation")):
            return (
                "把机器人操作任务拆成可理解的中间状态或动作约束，再用实验成功率检验策略是否真的更稳",
                "关键在于方法是否改善多任务操作表现，而不是只给出一个新的任务名称",
            )
        if any(token in paper_text for token in ("transformer-actor-critic", "actor-critic")):
            return (
                "把 Transformer 表示能力接到 actor-critic 强化学习框架里，用序列建模能力辅助策略更新",
                "关键看这种结构是否改善样本效率和稳定性，而不是只替换网络骨架",
            )
        if any(token in paper_text for token in ("success visitation matching", "sparse rewards", "dense process reward", "process rewards", "稀疏奖励", "过程奖励", "成功轨迹匹配")):
            return (
                "把成功轨迹访问过的状态转成密集过程奖励，让原本只有终点反馈的任务在训练中获得连续信号",
                "实验重点是看稀疏奖励任务里的成功率是否提升，而不是只判断奖励函数形式是否更复杂",
            )
        if any(token in paper_text for token in ("physdrift", "co-speech motion", "embodiment gap", "physics-aware humanoid")):
            return (
                "在生成全身共语动作前加入物理一致性约束，让手势和身体运动更接近可执行的人形机器人动作",
                "实验重点是看物理一致性指标是否优于普通文本到动作或共语动作生成基线",
            )
        if any(token in paper_text for token in ("world model", "世界模型", "预测", "planning", "规划")):
            return (
                "用模型预测后续状态，再把预测结果交给规划或策略模块做决策",
                "这条路线的价值在于把世界模型从生成能力推进到可验证的决策中间层",
            )
        if any(token in paper_text for token in ("vla", "vision-language-action", "机器人", "robot")):
            return (
                "把视觉、语言和动作放到同一条策略链路里，让机器人根据任务描述直接产生动作",
                "真正要看的不是概念是否新，而是跨物体、跨场景和真实机器人成功率是否提升",
            )
        return (
            f"把{self._trim_reason(readable_target, 48)}转成可测试的方法假设，再用实验指标判断是否有效",
            "如果摘要没有给出机制细节，这条只适合作为附录线索，不应在重点区过度展开",
        )

    def _paper_technical_intro_is_substantive(self, text: str) -> bool:
        if not str(text or "").strip():
            return False
        bad_phrases = (
            "围绕核心任务设计",
            "真正要判断的是这个方法",
            "论文给出了实验、基准或真实任务验证",
            "后续应核对对应指标含义",
            "目前可用材料不够细",
            "只适合作为附录线索",
        )
        if any(phrase in text for phrase in bad_phrases):
            return False
        return bool(self.METHOD_WORD_PATTERN.search(text or "") and self.RESULT_WORD_PATTERN.search(text or ""))

    def _paper_technical_intro(self, item: Dict[str, Any]) -> str:
        facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
        title = str(item.get("title_cn") or item.get("title") or "").strip()
        target = str(facts.get("target") or "").strip() if isinstance(facts, dict) else ""
        action = str(facts.get("action") or "").strip() if isinstance(facts, dict) else ""
        summary = str(item.get("summary_display") or item.get("summary") or "").strip()
        preview = str(item.get("summary_preview") or "").strip()
        evidence = facts.get("evidence") if isinstance(facts, dict) else []
        if isinstance(evidence, str):
            evidence = [evidence]
        evidence = [re.sub(r"\s+", " ", str(point or "")).strip() for point in (evidence or []) if str(point or "").strip()]
        paper_text = " ".join([title, target, action, summary, preview, " ".join(evidence)]).lower()
        subject = self._paper_short_name(item, facts)
        method_fallback = self._paper_translate_english_target(target or action or summary or title)
        method = self._paper_clean_fact_field(str(facts.get("method") or ""), "method", method_fallback)
        if "从标题和摘要看" in method or "当前材料不足" in method:
            if "policy evaluation correlation" in paper_text or "real-world success rate" in paper_text:
                method = "用世界模型提前评估策略在真实任务中的成功率"
            else:
                method, _ = self._paper_specific_mechanism(item, facts, paper_text)
        dataset = self._paper_clean_fact_field(str(facts.get("dataset_or_benchmark") or ""), "dataset")
        metric = self._paper_clean_fact_field(str(facts.get("metric_result") or ""), "metric")
        baseline = self._paper_clean_fact_field(str(facts.get("baseline") or ""), "baseline")
        limitation = self._paper_clean_fact_field(str(facts.get("limitation") or ""), "limitation")
        deployment = self._paper_clean_fact_field(str(facts.get("deployment_context") or ""), "deployment")
        evidence_bits = [self._paper_translate_english_evidence(point) for point in evidence[:2]]
        evidence_bits = [bit for bit in evidence_bits if bit]
        if not metric and evidence_bits:
            metric = evidence_bits[0]
        if not baseline and len(evidence_bits) > 1:
            baseline = evidence_bits[1]

        specific_mechanism, specific_benefit = self._paper_specific_mechanism(item, facts, paper_text)
        specific_known = not specific_mechanism.startswith("围绕")

        mechanism = method
        if specific_known:
            mechanism = specific_mechanism
            benefit = specific_benefit
        elif "指令" in paper_text and "对齐" in paper_text:
            mechanism = "通过细粒度指令对齐把语言目标拆成更具体的操作约束，再让 VLA 策略按这些约束执行动作"
            benefit = "关键在于减少自然语言指令和低层动作之间的歧义，而不是只扩大训练数据规模"
        elif any(token in paper_text for token in ("predict-then-act", "motion-aware", "dynamic manipulation", "moving objects")):
            mechanism = "先用运动感知世界模型预测动态物体的下一步位置，再把预测结果交给冻结 VLA 做动作决策"
            benefit = "好处是不用重训底层策略模型，只在前面补一个负责预判的模块"
        elif any(token in paper_text for token in ("whole-body", "humanoid", "motion tracking")):
            mechanism = "用因果 Transformer 建模全身动作序列，把大规模动作语料转成可跟踪的控制策略"
            benefit = "关键不是生成更像人的动作，而是让同一个控制模型覆盖更多未见动作和任务"
        elif any(token in paper_text for token in ("lidar", "4d world", "scene synthesis")):
            mechanism = "先估计 4D LiDAR 场景里哪些区域更不确定，再把生成和补全计算集中到这些区域"
            benefit = "这会让场景合成从平均生成转向按难度分配建模能力"
        elif any(token in paper_text for token in ("uav", "aerial navigation", "reward design")):
            mechanism = "让智能体参与奖励函数设计，再用导航反馈反复修正策略"
            benefit = "它想减少人工调奖励的成本，把无人机导航训练从手工经验改成可迭代搜索"
        elif any(token in paper_text for token in ("human-in-the-loop", "preference-calibrated", "intervention")):
            mechanism = "把人类接管和纠偏轨迹转成偏好信号，再用这些信号重新分配强化学习里的奖励信用"
            benefit = "这比单纯收集成功轨迹更接近真实训练场景，因为它利用了人类介入时暴露出的错误信息"
        elif any(token in paper_text for token in ("speech-and-motion", "dyadic interaction", "full-duplex")):
            mechanism = "把流式语音理解、轮次管理和身体动作生成放进同一个交互模型"
            benefit = "重点不是多生成一种模态，而是处理人机双向响应里的时间同步问题"
        elif any(token in paper_text for token in ("textual world model", "agent policies", "future-aware")):
            mechanism = "让文本世界模型预测后续状态，再和智能体策略一起迭代更新"
            benefit = "这样做的意义是把行动前的预判显式放进策略学习，而不是只依赖事后奖励"
        elif any(token in paper_text for token in ("navigation function", "motion planning", "zero-shot")):
            mechanism = "把学习得到的导航目标嵌入结构化规划器，让规划器在约束内寻找路径"
            benefit = "它试图同时保留传统规划的可解释性和学习方法的泛化能力"
        elif any(token in paper_text for token in ("affordance", "functional mask", "post-contact motion")):
            mechanism = "同时预测可交互区域和接触后的运动轨迹"
            benefit = "机器人因此不只知道在哪里动手，也知道动手后动作该怎么延续"
        elif any(token in paper_text for token in ("benchmark", "dataset", "leaderboard", "数据集", "基准")):
            mechanism = "通过数据划分、任务定义和指标设计，把不同模型放到同一套可比较条件下"
            benefit = "它的价值不在模型本身，而在于减少后续论文各自定义任务造成的比较噪声"
        else:
            mechanism, benefit = self._paper_specific_mechanism(item, facts, paper_text)

        sentences = [
            f"{subject}的关键做法是{mechanism}",
            benefit,
        ]
        result_clause = metric or dataset or (evidence_bits[0] if evidence_bits else "")
        generic_result_clauses = {
            "论文给出了实验、基准或真实任务验证",
            "论文给出了实验、基准或真实任务验证。",
        }
        if result_clause in generic_result_clauses:
            result_clause = ""
        if baseline in generic_result_clauses:
            baseline = ""
        if result_clause:
            if baseline:
                sentences.append(f"实验里真正值得看的是{result_clause}，因为它提供了和{baseline}对照的信号")
            else:
                sentences.append(f"实验里真正值得看的是{result_clause}，因为它说明方法收益不是单纯来自题目设定")
        elif deployment:
            sentences.append(f"应用上要看{deployment}里能否稳定复现这个收益")
        if limitation:
            sentences.append(f"局限是{limitation}")
        intro = self._join_sentences(sentences, 270)
        intro = re.sub(
            r"\b[A-Za-z][A-Za-z0-9./_-]*(?:\s+[A-Za-z][A-Za-z0-9./_-]*){4,}\b",
            "方法中的关键模块",
            intro,
        )
        intro = re.sub(r"\ba 方法中的关键模块\b", "方法中的关键模块", intro)
        intro = self._clean_public_sentence(intro, 270)
        if not re.search(r"[。！？]$", intro):
            intro += "。"
        return intro

    def _paper_description_lines(self, item: Dict[str, Any]) -> List[Dict[str, str]]:
        description = self._paper_substantive_description(item)
        return [{"label": "论文实质描述", "text": description}] if description else []

    def _paper_deep_dive_hint(self, item: Dict[str, Any]) -> str:
        facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
        evidence = facts.get("evidence") if isinstance(facts, dict) else []
        if isinstance(evidence, str):
            evidence = [evidence]
        evidence = evidence or []
        text = " ".join(
            str(value or "")
            for value in [
                item.get("title_cn"),
                item.get("summary"),
                item.get("summary_preview"),
                facts.get("target") if isinstance(facts, dict) else "",
                " ".join(str(point or "") for point in evidence[:3]),
            ]
        ).lower()
        if any(token in text for token in ("github", "code", "open-source", "开源", "repo")):
            return "先看代码、数据和第三方复现是否跟论文结论一致。"
        if any(token in text for token in ("success", "成功率", "%", "benchmark", "基准", "leaderboard")):
            return "先看实验表格、对照基线和失败案例，确认收益是否稳定。"
        if any(token in text for token in ("robot", "机器人", "humanoid", "真实", "real-world", "xarm")):
            return "先看真实机器人任务、环境变化和长时间运行表现。"
        if any(token in text for token in ("world model", "世界模型", "planning", "规划", "prediction", "预测")):
            return "先看预测结果能不能改善规划效果，而不是只提升生成质量。"
        return "先看方法细节、评测设置和是否有可复现实验。"

    def _build_change_map(self, items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for item in items:
            buckets.setdefault(self._topic_bucket(item), []).append(item)
        result: List[Dict[str, str]] = []
        for label in ("Agent", "Robotics", "World Model", "Infrastructure", "Open Source", "Research", "Business"):
            bucket_items = self._sort_items(buckets.get(label, []))
            if not bucket_items:
                continue
            top = bucket_items[0]
            result.append(
                {
                    "label": label,
                    "count": str(len(bucket_items)),
                    "summary": self._preview_text(top) or str(top.get("title_cn") or top.get("title") or "")[:70],
                }
            )
        return result[:6]

    def _build_learning_map(self, items: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, str]]:
        result: List[Dict[str, str]] = []
        seen = set()
        for item in self._sort_items(items):
            takeaway = self._learning_takeaway(item)
            if not takeaway:
                continue
            key = re.sub(r"\W+", "", takeaway.lower())[:48]
            if key in seen:
                continue
            seen.add(key)
            result.append(
                {
                    "domain": self._domain_label(item),
                    "takeaway": takeaway,
                    "url": str(item.get("url") or ""),
                    "title": str(item.get("title_cn") or item.get("title") or ""),
                }
            )
            if len(result) >= limit:
                break
        return result

    def _build_editorial_decisions(self, items: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, str]]:
        decisions: List[Dict[str, str]] = []
        seen_domains = set()
        for item in self._sort_items(items):
            if item.get("quality_tier") == "brief":
                continue
            domain = self._domain_label(item)
            if domain in seen_domains:
                continue
            line = str(item.get("editorial_lead") or item.get("analysis_body") or item.get("learning_takeaway") or "").strip()
            if not line:
                continue
            decisions.append(
                {
                    "domain": domain,
                    "text": self._trim_reason(line, 110),
                    "url": str(item.get("url") or ""),
                    "source_identity": editorial_source_identity(item),
                }
            )
            seen_domains.add(domain)
            if len(decisions) >= limit:
                break
        if len(decisions) < limit:
            for item in self._sort_items(items):
                line = str(item.get("editorial_lead") or item.get("analysis_body") or item.get("learning_takeaway") or "").strip()
                url = str(item.get("url") or "")
                if not line or any(existing.get("url") == url for existing in decisions):
                    continue
                decisions.append(
                    {
                        "domain": self._domain_label(item),
                        "text": self._trim_reason(line, 110),
                        "url": url,
                        "source_identity": editorial_source_identity(item),
                    }
                )
                if len(decisions) >= limit:
                    break
        return decisions[:limit]

    def _build_today_first_reads(self, items: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        seen = set()
        for item in self._sort_items(items):
            if item.get("quality_tier") == "brief":
                continue
            url = str(item.get("url") or "")
            if not url or url in seen:
                continue
            result.append(item)
            seen.add(url)
            if len(result) >= limit:
                break
        return result

    def _paper_technical_digest_items(self, items: List[Dict[str, Any]], limit: int = 18) -> List[Dict[str, Any]]:
        papers = [
            item
            for item in self._sort_items(items)
            if item.get("content_type") == "paper" and str(item.get("paper_technical_intro") or "").strip()
        ]
        return papers[:limit]

    def _build_domain_sections(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in items:
            grouped.setdefault(self._domain_key(item), []).append(item)
        sections: List[Dict[str, Any]] = []
        for domain_key in self.LEARNING_DOMAIN_ORDER:
            domain_items = self._sort_items(grouped.get(domain_key, []))
            if not domain_items:
                continue
            focus_items = [
                item
                for item in domain_items
                if item.get("report_section") in {"must_read", "physical_ai", "watch", "featured_papers"}
            ][:4]
            papers = [item for item in domain_items if item.get("content_type") == "paper"][:5]
            reading = [
                item
                for item in domain_items
                if item.get("content_type") != "paper" and item not in focus_items
            ][:4]
            top = focus_items[0] if focus_items else domain_items[0]
            sections.append(
                {
                    "key": domain_key,
                    "label": self._domain_label(domain_key),
                    "learning_focus": self._domain_learning_focus(domain_key),
                    "today_conclusion": self._learning_takeaway(top),
                    "focus_items": focus_items,
                    "papers": papers,
                    "reading": reading,
                    "count": len(domain_items),
                }
            )
        return sections

    def _paper_appendix_groups(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in items:
            grouped.setdefault(self._domain_key(item), []).append(item)
        groups: List[Dict[str, Any]] = []
        for domain_key in self.LEARNING_DOMAIN_ORDER:
            papers = [item for item in self._sort_items(grouped.get(domain_key, [])) if item.get("content_type") == "paper"]
            if papers:
                groups.append({"label": self._domain_label(domain_key), "articles": papers})
        return groups

    def _build_research_groups(self, research_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in research_items:
            grouped.setdefault(self._research_group_label(item), []).append(item)
        result: List[Dict[str, Any]] = []
        for label in ("Physical AI", "World Model", "Robotics", "模型研究"):
            items = self._sort_items(grouped.get(label, []))
            if not items:
                continue
            result.append(
                {
                    "label": label,
                    "summary": self._preview_text(items[0]) or "本组论文围绕方法、实验指标和可复现价值展开。",
                    "items": items,
                }
            )
        return result

    def _build_quality_footnote(self, layers: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        visible_keys = ("must_read", "physical_ai", "watch", "featured_papers", "paper_appendix", "brief")
        all_items = [item for key in visible_keys for item in layers.get(key, [])]
        if not layers.get("featured_papers") and layers.get("research"):
            all_items.extend(layers.get("research", []))
        low_evidence_count = sum(1 for item in all_items if self._evidence_quality(item) < 0.35)
        aggregator_count = sum(1 for item in all_items if str(item.get("source_tier", "") or "") == "aggregator")
        domain_counts: Dict[str, int] = {}
        for item in all_items:
            key = self._domain_key(item)
            domain_counts[key] = domain_counts.get(key, 0) + 1
        domain_covered = sum(1 for key in self.LEARNING_DOMAIN_ORDER if domain_counts.get(key, 0))
        paper_count = len(layers.get("featured_papers", [])) + len(layers.get("paper_appendix", []))
        return {
            "model_path": str(
                self.report_config.get("model_path_label")
                or "GPT 结构化事实抽取 + 中文编辑"
            ),
            "low_evidence_count": low_evidence_count,
            "aggregator_count": aggregator_count,
            "high_evidence_count": sum(1 for item in all_items if self._is_high_evidence(item)),
            "total_count": len(all_items),
            "paper_count": paper_count,
            "domain_covered": domain_covered,
            "domain_total": len(self.LEARNING_DOMAIN_ORDER),
            "suspicious_claim_count": sum(1 for item in all_items if "suspicious_claim" in (item.get("quality_flags") or [])),
            "filtered_note": f"{low_evidence_count + aggregator_count} 条低证据或聚合源内容已从重点区移出，压缩到快讯/待确认中。",
            "closing_note": f"本期结束：{low_evidence_count} 条低证据内容已压缩或降级，下一封继续观察重点方向是否出现新的事实证据。",
        }

    def _clean_summary_for_display(self, item: Dict[str, Any]) -> str:
        summary = re.sub(r"\s+", " ", str(item.get("summary", "") or "")).strip()
        summary = self.SUMMARY_LABEL_PATTERN.sub("", summary)
        summary = self._clean_public_sentence(summary, 1200)
        if self._is_high_evidence(item):
            return summary
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[。！？])\s*", summary) if sentence.strip()]
        short_summary = "".join(sentences[:2]) if sentences else summary
        short_summary = short_summary[:150] + ("…" if len(short_summary) > 150 else "")
        return self._clean_public_sentence(short_summary, 180)

    def _specific_text(self, text: str, item: Dict[str, Any], limit: int = 90) -> str:
        cleaned = self._trim_reason(text, limit)
        if not cleaned or self.GENERIC_ANALYSIS_PATTERN.search(cleaned):
            return ""
        if self._evidence_quality(item) < 0.35 and not re.search(r"\d|%|客户|企业|产品|模型|芯片|部署|成本|基准|复现", cleaned, re.IGNORECASE):
            return ""
        return cleaned

    def _analysis_items(self, item: Dict[str, Any]) -> List[Dict[str, str]]:
        fields = [
            ("为什么现在", item.get("why_now", "")),
            ("直接影响", item.get("expected_effect", "")),
            ("后续变量", item.get("future_impact", "")),
        ]
        seen = set()
        result: List[Dict[str, str]] = []
        for label, value in fields:
            cleaned = self._specific_text(str(value or ""), item, 76)
            key = re.sub(r"\W+", "", cleaned.lower())
            if not cleaned or key in seen:
                continue
            seen.add(key)
            result.append({"label": label, "value": cleaned})
        return result

    def _dedupe_repeated_reasons(self, items: List[Dict[str, Any]], field: str, limit: int = 2) -> None:
        counts: Dict[str, int] = {}
        for item in items:
            value = str(item.get(field, "") or "")
            key = re.sub(r"\W+", "", value.lower())
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
            if counts[key] > limit:
                item[field] = ""

    def _highlight_reason(self, item: Dict[str, Any]) -> str:
        topic = str(item.get("display_topic", "") or "")
        content_type = str(item.get("content_type", "") or "")
        why_it_matters = self._trim_reason(item.get("why_it_matters", ""))
        why_now = self._trim_reason(item.get("why_now", ""))
        expected_effect = self._trim_reason(item.get("expected_effect", ""))
        future_impact = self._trim_reason(item.get("future_impact", ""))

        if content_type == "paper":
            if "世界模型" in topic:
                return expected_effect or why_it_matters or "先看它能否把预测能力变成更低试错成本"
            if "具身智能" in topic:
                return expected_effect or why_now or "先看它能否把实验结果带进真实环境闭环"
            if "机器人" in topic:
                return expected_effect or future_impact or "先看它能否把论文效果转成实机成功率"
            return why_it_matters or expected_effect or "先看它是否真的更接近可部署阶段"

        if "产品发布" in topic:
            return expected_effect or why_it_matters or "先看它会先替代哪些高频人工步骤"
        if "企业合作" in topic:
            return why_it_matters or future_impact or "先看合作能否补齐客户、渠道或交付短板"
        if "开源生态" in topic:
            return future_impact or why_it_matters or "先看它会不会变成社区默认做法"
        if "基础设施" in topic:
            return expected_effect or future_impact or "先看它会先改写哪些成本和供给关系"
        if "应用落地" in topic:
            return expected_effect or why_it_matters or "先看这套方案能不能跨场景复制"
        if "模型/研究" in topic:
            return why_it_matters or expected_effect or "先看能力提升会不会兑现成更稳交付"
        return why_it_matters or expected_effect or future_impact or "先看这条变化会先影响哪一层"

    def _card_reason(self, item: Dict[str, Any]) -> str:
        topic = str(item.get("display_topic", "") or "")
        content_type = str(item.get("content_type", "") or "")
        why_it_matters = self._trim_reason(item.get("why_it_matters", ""), 34)
        expected_effect = self._trim_reason(item.get("expected_effect", ""), 34)
        future_impact = self._trim_reason(item.get("future_impact", ""), 34)

        if content_type == "paper":
            if "世界模型" in topic:
                return expected_effect or "先看能否变成更低试错成本"
            if "具身智能" in topic:
                return expected_effect or "先看能否带到真实环境"
            if "机器人" in topic:
                return expected_effect or future_impact or "先看能否转成实机成功率"
            return why_it_matters or "先看是否更接近可部署"

        if "产品发布" in topic:
            return expected_effect or "先看会先替代哪些人工步骤"
        if "企业合作" in topic:
            return why_it_matters or "先看能否补齐客户或交付短板"
        if "开源生态" in topic:
            return future_impact or "先看会不会变成社区默认做法"
        if "基础设施" in topic:
            return expected_effect or "先看会先改写哪些成本关系"
        if "应用落地" in topic:
            return expected_effect or "先看能不能跨场景复制"
        if "模型/研究" in topic:
            return why_it_matters or "先看会不会兑现成更稳交付"
        return why_it_matters or expected_effect or future_impact or "先看这条变化先影响哪一层"

    def _decorate_highlights(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        decorated: List[Dict[str, Any]] = []
        for item in items:
            decorated_item = dict(item)
            decorated_item["highlight_reason"] = self._highlight_reason(item)
            decorated_item["summary_preview"] = self._preview_text(item)
            decorated.append(decorated_item)
        self._dedupe_repeated_reasons(decorated, "highlight_reason", limit=2)
        return decorated

    def _decorate_items(self, items: List[Dict[str, Any]], offset: int = 0, prefix: str = "card") -> List[Dict[str, Any]]:
        decorated: List[Dict[str, Any]] = []
        for index, item in enumerate(items, start=1 + offset):
            decorated_item = enrich_editorial_fields(dict(item))
            decorated_item["order"] = index
            decorated_item["anchor_id"] = f"{prefix}-{index:02d}"
            decorated_item["title_cn"] = decorated_item.get("editorial_title") or item.get("title_cn") or item.get("title") or "未命名情报"
            if self._classic_nav_title_needs_repair(str(decorated_item.get("title_cn") or "")):
                decorated_item["title_cn"] = self._classic_nav_lead_title(decorated_item)
            decorated_item["summary_preview"] = self._preview_text(item)
            decorated_item["impact_tag"] = item.get("impact_tag", "值得关注")
            decorated_item["content_kind"] = item.get(
                "content_kind",
                "论文" if item.get("content_type") == "paper" else "全网动态",
            )
            decorated_item["content_kind"] = self._display_label(decorated_item["content_kind"])
            decorated_item["display_topic"] = self._display_label(
                item.get("display_topic")
                or item.get("topic_cn")
                or item.get("category")
                or "其他"
            )
            decorated_item["quick_reason"] = self._card_reason(decorated_item)
            decorated_item["read_reason"] = decorated_item["quick_reason"]
            decorated_item["summary_display"] = self._clean_summary_for_display(decorated_item)
            decorated_item["why_it_matters_display"] = self._specific_text(str(decorated_item.get("why_it_matters", "") or ""), decorated_item, 86)
            decorated_item["analysis_items"] = self._analysis_items(decorated_item)
            decorated_item["evidence_tier"] = "high" if self._is_high_evidence(decorated_item) else "brief"
            decorated_item["evidence_tier_label"] = "重点" if decorated_item["evidence_tier"] == "high" else "短讯"
            decorated_item["evidence_tier_label"] = self._quality_label(decorated_item)
            decorated_item["source_tier_label"] = self._source_tier_label(decorated_item)
            decorated_item["tone_label"] = self._tone_label(decorated_item)
            decorated_item["evidence_quality_display"] = self._metric_text(self._evidence_quality(decorated_item))
            decorated_item["information_density_display"] = self._metric_text(self._information_density(decorated_item))
            decorated_item["brief_line"] = decorated_item.get("brief_line") or self._preview_text(decorated_item) or decorated_item["summary_display"][:80]
            decorated_item["research_group_label"] = self._research_group_label(decorated_item)
            decorated_item["research_value_tags"] = self._research_value_tags(decorated_item) if decorated_item.get("content_type") == "paper" else []
            decorated_item["paper_description"] = self._paper_substantive_description(decorated_item) if decorated_item.get("content_type") == "paper" else ""
            if decorated_item.get("content_type") == "paper":
                if self._is_v10_reader():
                    decorated_item["paper_technical_intro"] = str(decorated_item.get("paper_technical_intro") or "")
                    decorated_item["paper_plain_summary"] = str(decorated_item.get("paper_plain_summary") or "")
                else:
                    decorated_item["paper_technical_intro"] = self._paper_technical_intro(decorated_item)
            else:
                decorated_item["paper_technical_intro"] = ""
            decorated_item["paper_description_lines"] = self._paper_description_lines(decorated_item) if decorated_item.get("content_type") == "paper" else []
            decorated_item["plain_summary"] = ""
            decorated_item["plain_summary_lines"] = []
            decorated_item["memory_point"] = ""
            decorated_item["deep_dive_hint"] = self._paper_deep_dive_hint(decorated_item) if decorated_item.get("content_type") == "paper" else ""
            facts = decorated_item.get("facts") if isinstance(decorated_item.get("facts"), dict) else {}
            evidence_points = facts.get("evidence") if isinstance(facts, dict) else []
            evidence_points = evidence_points or []
            if isinstance(evidence_points, str):
                evidence_points = [evidence_points]
            decorated_item["evidence_points"] = [
                self._public_evidence_point(decorated_item, point)
                for point in evidence_points
                if str(point).strip()
            ][:3]
            decorated_item["evidence_inline"] = "；".join(decorated_item["evidence_points"][:2])
            decorated_item["audience"] = facts.get("audience", "") if isinstance(facts, dict) else ""
            decorated_item["domain_key"] = self._domain_key(decorated_item)
            decorated_item["domain_label"] = self._domain_label(decorated_item)
            decorated_item["learning_takeaway"] = decorated_item.get("editorial_lead") or self._learning_takeaway(decorated_item)
            decorated_item["technical_context"] = decorated_item.get("analysis_body") or self._technical_context(decorated_item)
            decorated_item["background_context"] = decorated_item.get("evidence_line") or self._background_context(decorated_item)
            decorated_item["deep_dive_prompt"] = decorated_item.get("reader_next_step") or self._deep_dive_prompt(decorated_item)
            decorated_item["technical_digest"] = self._technical_digest_points(decorated_item)
            decorated_item["editorial_brief"] = decorated_item.get("analysis_body") or self._editorial_brief(decorated_item)
            decision_line = decorated_item.get("highlight_reason") or decorated_item.get("quick_reason") or decorated_item.get("summary_preview")
            decorated_item["decision_line"] = self._trim_reason(str(decision_line or ""), 92)
            decorated_item["feedback_links"] = decorated_item.get("feedback_links") or {}
            decorated_item["primary_feedback_links"] = {
                key: value
                for key, value in decorated_item["feedback_links"].items()
                if key in {"useful", "track"}
            }
            decorated_item["secondary_feedback_links"] = {
                key: value
                for key, value in decorated_item["feedback_links"].items()
                if key in {"not_useful", "mute_similar", "too_shallow", "paper_too_shallow", "paper_unclear", "too_long", "too_generic", "source_suspicious", "not_memorable"}
            }
            section_labels = {
                "must_read": "必读",
                "physical_ai": "具身智能",
                "watch": "观察",
                "featured_papers": "论文精选",
                "paper_appendix": "论文附录",
                "research": "论文/研究",
                "brief": "快讯",
            }
            decorated_item["report_section_label"] = section_labels.get(str(decorated_item.get("report_section", "")), "")
            decorated_item["report_section_label"] = {
                "must_read": "必读",
                "physical_ai": "具身智能",
                "watch": "观察",
                "featured_papers": "论文精选",
                "paper_appendix": "论文附录",
                "research": "论文/研究",
                "brief": "快讯",
            }.get(str(decorated_item.get("report_section", "")), decorated_item["report_section_label"])
            decorated_item["is_expanded_default"] = False
            decorated.append(decorated_item)
        self._dedupe_repeated_reasons(decorated, "quick_reason", limit=2)
        return decorated

    def _decorate_layers(self, layers: Optional[Dict[str, List[Dict[str, Any]]]]) -> Dict[str, List[Dict[str, Any]]]:
        layers = layers or {}
        provided_items = [item for key in self.REPORT_SECTION_ORDER for item in layers.get(key, [])]
        if provided_items and all(item.get("_final_render_item") for item in provided_items):
            return {key: [dict(item) for item in layers.get(key, [])] for key in self.REPORT_SECTION_ORDER}
        result: Dict[str, List[Dict[str, Any]]] = {}
        offset = 0
        for key in self.REPORT_SECTION_ORDER:
            items = self._decorate_items(layers.get(key, []), offset, key)
            for item in items:
                item["report_section"] = item.get("report_section") or key
                item["report_section_label"] = {
                    "must_read": "必读",
                    "physical_ai": "具身智能",
                    "watch": "观察",
                    "featured_papers": "论文精选",
                    "paper_appendix": "论文附录",
                    "research": "论文/研究",
                    "brief": "快讯",
                }.get(key, item.get("report_section_label", ""))
            result[key] = items
            offset += len(items)
        if not result.get("featured_papers") and layers.get("research"):
            fallback_research = self._decorate_items(layers.get("research", []), offset, "research")
            result["featured_papers"] = fallback_research[:6]
            result["paper_appendix"] = fallback_research[6:] + result.get("paper_appendix", [])
        return result

    def _hero_description(
        self,
        papers: List[Dict[str, Any]],
        updates: List[Dict[str, Any]],
        report_summary: Dict[str, Any],
    ) -> str:
        combined = self._decorate_items((papers or [])[:6] + (updates or [])[:8], 0, "hero")
        high_items = [item for item in combined if self._is_high_evidence(item)]
        if high_items:
            domain_lines: Dict[str, str] = {}
            for item in high_items:
                domain = self._domain_label(item)
                if domain in domain_lines:
                    continue
                takeaway = self._learning_takeaway(item)
                if domain == "World Model":
                    domain_lines[domain] = f"世界模型线开始从演示式生成，转向{self._trim_reason(takeaway, 38)}"
                elif domain == "Physical AI / Robotics":
                    domain_lines[domain] = f"机器人线更重视真实任务证据，代表信号是{self._trim_reason(takeaway, 38)}"
                elif domain == "Agent / Models":
                    domain_lines[domain] = f"Agent 和模型线继续从聊天入口推进到{self._trim_reason(takeaway, 38)}"
                elif domain == "Infra / Open Source":
                    domain_lines[domain] = f"基础设施和开源线的变量集中在{self._trim_reason(takeaway, 38)}"
                else:
                    domain_lines[domain] = f"产品和商业线需要记住{self._trim_reason(takeaway, 38)}"
                if len(domain_lines) >= 3:
                    break
            if domain_lines:
                ordered = [domain_lines[key] for key in self.LEARNING_DOMAIN_LABELS.values() if key in domain_lines]
                if not ordered:
                    ordered = list(domain_lines.values())
                prefixes = ["第一", "第二", "第三"]
                parts = [f"{prefixes[index]}，{line}" for index, line in enumerate(ordered[:3])]
                return "今天 AI 技术线最值得记住的是三件事：" + "；".join(parts) + "。"
        lead_summary = str(report_summary.get("lead_summary", "") or "").strip()
        if lead_summary and not self._has_bad_public_sentence(lead_summary):
            first_sentence = re.split(r"(?<=[。！？])\s*", lead_summary)[0].strip()
            if first_sentence:
                return self._clean_public_sentence(first_sentence, 160) + "。"
        return (
            f"本期聚焦具身智能、世界模型、机器人与全网 AI 动态，"
            f"共精选 {len(papers)} 篇论文和 {len(updates)} 条高价值情报。"
        )

    def _hero_scope(
        self,
        papers: List[Dict[str, Any]],
        updates: List[Dict[str, Any]],
        report_summary: Dict[str, Any],
    ) -> List[str]:
        scope = [
            f"{len(papers)} 篇精选论文",
            f"{len(updates)} 条全网动态",
            "论文与情报混排展示",
        ]
        hot_topics = [str(item).strip() for item in (report_summary.get("hot_topics") or []) if str(item).strip()]
        scope.extend(hot_topics[:3])
        unique_scope: List[str] = []
        seen = set()
        for item in scope:
            if item in seen:
                continue
            seen.add(item)
            unique_scope.append(item)
        return unique_scope[:6]

    def _pick_top_highlights(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sorted_all = self._sort_items(items)
        eligible = [item for item in sorted_all if self._is_high_evidence(item)]
        highlight_pool = eligible if len(eligible) >= 5 else sorted_all
        updates = [item for item in highlight_pool if item.get("content_type") != "paper"]
        papers = [item for item in highlight_pool if item.get("content_type") == "paper"]

        selected: List[Dict[str, Any]] = []
        selected_urls = set()
        update_target = 3 if len(updates) >= 3 else max(1, len(updates))
        paper_target = 2 if len(papers) >= 2 else max(1, len(papers))

        for candidate in updates[:update_target] + papers[:paper_target]:
            url = candidate.get("url")
            if not url or url in selected_urls:
                continue
            selected.append(candidate)
            selected_urls.add(url)

        for candidate in highlight_pool:
            if len(selected) >= 5:
                break
            url = candidate.get("url")
            if not url or url in selected_urls:
                continue
            selected.append(candidate)
            selected_urls.add(url)

        return self._sort_items(selected)[:5]

    def _exclude_highlight_duplicates(
        self,
        card_items: List[Dict[str, Any]],
        top_highlights: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        highlight_urls = {str(item.get("url", "")) for item in top_highlights if item.get("url")}
        body_items = [item for item in card_items if str(item.get("url", "")) not in highlight_urls]
        return self._decorate_items(body_items, 0, "mix-body")

    def _render_card_block(self, lines: List[str], items: List[Dict[str, Any]], heading: str, note: str = "") -> None:
        if not items:
            return
        lines.append(f"## {heading}")
        if note:
            lines.append(note)
            lines.append("")
        for article in items:
            lines.append(f"### {article['order']}. {article.get('title_cn')}")
            lines.append(
                f"- 类型: {article.get('content_kind')} | 主题: {article.get('display_topic')} | "
                f"影响标签: {article.get('impact_tag')} | Score: {article.get('score')} | 时间: {str(article.get('publish_date', ''))[:16]}"
            )
            preview = article.get("summary_preview")
            if preview:
                lines.append(f"- 导语: {preview}")
            lines.append(f"- 摘要: {article.get('summary_display') or article.get('summary')}")
            if article.get("content_type") == "paper" and article.get("paper_technical_intro"):
                lines.append(f"- 技术介绍: {article.get('paper_technical_intro')}")
            why = article.get("why_it_matters_display")
            if why:
                lines.append(f"- 为什么值得看: {why}")
            for analysis_item in article.get("analysis_items") or []:
                lines.append(f"- {analysis_item.get('label')}: {analysis_item.get('value')}")
            lines.append(f"- 原文: {article.get('url')}")
            lines.append("")

    def _build_tracking_items(self, items: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, str]]:
        result: List[Dict[str, str]] = []
        seen = set()
        for item in self._sort_items(items):
            title = str(item.get("title_cn") or item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title or url in seen:
                continue
            reason = (
                self._specific_text(str(item.get("future_impact", "") or ""), item, 72)
                or self._specific_text(str(item.get("expected_effect", "") or ""), item, 72)
                or str(item.get("quick_reason") or item.get("read_reason") or item.get("summary_preview") or "").strip()
            )
            if not reason:
                continue
            question = self._tracking_question(item, reason)
            result.append(
                {
                    "title": title,
                    "reason": question,
                    "url": url,
                    "section": str(item.get("report_section_label") or item.get("display_topic") or ""),
                }
            )
            seen.add(url)
            if len(result) >= limit:
                break
        return result

    def _tracking_question(self, item: Dict[str, Any], reason: str) -> str:
        title = str(item.get("title_cn") or item.get("title") or "").strip()
        topic = str(item.get("display_topic") or item.get("category") or "").lower()
        text = " ".join(str(item.get(key, "") or "") for key in ("summary", "summary_preview", "future_impact", "expected_effect")).lower()
        subject = self._trim_reason(title, 34) or "这条进展"
        if item.get("content_type") == "paper":
            if "github" in text or "code" in text or "开源" in text:
                return f"{subject} 是否出现第三方复现或更多实测结果？"
            return f"{subject} 是否开源代码、数据，或出现独立复现实验？"
        if "具身" in topic or "robot" in topic or "机器人" in topic or "physical ai" in topic:
            return f"{subject} 是否出现真实客户案例、部署数据或第三方评测？"
        if "基础设施" in topic or "infrastructure" in topic or "算力" in text or "芯片" in text:
            return f"{subject} 是否披露客户、成本变化或可交付时间表？"
        if "合作" in topic or "融资" in topic or "partnership" in topic or "funding" in text:
            return f"{subject} 是否进入具体产品、客户或资源投入阶段？"
        return f"{subject} 是否出现新的采用数据、客户案例或产品细节？"

    def _classic_overview(self, report_summary: Dict[str, Any], all_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        hot_topics = [str(item).strip() for item in (report_summary.get("hot_topics") or []) if str(item).strip()]
        key_takeaways = [
            str(item).strip()
            for item in (report_summary.get("key_takeaways") or [])
            if str(item).strip()
        ][:5]
        watchlist = [
            str(item).strip()
            for item in (report_summary.get("watchlist") or [])
            if str(item).strip()
        ][:5]

        if not key_takeaways:
            for item in self._sort_items(all_items)[:5]:
                line = (
                    str(item.get("summary_preview") or "").strip()
                    or self._trim_reason(str(item.get("summary_display") or item.get("summary") or ""), 86)
                )
                if line:
                    key_takeaways.append(line)
        if not watchlist:
            for item in self._sort_items(all_items):
                reason = str(item.get("reader_next_step") or item.get("quick_reason") or "").strip()
                title = str(item.get("title_cn") or item.get("title") or "").strip()
                if title and reason:
                    watchlist.append(f"{self._trim_reason(title, 28)}：{self._trim_reason(reason, 72)}")
                if len(watchlist) >= 5:
                    break

        return {
            "overall": str(report_summary.get("lead_summary") or "").strip() or self._editor_judgement(all_items),
            "paper_trend": str(report_summary.get("paper_summary") or "").strip(),
            "update_trend": str(report_summary.get("update_summary") or "").strip(),
            "hot_topics": hot_topics[:8],
            "key_takeaways": key_takeaways[:5],
            "watchlist": watchlist[:5],
        }

    def _classic_analysis_items(self, item: Dict[str, Any]) -> List[Dict[str, str]]:
        fields = [
            ("为什么这么做", item.get("why_now", "")),
            ("会带来什么效果", item.get("expected_effect", "")),
            ("未来影响", item.get("future_impact", "")),
        ]
        seen = set()
        result: List[Dict[str, str]] = []
        for label, value in fields:
            cleaned = self._specific_text(str(value or ""), item, 92)
            key = re.sub(r"\W+", "", cleaned.lower())
            if not cleaned or key in seen:
                continue
            seen.add(key)
            result.append({"label": label, "value": cleaned})
        return result

    def _classic_priority(self, item: Dict[str, Any]) -> Dict[str, str]:
        section = str(item.get("report_section") or "")
        content_type = str(item.get("content_type") or "")
        if section == "must_read":
            return {"label": "先读", "class": "priority-must"}
        if section == "physical_ai":
            return {"label": "具身", "class": "priority-physical"}
        if section in {"featured_papers", "paper_appendix"} or content_type == "paper":
            return {"label": "论文", "class": "priority-paper"}
        if section == "brief" or item.get("evidence_tier") != "high":
            return {"label": "快讯", "class": "priority-brief"}
        return {"label": "观察", "class": "priority-watch"}

    def _classic_nav_domain_key(self, item: Dict[str, Any]) -> str:
        topic_text = " ".join(
            str(item.get(key, "") or "")
            for key in ("display_topic", "topic_cn", "category", "content_kind")
        ).lower()
        item_text = self._item_search_text(item)
        if "world model" in topic_text or "世界模型" in topic_text:
            return "world_model"
        if any(token in topic_text for token in ("physical ai", "robotics", "具身", "机器人")):
            return "physical_ai"
        if any(token in topic_text for token in ("open source", "开源", "基础设施", "infrastructure")):
            return "infra_open_source"
        if any(token in item_text for token in ("open source", "github", "license", "开源", "gpu", "inference", "chip", "datacenter", "算力", "芯片", "推理芯片")):
            return "infra_open_source"
        if any(token in item_text for token in ("agent", "workflow", "reasoning", "llm", "gpt", "claude", "gemini", "智能体", "工作流", "推理模型")):
            return "agent_models"
        if any(token in topic_text for token in ("产品", "product release", "application", "应用", "business", "合作", "行业动态")):
            return "products_business"
        if any(token in item_text for token in ("world model", "world models", "世界模型", "jepa", "latent dynamics", "video prediction", "predictive model")):
            return "world_model"
        if self._is_physical_ai_item(item):
            return "physical_ai"
        return self._domain_key(item)

    def _classic_nav_representative_score(self, domain_key: str, item: Dict[str, Any]) -> float:
        text = self._item_search_text(item)
        topic_text = " ".join(
            str(item.get(key, "") or "")
            for key in ("display_topic", "topic_cn", "category", "content_kind", "title_cn", "title")
        ).lower()
        title_text = " ".join(
            str(item.get(key, "") or "")
            for key in ("title_cn", "title", "summary_preview")
        ).lower()
        score = self._score_value(item) + self._evidence_quality(item) * 1.5 + self._information_density(item)
        if item.get("evidence_tier") == "high":
            score += 1.0
        if item.get("report_section") == "must_read":
            score += 0.8
        if item.get("content_type") == "paper":
            score += 0.35
            facts = self._facts_for_item(item)
            paper_subject = str(facts.get("who") or "").strip().lower()
            paper_target = str(facts.get("target") or "").strip()
            if paper_subject in {"", "unknown", "this paper", "researchers"} or not paper_target:
                score -= 12.0
        lead_title = self._classic_nav_lead_title(item)
        if re.search(r"这篇论文|当前材料不足|从标题和摘要看|只适合作为附录线索", lead_title):
            score -= 10.0
        domain_terms = {
            "world_model": (
                "world model",
                "world models",
                "世界模型",
                "latent dynamics",
                "jepa",
                "video prediction",
                "predictive model",
                "rollout",
                "planning",
                "规划",
                "预测",
            ),
            "physical_ai": (
                "physical ai",
                "robot",
                "robotics",
                "embodied",
                "humanoid",
                "vla",
                "manipulation",
                "具身",
                "机器人",
                "机械臂",
                "操作",
            ),
            "agent_models": (
                "agent",
                "workflow",
                "reasoning",
                "tool use",
                "智能体",
                "工作流",
                "推理",
                "工具调用",
                "任务执行",
            ),
            "infra_open_source": (
                "open source",
                "github",
                "license",
                "gpu",
                "inference",
                "chip",
                "datacenter",
                "开源",
                "算力",
                "推理",
                "芯片",
                "部署",
            ),
            "products_business": (
                "product",
                "customer",
                "enterprise",
                "funding",
                "partnership",
                "产品",
                "客户",
                "企业",
                "融资",
                "合作",
                "商业",
            ),
        }.get(domain_key, ())
        for term in domain_terms:
            if term in topic_text:
                score += 2.2
            elif term in text:
                score += 1.0
        if domain_key == "agent_models" and re.search(r"\bocr\b|文档处理|识别", text) and not re.search(r"agent|workflow|智能体|工作流|任务执行", text):
            score -= 2.0
        if domain_key == "world_model":
            if re.search(r"world model|world models|世界模型|latent dynamics|jepa|video prediction|predictive model|rollout", title_text):
                score += 5.0
            if not re.search(r"world model|世界模型|latent dynamics|jepa|video prediction|predictive|rollout", text):
                score -= 1.4
            if re.search(r"physical ai|robot|robotics|vla|pose|slam|机器人|具身|姿态估计", title_text) and not re.search(
                r"world model|world models|世界模型|latent dynamics|jepa|video prediction|predictive model", title_text
            ):
                score -= 2.5
        if domain_key == "physical_ai" and re.search(r"physical ai|robot|robotics|vla|humanoid|manipulation|机器人|具身|机械臂|操作", title_text):
            score += 4.0
        return score

    def _classic_nav_representative(self, domain_key: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        return max(items, key=lambda item: self._classic_nav_representative_score(domain_key, item))

    def _classic_nav_focus(self, domain_key: str) -> str:
        return {
            "world_model": "看预测、潜空间动态和规划闭环是否变得可验证。",
            "physical_ai": "看方法是否跨过仿真，进入真实机器人任务证据。",
            "agent_models": "看模型能力如何变成工具调用、权限和工作流执行。",
            "infra_open_source": "看算力、推理成本、部署栈和开源路线变化。",
            "products_business": "看产品入口、客户采用和商业协作信号。",
        }.get(domain_key, "看这类内容改变了哪条 AI 学习主线。")

    def _classic_nav_title_needs_repair(self, title: str) -> bool:
        if not title or contains_mojibake(title):
            return True
        if re.search(r"Product\s+Rele|Industry\s+Upda|AI领域新进展|相关进展", title, re.IGNORECASE):
            return True
        if re.search(r"[\u4e00-\u9fff][a-z]{1,3}\b", title):
            return True
        if not re.search(r"[\u4e00-\u9fff]", title) and len(re.findall(r"[A-Za-z][A-Za-z0-9.+#'/_-]{2,}", title)) >= 4:
            return True
        if re.search(r"[A-Za-z]{4,}\s+[A-Za-z]{3,}\s*$", title) and len(title) > 30:
            return True
        if re.search(
            r"(?:增加|减少|提升|降低|扩展|优化|改进|支持|引入|采用|实现|构建|发布|更新|训练|部署)，(?:也|并|但|同时|仍)",
            title,
        ):
            return True
        return False

    def _classic_nav_lead_title(self, item: Dict[str, Any]) -> str:
        for key in ("editorial_title", "title_cn", "summary_preview", "title"):
            candidate = str(item.get(key) or "").strip()
            if candidate and not self._classic_nav_title_needs_repair(candidate):
                return self._trim_reason(candidate, 46)

        facts = self._facts_for_item(item)
        subject = self._public_subject_text(
            str(facts.get("who") or item.get("source_detail") or item.get("display_topic") or ""),
            "这条进展",
        )
        raw_target = str(facts.get("target") or item.get("summary_preview") or item.get("display_topic") or item.get("category") or "")
        if item.get("content_type") == "paper":
            subject = self._paper_short_name(item, facts)
            target = self._paper_translate_english_target(raw_target)
            return self._trim_reason(f"{subject}：{target}", 46)

        target = self._public_subject_text(raw_target, str(item.get("display_topic") or "产品或技术变化"))
        action = str(facts.get("action") or "").strip()
        if not action or self._mostly_english(action):
            action = "更新"
        return self._trim_reason(self._fact_phrase(subject, action, target), 46)

    def _classic_learning_nav(self, classic_items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for item in classic_items:
            buckets.setdefault(self._classic_nav_domain_key(item), []).append(item)
        result: List[Dict[str, str]] = []
        for domain_key in self.LEARNING_DOMAIN_ORDER:
            items = buckets.get(domain_key, [])
            if not items:
                continue
            first = self._classic_nav_representative(domain_key, items)
            result.append(
                {
                    "label": self._domain_label(domain_key),
                    "count": str(len(items)),
                    "anchor_id": str(first.get("anchor_id") or ""),
                    "takeaway": self._classic_nav_focus(domain_key),
                    "lead_title": self._classic_nav_lead_title(first),
                }
            )
        return result

    def _classic_read_time(self, item: Dict[str, Any]) -> str:
        if item.get("evidence_tier") != "high" or item.get("report_section") == "brief":
            return "30秒"
        detail_text = " ".join(
            str(item.get(key, "") or "")
            for key in ("summary_display", "paper_technical_intro", "why_now", "expected_effect", "future_impact")
        )
        if item.get("content_type") == "paper" or len(detail_text) >= 260:
            return "2分钟"
        return "1分钟"

    def _classic_mixed_items(self, all_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        high_items = [item for item in all_items if item.get("evidence_tier") == "high"]
        brief_items = [item for item in all_items if item.get("evidence_tier") != "high"]
        sorted_items = self._sort_items(high_items) + self._sort_items(brief_items)
        mixed: List[Dict[str, Any]] = []
        seen = set()
        for item in sorted_items:
            url = str(item.get("url") or "")
            title = str(item.get("title_cn") or item.get("title") or "")
            key = url or title
            if key in seen:
                continue
            seen.add(key)
            card = dict(item)
            card["order"] = len(mixed) + 1
            card["anchor_id"] = f"classic-{len(mixed) + 1:02d}"
            card["classic_analysis_items"] = self._classic_analysis_items(card)
            priority = self._classic_priority(card)
            card["classic_priority_label"] = priority["label"]
            card["classic_priority_class"] = priority["class"]
            card["classic_read_time"] = self._classic_read_time(card)
            mixed.append(card)
        return mixed

    def _v8_trim(self, value: Any, limit: int) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if len(text) <= limit:
            return text
        clipped = text[:limit]
        sentence_end = max(clipped.rfind("。"), clipped.rfind("！"), clipped.rfind("？"))
        if sentence_end >= max(30, int(limit * 0.62)):
            return clipped[: sentence_end + 1]
        clause_end = max(clipped.rfind("；"), clipped.rfind(";"), clipped.rfind("，"), clipped.rfind(","))
        if clause_end >= max(30, int(limit * 0.55)):
            return clipped[:clause_end].rstrip("，,；;：: ") + "。"
        return clipped.rstrip("，,；;：:。 ") + "。"

    @staticmethod
    def _primary_section(item: Dict[str, Any]) -> str:
        facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
        section = str(item.get("primary_section") or facts.get("primary_section") or "").strip().lower()
        if section in {"news", "technical", "paper"}:
            return section
        if str(item.get("content_type") or "").strip().lower() == "paper":
            return "paper"
        if str(item.get("content_type") or "").strip().lower() in {"project", "open_source", "opensource"}:
            return "technical"
        return "news"

    def _v8_card(self, item: Dict[str, Any]) -> Dict[str, Any]:
        card = dict(item)
        card["v11_item_key"] = editorial_item_render_key(card)
        public_title = str(card.get("title_cn") or card.get("title") or "").strip()
        curated_copy = str(card.get("model_used") or "") == "codex-automation"
        curated_title_limit = 56
        if not curated_copy and (
            self._classic_nav_title_needs_repair(public_title)
            or any(marker in public_title for marker in ("…", "..."))
            or len(public_title) > curated_title_limit
        ):
            card["title_cn"] = self._classic_nav_lead_title(card)
        repaired_title = str(card.get("title_cn") or "").strip()
        if not curated_copy and (
            self._classic_nav_title_needs_repair(repaired_title)
            or any(marker in repaired_title for marker in ("…", "..."))
            or len(repaired_title) > curated_title_limit
        ):
            if card.get("content_type") == "paper":
                facts = self._facts_for_item(card)
                paper_name = self._paper_short_name(card, facts)
                paper_name = re.sub(r"\s+", " ", str(paper_name or "")).strip()
                if len(paper_name) > 22:
                    acronym = re.search(r"\b[A-Z][A-Za-z0-9$^+._-]{2,18}\b", paper_name)
                    paper_name = acronym.group(0) if acronym else "本篇论文"
                card["title_cn"] = f"{paper_name or '本篇论文'}：{self._domain_label(card)}方法与实验"
            else:
                card["title_cn"] = self._classic_nav_lead_title({**card, "editorial_title": ""})
        news_limit = int(self.report_config.get("news_body_char_limit", 180) or 180)
        paper_limit = int(self.report_config.get("paper_body_char_limit", 220) or 220)
        is_paper = card.get("content_type") == "paper"
        body = card.get("paper_technical_intro") if is_paper else card.get("analysis_body") or card.get("summary_display")
        if not is_paper:
            body = re.split(r"(?:。)?(?:直接证据|验证信息|公开论据)是[:：]", str(body or ""), maxsplit=1)[0].rstrip("。 ") + "。"
        card["v8_body"] = self._v8_trim(body, paper_limit if is_paper else news_limit)
        card["v11_section"] = self._primary_section(card)
        v11_limit = news_limit
        if card["v11_section"] == "technical":
            v11_limit = int(self.report_config.get("technical_body_char_limit", 400) or 400)
        elif str(card.get("content_type") or "").lower() in {"interview", "podcast", "video"}:
            v11_limit = int(self.report_config.get("interview_body_char_limit", 500) or 500)
        v11_body = (
            card.get("analysis_body")
            or card.get("summary_display")
            or card.get("summary")
            or card.get("source_excerpt")
            or card.get("brief_line")
        )
        if str(card.get("model_used") or "") == "codex-automation":
            card["v11_body"] = str(v11_body or "").strip()
        else:
            card["v11_body"] = self._v8_trim(v11_body, v11_limit)
        deck = self._v8_trim(card.get("summary_preview") or card.get("editorial_lead"), 110)
        normalized_deck = re.sub(r"\W+", "", deck).lower()
        normalized_title = re.sub(r"\W+", "", str(card.get("title_cn") or "")).lower()
        normalized_opening = re.sub(r"\W+", "", card["v11_body"][:160]).lower()
        deck_repeats_title = bool(
            normalized_deck
            and normalized_title
            and (
                normalized_deck in normalized_title
                or normalized_title in normalized_deck
                or SequenceMatcher(None, normalized_deck, normalized_title).ratio() >= 0.78
            )
        )
        card["v12_deck"] = "" if normalized_deck and (
            normalized_deck in normalized_opening or deck_repeats_title
        ) else deck
        raw_facts = card.get("facts") if isinstance(card.get("facts"), dict) else {}
        source_excerpt = raw_facts.get("source_excerpt") or card.get("source_excerpt")
        evidence_locator = raw_facts.get("evidence_locator") or card.get("evidence_locator")
        if curated_copy:
            card["v11_source_excerpt"] = str(source_excerpt or "").strip()
            card["v11_evidence_locator"] = str(evidence_locator or "").strip()
        else:
            card["v11_source_excerpt"] = self._v8_trim(source_excerpt, 220)
            card["v11_evidence_locator"] = self._v8_trim(evidence_locator, 100)
        publish_date = str(card.get("publish_date") or "").strip()[:10]
        supplemental = "supplemental_older_source" in set(card.get("quality_flags") or [])
        card["v11_date_label"] = (
            f"补充阅读 · {publish_date}" if supplemental and publish_date else publish_date
        )
        claim_type = str(card.get("claim_type") or raw_facts.get("claim_type") or "").strip().lower()
        card["v11_claim_label"] = {
            "verified_fact": "可核验事实",
            "official_claim": "发布方声明",
            "interview_opinion": "受访者观点",
            "analysis": "分析判断",
            "research_result": "研究结果",
        }.get(claim_type, "")
        card["v8_evidence"] = self._v8_trim(card.get("evidence_line"), 150)
        card["v8_brief_line"] = self._v8_trim(card.get("brief_line"), 120)
        card["v8_domain"] = (
            self._v11_domain_label(card)
            if self._is_v11_product()
            else self._domain_label(card)
        )
        card["v8_source"] = str(card.get("source_detail") or card.get("platform") or card.get("source_tier_label") or "来源")
        card["v8_freshness"] = str(card.get("paper_status_label") or card.get("freshness_label") or "今日新增")
        card["v10_change_reason"] = self._v8_trim(card.get("paper_change_reason"), 110)
        if curated_copy:
            card["v10_plain_summary"] = str(
                card.get("paper_plain_summary") or ""
            ).strip()
            card["v10_technical_intro"] = str(
                card.get("paper_technical_intro") or ""
            ).strip()
        else:
            card["v10_plain_summary"] = self._v8_trim(
                card.get("paper_plain_summary"), 180
            )
            card["v10_technical_intro"] = self._v8_trim(
                card.get("paper_technical_intro"), 300
            )
        card["v10_index_only"] = str(card.get("summary_quality_tier") or "") == "index_only"
        card["v10_source_title"] = self._v8_trim(
            card.get("source_display_title") or card.get("title") or card.get("title_cn"),
            180,
        )
        card["v10_source_excerpt"] = self._v8_trim(
            card.get("source_excerpt") or card.get("brief_line"),
            280,
        )
        appendix_text = (
            card.get("paper_compact_summary")
            or card.get("paper_plain_summary")
            or card.get("paper_technical_intro")
            or card.get("editorial_lead")
        )
        card["v10_appendix_summary"] = self._v8_trim(appendix_text, 88)
        card["v9_delta"] = self._v8_trim(card.get("delta_summary"), 190)
        card["v9_previous"] = self._v8_trim(card.get("previous_conclusion"), 150)
        card["v9_confidence"] = str(card.get("confidence_label") or "")
        card["v9_confidence_reason"] = self._v8_trim(card.get("confidence_reason"), 150)
        card["v9_lineage"] = self._v8_trim(card.get("technical_lineage"), 160)
        card["v9_difference"] = self._v8_trim(card.get("difference_from_prior"), 160)
        card["v9_judgment"] = self._v8_trim(card.get("judgment_effect"), 170)
        card["v9_related"] = dict(card.get("related_reading") or {})
        card["v8_feedback"] = {
            key: value
            for key, value in (card.get("primary_feedback_links") or {}).items()
            if key in {"useful", "track"}
        }
        return card

    def _v8_reader_context(self, layers: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        must_read = [self._v8_card(item) for item in layers.get("must_read", [])[:5]]
        domain_candidates = [self._v8_card(item) for item in layers.get("watch", [])]
        featured_papers = [self._v8_card(item) for item in layers.get("featured_papers", [])[:8]]
        featured_urls = {str(item.get("canonical_url") or item.get("url") or "") for item in featured_papers}
        paper_index = [
            self._v8_card(item)
            for item in layers.get("paper_appendix", [])
            if str(item.get("canonical_url") or item.get("url") or "") not in featured_urls
        ]
        domain_sections: List[Dict[str, Any]] = []
        domain_limit = max(1, int(self.report_config.get("domain_item_limit", 2) or 2))
        for domain_key in self.LEARNING_DOMAIN_ORDER:
            items = [item for item in domain_candidates if self._domain_key(item) == domain_key][:domain_limit]
            paper_links = [item for item in featured_papers if self._domain_key(item) == domain_key][:3]
            domain_sections.append(
                {
                    "key": domain_key,
                    "anchor_id": f"domain-{domain_key.replace('_', '-')}",
                    "label": self._domain_label(domain_key),
                    "entries": items,
                    "paper_links": paper_links,
                }
            )
        briefs = [self._v8_card(item) for item in layers.get("brief", [])[:8]]

        memory_candidates = must_read + featured_papers
        memory = []
        for item in memory_candidates:
            text = self._v8_trim(item.get("editorial_lead") or item.get("title_cn"), 80)
            if not text or text in {entry["text"] for entry in memory}:
                continue
            memory.append({"text": text, "url": item.get("url", ""), "domain": item.get("v8_domain", "")})
            if len(memory) >= 3:
                break
        return {
            "memory": memory,
            "anchors": [section for section in domain_sections],
            "must_read": must_read,
            "domains": domain_sections,
            "featured_papers": featured_papers,
            "paper_index": paper_index,
            "briefs": briefs,
            "topic_dossiers": list((self.report_config.get("v9_context") or {}).get("topic_dossiers") or []),
            "reading_queue": list((self.report_config.get("v9_context") or {}).get("reading_queue") or []),
            "closing_memory": dict((self.report_config.get("v9_context") or {}).get("closing_memory") or {}),
            "weekly_digest": dict((self.report_config.get("v9_context") or {}).get("weekly_digest") or {}),
            "show_weekly_digest": bool((self.report_config.get("v9_context") or {}).get("show_weekly_digest", False)),
        }

    def _v10_reader_context(self, layers: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        def unique(items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
            result: List[Dict[str, Any]] = []
            seen = set()
            for raw in items:
                item = self._v8_card(raw)
                key = str(item.get("canonical_url") or item.get("url") or item.get("title_cn") or "")
                if not key or key in seen:
                    continue
                seen.add(key)
                result.append(item)
                if len(result) >= limit:
                    break
            return result

        must_read_limit = max(1, int(self.report_config.get("must_read_limit", 10) or 10))
        all_nonpaper = unique(
            [
                item
                for item in (
                    list(layers.get("must_read", []))
                    + list(layers.get("physical_ai", []))
                    + list(layers.get("watch", []))
                    + list(layers.get("brief", []))
                )
                if item.get("content_type") != "paper"
            ],
            200,
        )
        news_limit = max(20, int(self.report_config.get("news_section_limit", 30) or 30))
        technical_limit = max(20, int(self.report_config.get("technical_section_limit", 24) or 24))
        news_items = [item for item in all_nonpaper if self._primary_section(item) == "news"][:news_limit]
        technical_items = [item for item in all_nonpaper if self._primary_section(item) == "technical"][:technical_limit]
        must_read_candidates = [
            item
            for item in (
                list(layers.get("must_read", []))
                + list(layers.get("physical_ai", []))
                + list(layers.get("watch", []))
            )
            if item.get("content_type") != "paper" and str(item.get("quality_tier") or "") != "brief"
        ]
        must_read = unique(must_read_candidates, must_read_limit)
        must_read_urls = {str(item.get("canonical_url") or item.get("url") or "") for item in must_read}

        domain_limit = max(1, int(self.report_config.get("domain_item_limit", 4) or 4))
        domain_candidates = [
            item
            for item in (
                list(layers.get("must_read", []))
                + list(layers.get("physical_ai", []))
                + list(layers.get("watch", []))
            )
            if item.get("content_type") != "paper"
            and str(item.get("quality_tier") or "") != "brief"
            and str(item.get("canonical_url") or item.get("url") or "") not in must_read_urls
        ]
        domains: List[Dict[str, Any]] = []
        domain_urls: set[str] = set()
        for domain_key in self.LEARNING_DOMAIN_ORDER:
            entries = unique(
                [item for item in domain_candidates if self._domain_key(item) == domain_key],
                domain_limit,
            )
            if not entries:
                continue
            domain_urls.update(str(item.get("canonical_url") or item.get("url") or "") for item in entries)
            domains.append(
                {
                    "key": domain_key,
                    "anchor_id": f"domain-{domain_key.replace('_', '-')}",
                    "label": self._domain_label(domain_key),
                    "entries": entries,
                }
            )

        minimum_information_count = max(
            must_read_limit,
            int(self.report_config.get("min_visible_information_count", 25) or 25),
        )
        visible_information_count = len(must_read) + sum(len(section["entries"]) for section in domains)
        more_update_limit = max(0, minimum_information_count - visible_information_count)
        more_updates = (
            unique(
                [
                    item
                    for item in domain_candidates
                    if str(item.get("canonical_url") or item.get("url") or "") not in domain_urls
                ],
                more_update_limit,
            )
            if more_update_limit
            else []
        )

        featured_limit = max(1, int(self.report_config.get("paper_featured_limit", 12) or 12))
        featured_papers_all = unique(
            [
                item
                for item in layers.get("featured_papers", [])
                if paper_plain_summary_passes(item.get("paper_plain_summary"))
                and paper_technical_intro_passes(item.get("paper_technical_intro"))
            ],
            featured_limit,
        )
        featured_urls = {str(item.get("canonical_url") or item.get("url") or "") for item in featured_papers_all}
        appendix_limit = max(0, int(self.report_config.get("paper_appendix_limit", 24) or 24))
        more_papers_all = unique(
            [
                item
                for item in layers.get("paper_appendix", [])
                if str(item.get("canonical_url") or item.get("url") or "") not in featured_urls
                and bool(
                    str(
                        item.get("paper_compact_summary")
                        or item.get("paper_plain_summary")
                        or item.get("paper_technical_intro")
                        or ("index-only" if str(item.get("summary_quality_tier") or "") == "index_only" else "")
                        or ""
                    ).strip()
                )
            ],
            appendix_limit,
        )
        paper_updates = [
            item for item in featured_papers_all + more_papers_all
            if bool(item.get("is_reappeared_update"))
        ]
        featured_papers = [item for item in featured_papers_all if not item.get("is_reappeared_update")]
        more_papers = [item for item in more_papers_all if not item.get("is_reappeared_update")]

        for index, item in enumerate(news_items, start=1):
            item["v11_position"] = index
            item["v11_total"] = len(news_items)
        for index, item in enumerate(technical_items, start=1):
            item["v11_position"] = index
            item["v11_total"] = len(technical_items)
        ordered_papers = paper_updates + featured_papers + more_papers
        for index, item in enumerate(ordered_papers, start=1):
            item["v11_position"] = index
            item["v11_total"] = len(ordered_papers)

        decision_limit = min(
            7,
            max(5, int(self.report_config.get("editorial_decision_limit", 6) or 6)),
        )
        decision_candidates = unique(
            must_read + news_items + technical_items + featured_papers + more_papers,
            max(decision_limit * 3, decision_limit),
        )
        decisions: List[Dict[str, Any]] = []
        seen_decision_domains: set[str] = set()
        deferred_decisions: List[Dict[str, Any]] = []
        for item in decision_candidates:
            lead = self._v8_trim(item.get("editorial_lead") or item.get("analysis_body"), 92)
            evidence = self._v8_trim(item.get("evidence_line"), 108)
            if not lead:
                continue
            decision = {
                "domain": item.get("v8_domain") or self._domain_label(item),
                "text": lead,
                "evidence": evidence,
                "url": item.get("url", ""),
                "source_identity": editorial_source_identity(item),
            }
            if decision["domain"] in seen_decision_domains:
                deferred_decisions.append(decision)
                continue
            decisions.append(decision)
            seen_decision_domains.add(decision["domain"])
            if len(decisions) >= decision_limit:
                break
        if len(decisions) < decision_limit:
            existing_urls = {str(item.get("url") or "") for item in decisions}
            for decision in deferred_decisions:
                url = str(decision.get("url") or "")
                if url in existing_urls:
                    continue
                decisions.append(decision)
                existing_urls.add(url)
                if len(decisions) >= decision_limit:
                    break

        source_news_limit = max(0, int(self.report_config.get("source_news_brief_limit", 12) or 12))
        source_news_briefs = unique(
            [item for item in layers.get("brief", []) if item.get("source_grounded_brief")],
            source_news_limit,
        )

        return {
            "editorial_decisions": decisions,
            "must_read": must_read,
            "domains": domains,
            "more_updates": more_updates,
            "paper_updates": paper_updates,
            "featured_papers": featured_papers,
            "more_papers": more_papers,
            "briefs": source_news_briefs,
            "news_items": news_items,
            "technical_items": technical_items,
            "news_count": len(news_items),
            "technical_count": len(technical_items),
            "paper_count": len(paper_updates) + len(featured_papers) + len(more_papers),
            "information_count": len(news_items) + len(technical_items),
            "paper_freshness": dict(self.report_config.get("paper_freshness_metrics") or {}),
        }

    def generate_html(
        self,
        papers: List[Dict[str, Any]],
        updates: List[Dict[str, Any]],
        mixed_items: List[Dict[str, Any]],
        report_summary: Dict[str, Any],
        title: str = "AI Daily Report",
        collector_summary: Optional[Dict[str, Any]] = None,
        trend_summary: Optional[Dict[str, Any]] = None,
        alert_summary: Optional[Dict[str, Any]] = None,
        layered_updates: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        archive_summary: Optional[Dict[str, Any]] = None,
    ) -> str:
        decorated_layers = self._decorate_layers(layered_updates)
        if any(decorated_layers.values()):
            top_highlights = self._decorate_highlights(decorated_layers["must_read"])
            body_items = (
                decorated_layers["physical_ai"]
                + decorated_layers["watch"]
                + decorated_layers["featured_papers"]
                + decorated_layers["paper_appendix"]
                + decorated_layers["brief"]
            )
        else:
            card_items = self._decorate_items(mixed_items, 0, "mix")
            top_highlights = self._decorate_highlights(self._pick_top_highlights(card_items))
            body_items = self._exclude_highlight_duplicates(card_items, top_highlights)
            physical_ai_items = [
                item
                for item in body_items
                if item.get("content_type") != "paper" and item.get("evidence_tier") == "high" and self._is_physical_ai_item(item)
            ][:6]
            physical_ai_urls = {str(item.get("url", "")) for item in physical_ai_items if item.get("url")}
            decorated_layers = {
                "must_read": top_highlights,
                "physical_ai": physical_ai_items,
                "watch": [
                    item
                    for item in body_items
                    if item.get("content_type") != "paper"
                    and item.get("evidence_tier") == "high"
                    and str(item.get("url", "")) not in physical_ai_urls
                ],
                "featured_papers": [item for item in body_items if item.get("content_type") == "paper"][:6],
                "paper_appendix": [item for item in body_items if item.get("content_type") == "paper"][6:18],
                "brief": [item for item in body_items if item.get("evidence_tier") != "high"],
            }

        all_layered_items = (
            decorated_layers.get("must_read", [])
            + decorated_layers.get("physical_ai", [])
            + decorated_layers.get("watch", [])
            + decorated_layers.get("featured_papers", [])
            + decorated_layers.get("paper_appendix", [])
            + decorated_layers.get("brief", [])
        )

        tracking_items = self._build_tracking_items(
            decorated_layers.get("must_read", [])
            + decorated_layers.get("physical_ai", [])
            + decorated_layers.get("watch", [])
            + decorated_layers.get("featured_papers", [])
            + decorated_layers.get("paper_appendix", [])
        )

        editorial_decisions = self._build_editorial_decisions(all_layered_items)
        if not editorial_decisions:
            fallback_decision = str(report_summary.get("lead_summary") or self._hero_description(papers, updates, report_summary) or "").strip()
            if fallback_decision:
                editorial_decisions = [{"domain": "编辑判断", "text": self._trim_reason(fallback_decision, 110), "url": ""}]
        design_version = self.design_version
        classic_mode = self._is_classic_briefing()
        v8_mode = self._is_v8_reader()
        v9_mode = self._is_v9_reader()
        v10_mode = self._is_v10_reader()
        domain_sections = [] if classic_mode else self._build_domain_sections(all_layered_items)
        classic_mixed_items = self._classic_mixed_items(all_layered_items) if classic_mode else []
        classic_learning_nav = self._classic_learning_nav(classic_mixed_items) if classic_mode else []

        v11_product = self._is_v11_product()
        display_title = "AI 前沿情报日报" if v11_product else title
        display_subtitle = ""
        if v11_product and "·" in title:
            display_subtitle = title.split("·", 1)[1].strip()
        v10_context = self._v10_reader_context(decorated_layers) if v10_mode else {}
        edition_counts = report_summary.get("edition_counts") if isinstance(report_summary, dict) else None
        if v10_context:
            edition_counts = edition_counts if isinstance(edition_counts, dict) else {}
            v10_context.update({
                "edition_news_count": int(edition_counts.get("news", v10_context.get("news_count", 0)) or 0),
                "edition_technical_count": int(edition_counts.get("technical", v10_context.get("technical_count", 0)) or 0),
                "edition_paper_count": int(edition_counts.get("paper", v10_context.get("paper_count", 0)) or 0),
            })
            v10_context["edition_total_count"] = (
                v10_context["edition_news_count"]
                + v10_context["edition_technical_count"]
                + v10_context["edition_paper_count"]
            )
        context = {
            "title": title,
            "display_title": display_title,
            "display_subtitle": display_subtitle,
            "design_version": design_version,
            "classic_mode": classic_mode,
            "v8_mode": v8_mode,
            "v9_mode": v9_mode,
            "v10_mode": v10_mode,
            "v8": self._v8_reader_context(decorated_layers) if v8_mode and not v10_mode else {},
            "v10": v10_context,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "paper_count": len(papers),
            "update_count": len(updates),
            "hero_description": "" if v8_mode else self._hero_description(papers, updates, report_summary),
            "hero_scope": [] if v8_mode else self._hero_scope(papers, updates, report_summary),
            "summary_lines": self._summary_lines(report_summary),
            "learning_map": [] if classic_mode or v8_mode or design_version == "v6-editorial-learning" else self._build_learning_map(all_layered_items),
            "classic_overview": self._classic_overview(report_summary, all_layered_items) if classic_mode else {},
            "classic_mixed_items": classic_mixed_items,
            "classic_learning_nav": classic_learning_nav,
            "editorial_decisions": [] if classic_mode or v8_mode else editorial_decisions,
            "today_first_reads": [] if classic_mode or v8_mode else self._build_today_first_reads(
                decorated_layers.get("must_read", [])
                + decorated_layers.get("physical_ai", [])
                + decorated_layers.get("watch", [])
                + decorated_layers.get("featured_papers", [])
            ),
            "paper_technical_digest_items": [] if v8_mode else self._paper_technical_digest_items(
                decorated_layers.get("featured_papers", []) + decorated_layers.get("paper_appendix", [])
            ),
            "editor_judgement": self._editor_judgement(all_layered_items),
            "top_highlights": [] if v8_mode else top_highlights,
            "card_items": body_items,
            "layered_sections": decorated_layers,
            "must_read_items": [] if v8_mode else decorated_layers.get("must_read", []),
            "physical_ai_items": [] if v8_mode else decorated_layers.get("physical_ai", []),
            "physical_ai_radar_groups": self._physical_ai_radar_groups(decorated_layers.get("physical_ai", [])),
            "watch_items": [] if v8_mode else decorated_layers.get("watch", []),
            "featured_papers": [] if v8_mode else decorated_layers.get("featured_papers", []),
            "paper_appendix": [] if v8_mode else decorated_layers.get("paper_appendix", []),
            "research_groups": self._build_research_groups(decorated_layers.get("featured_papers", [])),
            "brief_items": [] if v8_mode else decorated_layers.get("brief", []),
            "tracking_items": [] if v8_mode else tracking_items,
            "dashboard_items": [] if v8_mode or design_version == "v6-editorial-learning" else self._build_dashboard(decorated_layers, report_summary),
            "decision_entry": self._decision_entry(decorated_layers, report_summary),
            "change_map": [] if v8_mode else self._build_change_map(all_layered_items),
            "domain_sections": [] if v8_mode else domain_sections,
            "paper_appendix_groups": [] if v8_mode else self._paper_appendix_groups(decorated_layers.get("paper_appendix", [])),
            "world_model_focus_groups": [] if v8_mode else self._world_model_focus_groups(all_layered_items),
            "quality_footnote": self._build_quality_footnote(decorated_layers),
            "report_summary": report_summary,
            "collector_summary": collector_summary or {},
            "trend_summary": {"items": []} if v8_mode else (trend_summary or {}),
            "alert_summary": alert_summary or {},
            "archive_summary": archive_summary or {},
        }
        template = self.env.get_template("daily_report.html")
        rendered = template.render(**context)
        return self._inline_v11_critical_styles(rendered) if v11_product else rendered

    def generate_markdown(
        self,
        papers: List[Dict[str, Any]],
        updates: List[Dict[str, Any]],
        mixed_items: List[Dict[str, Any]],
        report_summary: Dict[str, Any],
        title: str = "AI Daily Report",
        collector_summary: Optional[Dict[str, Any]] = None,
        trend_summary: Optional[Dict[str, Any]] = None,
        alert_summary: Optional[Dict[str, Any]] = None,
        layered_updates: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        archive_summary: Optional[Dict[str, Any]] = None,
    ) -> str:
        trend_summary = trend_summary or {}
        alert_summary = alert_summary or {}
        archive_summary = archive_summary or {}

        decorated_layers = self._decorate_layers(layered_updates)
        if any(decorated_layers.values()):
            top_highlights = self._decorate_highlights(decorated_layers["must_read"])
            body_items = (
                decorated_layers["physical_ai"]
                + decorated_layers["watch"]
                + decorated_layers["featured_papers"]
                + decorated_layers["paper_appendix"]
                + decorated_layers["brief"]
            )
        else:
            card_items = self._decorate_items(mixed_items, 0, "mix")
            top_highlights = self._decorate_highlights(self._pick_top_highlights(card_items))
            body_items = self._exclude_highlight_duplicates(card_items, top_highlights)

        all_layered_items = (
            decorated_layers.get("must_read", [])
            + decorated_layers.get("physical_ai", [])
            + decorated_layers.get("watch", [])
            + decorated_layers.get("featured_papers", [])
            + decorated_layers.get("paper_appendix", [])
            + decorated_layers.get("brief", [])
        )

        if self._is_v10_reader():
            v10 = self._v10_reader_context(decorated_layers)
            lines = [f"# {title} - {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
            lines.append("## 今日编辑判断")
            for item in v10.get("editorial_decisions", []):
                lines.append(f"- **{item.get('domain')}** {item.get('text')}")
                if item.get("evidence"):
                    lines.append(f"  {item.get('evidence')}")
            lines.append("")
            if v10.get("must_read"):
                lines.append("## 今日重点情报")
                for article in v10["must_read"]:
                    lines.append(f"### {article.get('title_cn')}")
                    lines.append(str(article.get("v8_body") or ""))
                    if article.get("v8_evidence"):
                        lines.append(f"> {article.get('v8_evidence')}")
                    lines.append(f"[查看原始来源]({article.get('url')})")
                    lines.append("")
            for domain in v10.get("domains", []):
                lines.append(f"## {domain.get('label')}")
                for article in domain.get("entries", []):
                    lines.append(f"### {article.get('title_cn')}")
                    lines.append(str(article.get("v8_body") or ""))
                    if article.get("v8_evidence"):
                        lines.append(f"> {article.get('v8_evidence')}")
                    lines.append(f"[查看原始来源]({article.get('url')})")
                    lines.append("")
            if v10.get("paper_updates"):
                lines.append("## 重要进展更新")
                for article in v10["paper_updates"]:
                    lines.append(f"### [{article.get('v8_freshness')}] {article.get('title_cn')}")
                    if article.get("v10_change_reason"):
                        lines.append(f"> 本次变化：{article.get('v10_change_reason')}")
                    lines.append(str(article.get("v10_plain_summary") or ""))
                    lines.append("")
                    lines.append(str(article.get("v10_technical_intro") or ""))
                    lines.append(f"[查看论文]({article.get('url')})")
                    lines.append("")
            if v10.get("featured_papers"):
                lines.append("## 论文精读")
                for article in v10["featured_papers"]:
                    lines.append(f"### {article.get('title_cn')}")
                    lines.append(str(article.get("v10_plain_summary") or ""))
                    lines.append("")
                    lines.append(str(article.get("v10_technical_intro") or ""))
                    if article.get("v8_evidence"):
                        lines.append(f"> {article.get('v8_evidence')}")
                    lines.append(f"[查看论文]({article.get('url')})")
                    lines.append("")
            if v10.get("more_papers"):
                lines.append("## 更多论文")
                for article in v10["more_papers"]:
                    lines.append(f"- **[{article.get('v8_domain')}] [{article.get('title_cn')}]({article.get('url')})**")
                    lines.append(f"  {article.get('v10_appendix_summary')}")
            return "\n".join(lines)

        lines = [f"# {title} - {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
        lines.append(self._hero_description(papers, updates, report_summary))
        lines.append("")

        hero_scope = self._hero_scope(papers, updates, report_summary)
        if hero_scope:
            lines.append("范围: " + " | ".join(hero_scope))
            lines.append("")

        if alert_summary.get("needs_alert"):
            lines.append("## 异常提醒")
            for issue in alert_summary.get("issues", []):
                lines.append(f"- {issue}")
            lines.append("")

        if self._is_classic_briefing():
            overview = self._classic_overview(report_summary, all_layered_items)
            lines.append("## 今日总览")
            if overview.get("overall"):
                lines.append(f"### 总体判断\n{overview['overall']}\n")
            if overview.get("paper_trend"):
                lines.append(f"### 论文趋势\n{overview['paper_trend']}\n")
            if overview.get("update_trend"):
                lines.append(f"### 全网动态趋势\n{overview['update_trend']}\n")
            if overview.get("hot_topics"):
                lines.append("### 热点标签")
                lines.append("、".join(overview["hot_topics"]))
                lines.append("")
            if overview.get("key_takeaways"):
                lines.append("### 关键结论")
                for item in overview["key_takeaways"]:
                    lines.append(f"- {item}")
                lines.append("")
            if overview.get("watchlist"):
                lines.append("### 后续观察")
                for item in overview["watchlist"]:
                    lines.append(f"- {item}")
                lines.append("")

            lines.append("## 混排情报流")
            for article in self._classic_mixed_items(all_layered_items):
                lines.append(f"### {article['order']}. {article.get('title_cn')}")
                if article.get("summary_preview"):
                    lines.append(article["summary_preview"])
                lines.append(f"- 类型: {article.get('content_kind')} | 主题: {article.get('display_topic')} | Score: {article.get('score')}")
                lines.append(f"- 摘要: {article.get('summary_display') or article.get('summary')}")
                if article.get("content_type") == "paper" and article.get("paper_technical_intro"):
                    lines.append(f"- 技术介绍: {article.get('paper_technical_intro')}")
                if article.get("why_it_matters_display"):
                    lines.append(f"- 为什么值得看: {article.get('why_it_matters_display')}")
                for detail in article.get("classic_analysis_items") or []:
                    lines.append(f"- {detail.get('label')}: {detail.get('value')}")
                lines.append(f"- 原文: {article.get('url')}")
                lines.append("")
            return "\n".join(lines)

        learning_map = self._build_learning_map(all_layered_items)
        if learning_map:
            lines.append("## 今日学习地图")
            for item in learning_map:
                lines.append(f"- [{item['domain']}] {item['takeaway']} | {item['url']}")
            lines.append("")

        lines.append("## 今天必须记住的 3-5 件事")
        lines.append(report_summary.get("lead_summary", ""))
        lines.append("")

        lines.append("## 论文趋势")
        lines.append(report_summary.get("paper_summary", ""))
        lines.append("")

        lines.append("## 全网动态趋势")
        lines.append(report_summary.get("update_summary", ""))
        lines.append("")

        hot_topics = report_summary.get("hot_topics") or []
        if hot_topics:
            lines.append("## 热点标签")
            lines.append("、".join(hot_topics))
            lines.append("")

        key_takeaways = report_summary.get("key_takeaways") or []
        if key_takeaways:
            lines.append("## 关键结论")
            for item in key_takeaways:
                lines.append(f"- {item}")
            lines.append("")

        watchlist = report_summary.get("watchlist") or []
        if watchlist:
            lines.append("## 后续观察")
            for item in watchlist:
                lines.append(f"- {item}")
            lines.append("")

        trend_items = trend_summary.get("items") or []
        if trend_items:
            lines.append("## 趋势追踪")
            for item in trend_items:
                label = item.get("label") or item.get("name") or "趋势"
                summary = item.get("summary") or ""
                lines.append(f"- {label}: {summary}")
            lines.append("")

        domain_sections = self._build_domain_sections(all_layered_items)
        if domain_sections:
            lines.append("## 领域学习导航")
            for section in domain_sections:
                lines.append(f"- {section['label']}：{section['today_conclusion']}")
            lines.append("")
            for section in domain_sections:
                lines.append(f"## {section['label']}")
                lines.append(section["learning_focus"])
                lines.append("")
                for article in section["focus_items"]:
                    lines.append(f"### {article.get('title_cn')}")
                    lines.append(str(article.get("editorial_brief") or article.get("technical_context") or article.get("learning_takeaway") or ""))
                    lines.append(f"依据: {article.get('background_context')}")
                    lines.append(f"继续深挖: {article.get('deep_dive_prompt')}")
                    lines.append(f"- 原文: {article.get('url')}")
                    lines.append("")
                if section["papers"]:
                    lines.append("### 论文 / 技术")
                    for article in section["papers"]:
                        lines.append(f"- {article.get('title_cn')}：{article.get('paper_technical_intro') or article.get('technical_context')} | {article.get('url')}")
                    lines.append("")
                if section["reading"]:
                    lines.append("### 延伸阅读")
                    for article in section["reading"]:
                        lines.append(f"- {article.get('title_cn')}：{article.get('summary_preview') or article.get('brief_line')} | {article.get('url')}")
                    lines.append("")
        else:
            self._render_card_block(lines, decorated_layers.get("physical_ai", []), "具身智能 / Physical AI", "单独跟踪机器人、VLA、人形机器人和真实环境落地。")
            self._render_card_block(lines, decorated_layers.get("watch", []), "观察列表", "值得跟进但证据或影响仍需确认。")
            self._render_card_block(lines, decorated_layers.get("featured_papers", []), "论文精选", "前 6 篇保留技术介绍、证据和深挖线索。")
        appendix = decorated_layers.get("paper_appendix", [])
        if appendix:
            lines.append("## 论文附录")
            for article in appendix:
                lines.append(f"- {article.get('title_cn')}：{article.get('brief_line') or article.get('summary_preview')} | {article.get('url')}")
            lines.append("")
        self._render_card_block(lines, decorated_layers.get("brief", []), "快讯 / 待确认", "低证据、聚合源或短新闻只保留事实短句。")

        if archive_summary.get("entries"):
            lines.append("## 历史归档")
            for entry in archive_summary["entries"][:10]:
                lines.append(f"- {entry['label']} | HTML: {entry['html_path']} | Markdown: {entry['markdown_path']}")
            lines.append("")

        return "\n".join(lines)
